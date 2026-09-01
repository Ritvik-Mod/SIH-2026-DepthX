"""HeightNet = DINOv2 encoder + DPT neck (both inherited from Depth Anything V2)
+ new metric height head, with optional GSD conditioning, semantics and uncertainty.

What transfers and what does not:
  backbone  keep   general visual features -- roofs, roads, trees, shadow edges
  neck      keep structure, retrain   coarse-to-fine fusion is right, weights are depth-tuned
  head      discard                   encodes perspective geometry and affine invariance
"""
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModelForDepthEstimation

from .heads import FiLM, BinsHeightHead, RegressionHeightHead, SemanticHead, UncertaintyHead


class HeightNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        da = AutoModelForDepthEstimation.from_pretrained(cfg.model.checkpoint)
        self.backbone = da.backbone
        self.neck = da.neck
        del da.head                                     # the part we are replacing

        self.patch_size = int(getattr(da.config, "patch_size", 14))
        ch = int(da.config.fusion_hidden_size)
        self.cfg = cfg

        self.film = FiLM(ch, gsd_ref=float(cfg.data.gsd_base)) if cfg.model.use_gsd else None
        if cfg.model.head == "bins":
            self.height_head = BinsHeightHead(ch, int(cfg.model.n_bins), float(cfg.model.h_max))
        else:
            self.height_head = RegressionHeightHead(ch, float(cfg.model.h_max))
        self.sem_head = SemanticHead(ch, int(cfg.model.n_classes)) if cfg.model.use_semantic else None
        self.unc_head = UncertaintyHead(ch) if cfg.model.use_uncertainty else None
        self._freeze_structurally_dead()

    def _freeze_structurally_dead(self):
        """Some inherited parameters are never reached by this forward pass:

          embeddings.mask_token   used only for masked-image-modelling pretraining
          fusion_stage.layers.0.residual_layer1
                                  DPT fuses coarse-to-fine, and the FIRST (deepest)
                                  fusion layer has no coarser input to add, so its
                                  first residual unit is unreachable by construction

        Freezing them explicitly keeps 'every trainable parameter must receive a
        gradient' a meaningful assertion instead of one with silent exceptions."""
        dead = 0
        for n, p in self.named_parameters():
            if n.endswith("embeddings.mask_token") or "fusion_stage.layers.0.residual_layer1" in n:
                p.requires_grad_(False)
                dead += 1
        self._n_dead = dead

    # ---- feature extraction -------------------------------------------------
    def features(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(f"input {H}x{W} must be a multiple of patch size {self.patch_size}")
        ph, pw = H // self.patch_size, W // self.patch_size
        fmaps = self.backbone(x).feature_maps          # 4 x (B, 1+N(+regs), D) token seqs
        fused = self.neck(list(fmaps), ph, pw)         # 4 maps, finest last
        return fused[-1]

    def forward(self, x: torch.Tensor, gsd: torch.Tensor | None = None) -> dict:
        out_hw = x.shape[-2:]
        f = self.features(x)
        if self.film is not None:
            if gsd is None:
                raise ValueError("model.use_gsd=true but no gsd passed to forward()")
            f = self.film(f, gsd)
        out = self.height_head(f, out_hw)
        if self.sem_head is not None:
            out["semantic"] = self.sem_head(f, out_hw)
        if self.unc_head is not None:
            out["log_var"] = self.unc_head(f, out_hw)
            out["sigma"] = torch.exp(0.5 * out["log_var"])
        return out

    # ---- optimisation helpers ----------------------------------------------
    def param_groups(self, lr_encoder: float, lr_decoder: float, lr_head: float, weight_decay: float):
        """Three rates because the three parts know different amounts.  Norm and bias
        parameters are excluded from weight decay as is standard for ViT fine-tuning."""
        buckets = {"encoder": (self.backbone, lr_encoder),
                   "decoder": (self.neck, lr_decoder)}
        heads = nn.ModuleList([m for m in (self.film, self.height_head, self.sem_head, self.unc_head)
                               if m is not None])
        buckets["head"] = (heads, lr_head)
        groups = []
        for name, (mod, lr) in buckets.items():
            decay, no_decay = [], []
            for pn, p in mod.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if p.ndim <= 1 or pn.endswith(".bias") else decay).append(p)
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay, "name": f"{name}/decay"})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": f"{name}/no_decay"})
        return groups

    def set_encoder_trainable(self, flag: bool):
        for p in self.backbone.parameters():
            p.requires_grad_(flag)
        self._freeze_structurally_dead()   # re-apply: unfreezing the encoder would revive mask_token
