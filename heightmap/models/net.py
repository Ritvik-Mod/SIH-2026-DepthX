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
    # ---- block-level surgery ------------------------------------------------
    def n_blocks(self) -> int:
        return len(self.backbone.encoder.layer)

    @staticmethod
    def _parse_blocks(spec, n: int) -> set[int]:
        """'all' | 'none' | '0-7' | '16-23' | '0-3,20-23' -> a set of block indices.

        Indices are 0-based over the encoder's own layer list, so ViT-S is 0..11 and
        ViT-L is 0..23.  Note this is NOT the same numbering as the config's
        `out_indices` ([3,6,9,12] / [5,12,18,24]), which is 1-based over stages -- so
        DAv2-Large's finest DPT tap, "stage5", is block index 4 here.  Getting this
        wrong silently freezes the wrong end of the network, which is exactly the class
        of bug this project keeps finding.
        """
        s = str(spec).strip().lower()
        if s in ("all", "", "*"):
            return set(range(n))
        if s == "none":
            return set()
        keep: set[int] = set()
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                lo, hi = int(a), int(b)
            else:
                lo = hi = int(part)
            if not (0 <= lo <= hi < n):
                raise ValueError(f"block range '{part}' outside 0..{n-1} for this backbone")
            keep |= set(range(lo, hi + 1))
        return keep

    def set_trainable_blocks(self, spec, train_embed: bool = False):
        """Freeze every encoder block except those named.  Returns the kept set.

        The patch/pos embedding is frozen by default even when block 0 is trained: it is
        a single projection shared by every token, it is the most heavily pretrained
        thing in the model, and a small target set moves it in ways that do not
        generalise.  Pass train_embed=True to include it deliberately.
        """
        n = self.n_blocks()
        keep = self._parse_blocks(spec, n)
        for i, blk in enumerate(self.backbone.encoder.layer):
            for p in blk.parameters():
                p.requires_grad_(i in keep)
        for pn, p in self.backbone.named_parameters():
            if pn.startswith("embeddings"):
                p.requires_grad_(bool(train_embed))
        self._freeze_structurally_dead()      # mask_token must stay dead
        self._trainable_blocks = sorted(keep)
        self._train_embed = bool(train_embed)
        return self._trainable_blocks

    def set_neck_trainable(self, flag: bool):
        for p in self.neck.parameters():
            p.requires_grad_(flag)
        self._freeze_structurally_dead()

    def param_groups(self, lr_encoder: float, lr_decoder: float, lr_head: float,
                     weight_decay: float, llrd: float = 1.0):
        """Three rates because the three parts know different amounts.  Norm and bias
        parameters are excluded from weight decay as is standard for ViT fine-tuning.

        `llrd` is layerwise learning-rate decay over the encoder blocks:
        block i gets lr_encoder * llrd**(n-1-i), so the LAST block keeps lr_encoder and
        earlier blocks are scaled down.  llrd=1.0 reproduces the flat behaviour exactly,
        which is what A1-A7 trained with -- this is a strict generalisation, not a
        change of default.  Values around 0.65-0.75 are the usual ViT fine-tuning range.
        Set llrd>1 to invert it and move the EARLY blocks faster, which is what the
        surgical-fine-tuning result argues for under an input-level shift.
        """
        groups = []

        def add(name, named_params, lr):
            decay, no_decay = [], []
            for pn, p in named_params:
                if not p.requires_grad:
                    continue
                (no_decay if p.ndim <= 1 or pn.endswith(".bias") else decay).append(p)
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay,
                               "name": f"{name}/decay"})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0,
                               "name": f"{name}/no_decay"})

        n = self.n_blocks()
        blocks = set()
        for i, blk in enumerate(self.backbone.encoder.layer):
            scale = float(llrd) ** (n - 1 - i)
            add(f"encoder.blk{i:02d}", list(blk.named_parameters()), lr_encoder * scale)
            blocks |= {id(p) for p in blk.parameters()}
        rest = [(pn, p) for pn, p in self.backbone.named_parameters() if id(p) not in blocks]
        add("encoder.rest", rest, lr_encoder)

        add("decoder", list(self.neck.named_parameters()), lr_decoder)
        heads = nn.ModuleList([m for m in (self.film, self.height_head, self.sem_head, self.unc_head)
                               if m is not None])
        add("head", list(heads.named_parameters()), lr_head)
        return groups

    def set_encoder_trainable(self, flag: bool):
        """Warmup freeze/unfreeze, honouring any block spec set by set_trainable_blocks.

        The spec must win here.  param_groups() is built ONCE before the epoch loop and
        only collects parameters whose requires_grad was true at that moment, so a block
        frozen by the spec has no optimiser entry at all.  If this method later flipped
        it back on, it would accumulate gradients that nothing ever applies -- trainable
        by every assertion, silently frozen in fact.  That is precisely the class of
        silent divergence this project keeps finding, so the spec is re-applied rather
        than overridden.
        """
        spec = getattr(self, "_trainable_blocks", None)
        if not flag:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        elif spec is None:
            for p in self.backbone.parameters():
                p.requires_grad_(True)
        else:
            keep = set(spec)
            for i, blk in enumerate(self.backbone.encoder.layer):
                for p in blk.parameters():
                    p.requires_grad_(i in keep)
            for pn, p in self.backbone.named_parameters():
                if pn.startswith("embeddings"):
                    p.requires_grad_(bool(getattr(self, "_train_embed", False)))
        self._freeze_structurally_dead()   # re-apply: unfreezing the encoder would revive mask_token
