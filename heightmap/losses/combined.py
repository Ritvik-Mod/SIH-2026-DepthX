"""Assemble the loss suite.  Every term is logged separately -- you cannot debug a sum."""
from __future__ import annotations
import torch
from . import terms
from .shadow import shadow_consistency_loss
from ..data.transforms import denormalise
from ..data.classes import SHADOW_EXCLUDE


class CombinedLoss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.l = cfg.loss

    def __call__(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        gt, mask = batch["height"], batch["mask"]
        pred = out["height"]
        parts: dict[str, torch.Tensor] = {}

        if self.l.w_silog > 0:
            parts["silog"] = terms.silog_loss(pred, gt, mask, float(self.l.silog_lambda),
                                              float(self.l.silog_alpha))
        if self.l.w_nll > 0 and "log_var" in out:
            parts["nll"] = terms.gaussian_nll_loss(pred, gt, out["log_var"], mask)
        elif self.l.w_l1 > 0:
            parts["l1"] = terms.l1_loss(pred, gt, mask)
        if self.l.w_grad > 0:
            parts["grad"] = terms.multiscale_gradient_loss(pred, gt, mask, int(self.l.grad_scales))
        if self.l.w_bin_ce > 0 and "bin_logits" in out:
            parts["bin_ce"] = terms.bin_ce_loss(out["bin_logits"], out["bin_centres"], gt, mask)
        if self.l.w_chamfer > 0 and "bin_centres" in out:
            parts["chamfer"] = terms.chamfer_bins_loss(out["bin_centres"], gt, mask)
        if self.l.w_sem > 0 and "semantic" in out:
            parts["sem"] = terms.semantic_loss(out["semantic"], batch["cls"],
                                               int(self.cfg.model.ignore_index))
        if self.l.w_shadow > 0 and "sun" in batch:
            # Exclude the classes an image-based shadow detector confuses with shadow.
            # Use the GROUND-TRUTH mask, never the model's own prediction: predicting
            # "tree" everywhere would zero this loss out, which is a degenerate escape
            # hatch the optimiser would happily find.  (GAMUS ids: 2 low-veg, 4 water,
            # 6 tree.  Building is 3 and must NOT be excluded -- buildings are the
            # occluders whose shadows carry the height signal.)
            exclude = None
            src = batch.get("cls")
            if src is None and "semantic" in out:
                src = out["semantic"].argmax(1)          # inference-time fallback only
            if src is not None:
                c = src.unsqueeze(1) if src.dim() == 3 else src
                exclude = torch.zeros_like(c, dtype=torch.bool)
                for k in SHADOW_EXCLUDE:
                    exclude |= (c == k)
            parts["shadow"] = shadow_consistency_loss(
                pred, denormalise(batch["image"]), batch["gsd"], batch["sun"],
                self.l.shadow, exclude=exclude)

        w = {"silog": self.l.w_silog, "l1": self.l.w_l1, "nll": self.l.w_nll,
             "grad": self.l.w_grad, "bin_ce": self.l.w_bin_ce, "chamfer": self.l.w_chamfer,
             "sem": self.l.w_sem, "shadow": self.l.w_shadow}
        total = sum(float(w[k]) * v for k, v in parts.items())
        logs = {f"loss/{k}": float(v.detach()) for k, v in parts.items()}
        logs["loss/total"] = float(total.detach()) if torch.is_tensor(total) else float(total)
        return total, logs
