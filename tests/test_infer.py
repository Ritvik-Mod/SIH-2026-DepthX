"""Tiled inference must recover global structure and leave no seams.

The fixture reproduces the real failure mode exactly: the model recovers local structure
but re-references every tile to its own mean, because a tile containing no visible ground
has to guess where zero is.  That destroys the scene's low-frequency component and it
reappears as steps at tile boundaries.

The reference to score against is the SAME model with drift disabled, run on the whole
scene in one pass -- i.e. the model's own ideal answer, in the model's own units.
(Comparing against the raw image would be wrong: the model consumes ImageNet-normalised
input, so its output lives on a different scale.)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from heightmap.infer import predict_scene, _positions
from heightmap.export import write_outputs
from heightmap.data.transforms import normalise


class FakeModel(nn.Module):
    def __init__(self, drift=True):
        super().__init__(); self.drift = drift; self.p = nn.Parameter(torch.zeros(1))

    def forward(self, x, gsd=None):
        base = x.mean(1, keepdim=True) * 60.0
        if self.drift:
            base = base - base.mean(dim=(2, 3), keepdim=True) + 20.0
        return {"height": base}


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def boundary_step(a, bnds, axis=1):
    """A seam is a step: a spike in the FIRST difference localised at a tile edge.
    Measured relative to the local first-difference away from edges, so a scene with
    genuine structure does not register as seamy."""
    d = np.abs(np.diff(a, axis=axis))
    at = np.mean([d.take(range(b - 1, b + 1), axis=axis).mean() for b in bnds])
    mask = np.ones(d.shape[axis], bool)
    for b in bnds:
        mask[max(0, b - 4):b + 4] = False
    off = d.compress(mask, axis=axis).mean()
    return at / max(off, 1e-9)


def main():
    rng = np.random.default_rng(0)
    n = 1400
    yy, xx = np.mgrid[0:n, 0:n]
    smooth = (0.15 + 0.7 * (xx + yy) / (2.0 * n)
              + 0.06 * np.sin(xx / 37.0) * np.cos(yy / 41.0))
    img = np.clip(smooth[..., None].repeat(3, 2) + rng.normal(0, 0.01, (n, n, 3)), 0, 1)
    img = (img * 255).astype(np.uint8)

    # the model's own ideal answer: no drift, whole scene in one pass
    with torch.no_grad():
        t = torch.from_numpy(normalise(img).transpose(2, 0, 1)).unsqueeze(0)
        ideal = FakeModel(drift=False)(t)["height"][0, 0].numpy()
    ideal = np.clip(ideal, 0, None)

    m = FakeModel(drift=True).eval()
    kw = dict(gsd=0.5, tile=518, overlap=0.25, batch=4,
              device=torch.device("cpu"), progress=False)
    r_lv = predict_scene(m, img, level=True, **kw)
    r_no = predict_scene(m, img, level=False, **kw)

    c_lv, c_no = corr(r_lv.agl, ideal), corr(r_no.agl, ideal)
    print(f"  scene {n}x{n}, {r_lv.meta['n_tiles']} tiles, stride {r_lv.meta['stride']}")
    print(f"  correlation with the model's ideal whole-scene answer:")
    print(f"     levelled   {c_lv:.4f}")
    print(f"     un-levelled{c_no:>9.4f}")

    stride = r_lv.meta["stride"]
    xs = _positions(n, 518, stride)
    bnds = sorted({b for x in xs for b in (x, x + 518) if 6 < b < n - 6})
    s_lv, s_no = boundary_step(r_lv.agl, bnds), boundary_step(r_no.agl, bnds)
    print(f"  boundary step ratio (1.0 = no seam):  levelled {s_lv:.2f}   un-levelled {s_no:.2f}")
    print(f"  overlap disagreement: mean {r_lv.overlap_disagreement.mean():.4f} m")

    assert c_no < 0.75, f"fixture produced no drift to fix (un-levelled corr {c_no:.3f})"
    assert c_lv > 0.93, f"levelling failed to recover global structure (corr {c_lv:.3f})"
    assert c_lv > c_no + 0.2, f"levelling gave no meaningful gain ({c_no:.3f} -> {c_lv:.3f})"
    assert s_lv < 2.0, f"visible seams remain after levelling (ratio {s_lv:.2f})"

    for shape in ((733, 991), (300, 300), (2000, 519)):
        im2 = (np.clip(rng.random((*shape, 3)), 0, 1) * 255).astype(np.uint8)
        r2 = predict_scene(m, im2, **{**kw, "batch": 2})
        assert r2.agl.shape == shape, (r2.agl.shape, shape)
        assert np.isfinite(r2.agl).all()
        print(f"  odd shape {str(shape):>12} -> {r2.agl.shape} OK")

    meta = write_outputs("outputs/_test_export", "scene", r_lv.agl,
                         sigma=r_lv.overlap_disagreement, gsd=0.5,
                         model_info={"model": "FakeModel"}, infer_meta=r_lv.meta)
    assert meta["quantity"] == "AGL" and meta["nodata"] == -9999.0
    assert os.path.exists("outputs/_test_export/scene.json")
    assert os.path.exists("outputs/_test_export/scene.tif")
    print(f"  export OK: quantity={meta['quantity']} bands={list(meta['bands'].values())}")
    print("\nPASS test_infer")


if __name__ == "__main__":
    main()
