import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from heightmap import config
from heightmap.models.net import HeightNet

def main():
    cfg = config.load(overrides=["model.use_uncertainty=true"])
    net = HeightNet(cfg).eval()
    x = torch.randn(2, 3, 518, 518); gsd = torch.tensor([0.5, 1.2])
    with torch.no_grad():
        o = net(x, gsd)
    for k, v in o.items():
        print(f"  {k:12s} {tuple(v.shape)}  [{v.min():.3f}, {v.max():.3f}]")
    assert o["height"].shape == (2, 1, 518, 518)
    assert o["semantic"].shape == (2, cfg.model.n_classes, 518, 518)
    assert o["bin_logits"].shape[1] == cfg.model.n_bins
    assert o["bin_centres"].shape == (2, cfg.model.n_bins)
    assert 0 <= o["height"].min() and o["height"].max() <= cfg.model.h_max + 1e-3
    # bin centres must be sorted and span (0, h_max]
    c = o["bin_centres"]
    assert bool((c[:, 1:] > c[:, :-1]).all()), "bin centres not increasing"
    print("  bin centres: first %.3f  last %.3f  (h_max %.0f)" % (c[0,0], c[0,-1], cfg.model.h_max))

    g = net.param_groups(1e-5, 5e-5, 1e-4, 0.01)
    for gr in g:
        print(f"  group {gr['name']:18s} lr={gr['lr']:.0e} wd={gr['weight_decay']:.3g} n={len(gr['params'])}")
    total = sum(p.numel() for p in net.parameters())
    print("  params %.1fM" % (total / 1e6))

    # FiLM must actually depend on gsd
    cfg2 = config.load()
    net2 = HeightNet(cfg2).eval()
    with torch.no_grad():
        net2.film.mlp[-1].weight.normal_(0, 0.05); net2.film.mlp[-1].bias.normal_(0, 0.05)
        a = net2(x, torch.tensor([0.3, 0.3]))["height"]
        b = net2(x, torch.tensor([2.0, 2.0]))["height"]
    d = (a - b).abs().mean().item()
    print("  FiLM sensitivity to gsd: mean|d| = %.4f m" % d)
    assert d > 1e-4, "FiLM output does not depend on gsd"
    print("PASS test_model")

if __name__ == "__main__":
    main()
