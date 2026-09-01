import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from heightmap import config
from heightmap.data.gamus import GamusDataset, build_splits

ROOT = "data/GAMUS_synthetic"

def main():
    cfg = config.load(overrides=[f"data.root={ROOT}", f"data.sun_cache={ROOT}/sun_angles.json"])
    ds = GamusDataset(cfg, "train", train=True)
    b = ds[0]
    for k, v in b.items():
        print(f"  {k:9s}", tuple(v.shape) if hasattr(v, "shape") else v, getattr(v, "dtype", ""))
    assert b["image"].shape == (3, 518, 518) and b["height"].shape == (1, 518, 518)
    assert b["cls"].shape == (518, 518) and b["mask"].shape == (1, 518, 518)

    dl = DataLoader(ds, batch_size=2, num_workers=2, shuffle=True, persistent_workers=False)
    bb = next(iter(dl))
    print("batched ok:", {k: (tuple(v.shape) if hasattr(v, "shape") else "list") for k, v in bb.items()})
    print("gsd:", [round(x, 3) for x in bb["gsd"].tolist()], "sun:", bb["sun"].tolist())

    tr, va, te = build_splits(cfg)
    print("official:", {k: len(v) for k, v in tr.items()}, {k: len(v) for k, v in va.items()})

    cfg2 = config.load(overrides=[f"data.root={ROOT}", "data.split_mode=city_holdout"])
    tr2, va2, te2 = build_splits(cfg2)
    ctr = {c for v in tr2.values() for c, _ in v}
    cte = {c for v in te2.values() for c, _ in v}
    print("holdout cities  train:", ctr, " eval:", cte, " disjoint:", ctr.isdisjoint(cte))
    assert ctr.isdisjoint(cte) and cte == {"NYC"}
    print("PASS test_data")

if __name__ == "__main__":
    main()
