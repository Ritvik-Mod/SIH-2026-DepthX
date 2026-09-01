#!/usr/bin/env python
"""Gather every run's best validation metrics into one table.  Paste it into the
report unmodified -- hand-copying numbers is how wrong ones get published."""
import sys, os, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs")
    ap.add_argument("--out", default="outputs/results.md")
    a = ap.parse_args()

    rows = []
    z = os.path.join(a.runs, "a0_zero_shot.json")
    if os.path.exists(z):
        r = json.load(open(z))
        for tag, key in (("A0 zero-shot (global fit)", "global_fit"),
                         ("A0 zero-shot (per-tile oracle)", "per_tile_oracle")):
            o, b = r[key]["overall"], r[key]["building"]
            rows.append((tag, o["mae"], o["rmse"], o["r"], b["mae"], b["bias"], b["r"]))

    for ck in sorted(glob.glob(os.path.join(a.runs, "*", "best.pt"))):
        name = os.path.basename(os.path.dirname(ck))
        try:
            d = torch.load(ck, map_location="cpu", weights_only=False)
            v = d.get("val")
            if not v:
                continue
            o, b = v["overall"], v["building"]
            rows.append((name, o["mae"], o["rmse"], o["r"], b["mae"], b["bias"], b["r"]))
        except Exception as e:
            print(f"  skip {ck}: {e}")

    if not rows:
        sys.exit("no results found")

    hdr = (f"| {'run':<32} | {'MAE':>7} | {'RMSE':>7} | {'r':>6} | "
           f"{'bMAE':>7} | {'bBias':>8} | {'b_r':>6} |")
    sep = "|" + "|".join("-" * (len(c) + 2) for c in hdr.split("|")[1:-1]) + "|"
    lines = ["## Results", "",
             "MAE/RMSE/bias in metres.  `b` = building pixels only.  "
             "`r` is scale- and shift-invariant, so it isolates structure.", "",
             hdr, sep]
    for n, mae, rmse, r, bmae, bbias, br in rows:
        lines.append(f"| {n:<32} | {mae:>7.3f} | {rmse:>7.3f} | {r:>6.3f} | "
                     f"{bmae:>7.3f} | {bbias:>+8.3f} | {br:>6.3f} |")
    lines += ["", "Report the city-held-out row FIRST.  Leading with the best number "
                  "invites scepticism about everything else."]
    txt = "\n".join(lines)
    open(a.out, "w").write(txt + "\n")
    print(txt)
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
