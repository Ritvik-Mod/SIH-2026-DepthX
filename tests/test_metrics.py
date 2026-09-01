import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from heightmap.metrics import evaluate, sharpness, format_table

def main():
    rng = np.random.default_rng(0)
    gt = torch.from_numpy((rng.random((1,1,64,64)) * 30).astype(np.float32))
    m = torch.ones(1,1,64,64, dtype=torch.bool)
    sem = torch.full((1,64,64), 3, dtype=torch.long)  # 3 = building

    r = evaluate(gt, gt, m, sem)
    print("identity:", {k: round(r["overall"][k],9) for k in ("mae","rmse","bias","r","delta1")})
    assert abs(r["overall"]["mae"]) < 1e-9 and abs(r["overall"]["rmse"]) < 1e-9
    assert abs(r["overall"]["bias"]) < 1e-9 and abs(r["overall"]["r"]-1) < 1e-9
    assert abs(r["overall"]["delta1"]-1) < 1e-9

    # correlation is scale/shift invariant; MAE is not
    scaled = gt*1.4 + 3.0
    rs = evaluate(scaled, gt, m, sem)
    print(f"scaled 1.4x+3: MAE {rs['overall']['mae']:.3f}  r {rs['overall']['r']:.6f}")
    assert abs(rs["overall"]["r"]-1) < 1e-6, "r should be invariant to affine rescaling"
    assert rs["overall"]["mae"] > 3.0

    # signed bias detects systematic shortening
    short = gt - 4.0
    rb = evaluate(short.clamp(min=0), gt, m, sem)
    print(f"uniformly 4m short: MAE {rb['overall']['mae']:.3f}  bias {rb['overall']['bias']:+.3f}")
    assert rb["overall"]["bias"] < -2.0

    print("\nsharpness (blurred vs sharp):")
    sharp = torch.zeros(1,1,64,64); sharp[...,32:,:] = 20.0
    # reflect-pad so the blur does not manufacture an edge at the image border
    blur = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(sharp, (4,4,4,4), mode="reflect"), 9, 1, 0)
    sb = sharpness(blur, sharp, m); ss = sharpness(sharp, sharp, m)
    print("   blurred:", {k: round(v,4) if isinstance(v,float) else v for k,v in sb.items()})
    print("   sharp  :", {k: round(v,4) if isinstance(v,float) else v for k,v in ss.items()})
    assert sb["ratio"] < 0.5, sb
    assert abs(ss["ratio"]-1) < 1e-6, ss

    print("\n" + format_table(evaluate(gt + torch.randn_like(gt)*2, gt, m, sem)))
    print("\nPASS test_metrics")

if __name__ == "__main__":
    main()
