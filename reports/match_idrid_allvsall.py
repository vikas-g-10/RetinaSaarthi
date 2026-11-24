"""
All-vs-all correspondence test: IDRiD segmentation images vs IDRiD grading images.

Question: for each segmentation image IDRiD_XX, is the grading image with the
same number (IDRiD_0XX) the best match among ALL grading images of the same
split? If numbering really maps between the sets, the true partner should rank
near 1. If ranks look random, the numbering does not identify the same photo.

Method (no registration, no thresholds tuned on the answer):
  - both images reduced to 256 px wide, green channel, CLAHE, blur, normalised
  - the central 80% of the segmentation image is slid over the grading image
    (cv2.matchTemplate, normalised cross-correlation) so small shifts are tolerated
  - score = max NCC over shifts
Chance level for top-1 is 1 / (number of grading images).

Run training and testing splits SEPARATELY.

Usage:
  python reports/match_idrid_allvsall.py \
    --seg-dir "<segmentation split folder>" \
    --grade-dir "<grading split folder>" \
    --out reports/idrid_allvsall_train
"""
import argparse, csv, re
from pathlib import Path

import cv2
import numpy as np

W = 256


def num_id(p):
    m = re.search(r"IDRiD_(\d+)", p.stem)
    return int(m.group(1)) if m else None


def index_dir(d):
    out = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            n = num_id(p)
            if n is not None:
                out[n] = p
    return out


def prep(path):
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)  # fast, ~536 px wide
    if img is None:
        raise IOError(f"cannot read {path}")
    h, w = img.shape[:2]
    img = cv2.resize(img, (W, int(round(h * W / w))), interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(3.0, (8, 8)).apply(img[:, :, 1])
    g = cv2.GaussianBlur(g, (0, 0), 1.5).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def center_crop(g, frac=0.8):
    h, w = g.shape
    ch, cw = int(h * frac), int(w * frac)
    y, x = (h - ch) // 2, (w - cw) // 2
    return g[y:y + ch, x:x + cw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", required=True, type=Path)
    ap.add_argument("--grade-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/idrid_allvsall"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    seg, grade = index_dir(a.seg_dir), index_dir(a.grade_dir)
    print(f"Segmentation images: {len(seg)} | Grading images: {len(grade)}")
    print("Loading grading images...")
    G = {j: prep(p) for j, p in grade.items()}
    gids = sorted(G)

    rows, ranks = [], []
    for i in sorted(seg):
        t = center_crop(prep(seg[i]))
        scores = {}
        for j in gids:
            res = cv2.matchTemplate(G[j], t, cv2.TM_CCOEFF_NORMED)
            scores[j] = float(res.max())
        order = sorted(scores, key=scores.get, reverse=True)
        best = order[0]
        true_rank = order.index(i) + 1 if i in scores else None
        if true_rank is not None:
            ranks.append(true_rank)
        rows.append(dict(seg_id=i, true_partner=i if i in scores else "",
                         true_rank=true_rank, true_score=round(scores.get(i, float("nan")), 4),
                         best_id=best, best_score=round(scores[best], 4),
                         second_score=round(scores[order[1]], 4)))
        print(rows[-1])

    with open(a.out / "allvsall_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    r = np.array(ranks)
    n = len(r)
    print("\n===== SUMMARY =====")
    print(f"Pairs evaluated: {n} | candidates per query: {len(gids)} "
          f"| chance top-1: {1/len(gids):.4f}")
    print(f"Top-1: {(r == 1).sum()}/{n}  Top-5: {(r <= 5).sum()}/{n}  "
          f"Top-10: {(r <= 10).sum()}/{n}")
    print(f"Median true rank: {np.median(r):.1f} | mean: {r.mean():.1f} "
          f"(random expectation ~ {(len(gids)+1)/2:.0f})")
    print(f"Saved {a.out/'allvsall_results.csv'}")


if __name__ == "__main__":
    main()
