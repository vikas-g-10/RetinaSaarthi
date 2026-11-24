"""
Build the verified IDRiD segmentation -> grading correspondence table.

Searches EVERY segmentation image (train + test) against EVERY grading image
(train + test), ignoring file numbers. Images are keyed by (split, filename),
because numbering is reused across splits.

Outputs in --out:
  mapping.csv   one row per segmentation image, with status confident/uncertain
  sheets/       [seg | grading | |diff|] for every candidate with score >= 0.9

Sanity check built in: the 42 training pairs already confirmed should reappear.
Nothing is accepted automatically beyond the 'status' screening label.

Usage:
  python reports/build_idrid_mapping.py \
    --seg train="<seg train folder>" test="<seg test folder>" \
    --grade train="<grading train folder>" test="<grading test folder>" \
    --out reports/idrid_mapping
"""
import argparse, csv
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

W = 256
HI, MARGIN = 0.99, 0.05
EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def parse_kv(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k] = Path(v.strip('"'))
    return out


def list_images(split_dirs):
    return [(s, p) for s, d in split_dirs.items()
            for p in sorted(d.iterdir()) if p.suffix.lower() in EXT]


def prep(path):
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    h, w = img.shape[:2]
    img = cv2.resize(img, (W, int(round(h * W / w))), interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(3.0, (8, 8)).apply(img[:, :, 1])
    g = cv2.GaussianBlur(g, (0, 0), 1.5).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def crop(g, frac=0.8):
    h, w = g.shape
    ch, cw = int(h * frac), int(w * frac)
    y, x = (h - ch) // 2, (w - cw) // 2
    return g[y:y + ch, x:x + cw], (x, y)


def load_mid(path, width=1024):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", nargs="+", required=True)
    ap.add_argument("--grade", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/idrid_mapping"))
    a = ap.parse_args()
    (a.out / "sheets").mkdir(parents=True, exist_ok=True)

    segs = list_images(parse_kv(a.seg))
    grades = list_images(parse_kv(a.grade))
    print(f"{len(segs)} segmentation x {len(grades)} grading images")

    G = [prep(p) for _, p in grades]
    S = np.zeros((len(segs), len(grades)), np.float32)
    shifts = {}
    for r, (_, sp) in enumerate(segs):
        t, (ex, ey) = crop(prep(sp))
        for c in range(len(grades)):
            _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(G[c], t, cv2.TM_CCOEFF_NORMED))
            S[r, c] = mx
            shifts[(r, c)] = (loc[0] - ex, loc[1] - ey)
    np.save(a.out / "score_matrix.npy", S)

    best_seg_for_grade = np.argmax(S, axis=0)
    rows = []
    for r, (ssplit, sp) in enumerate(segs):
        order = np.argsort(-S[r])
        c, c2 = int(order[0]), int(order[1])
        score, margin = float(S[r, c]), float(S[r, c] - S[r, c2])
        mutual = int(best_seg_for_grade[c]) == r
        status = "confident" if (score >= HI and margin >= MARGIN and mutual) else "uncertain"
        gsplit, gp = grades[c]

        mae = ss = float("nan")
        if score >= 0.9:
            ai, bi = load_mid(sp), load_mid(gp)
            diff = cv2.absdiff(ai, bi)
            mae = float(diff.mean())
            ss = float(ssim(cv2.cvtColor(ai, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(bi, cv2.COLOR_BGR2GRAY), data_range=255))
            sheet = cv2.resize(np.hstack([ai, bi, diff]), (1536, 340))
            cv2.imwrite(str(a.out / "sheets" / f"{ssplit}_{sp.stem}__{gsplit}_{gp.stem}.jpg"), sheet)

        rows.append(dict(seg_split=ssplit, seg_file=sp.name, grade_split=gsplit,
                         grade_file=gp.name, score=round(score, 4),
                         second_file=f"{grades[c2][0]}/{grades[c2][1].name}",
                         second_score=round(float(S[r, c2]), 4), margin=round(margin, 4),
                         mutual_best=mutual, shift_px_at256=shifts[(r, c)],
                         pixel_mae=round(mae, 2), pixel_ssim=round(ss, 4), status=status))
        print(rows[-1])

    with open(a.out / "mapping.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    conf = [x for x in rows if x["status"] == "confident"]
    keys = [(x["grade_split"], x["grade_file"]) for x in conf]
    print("\n===== SUMMARY =====")
    for s in sorted({x["seg_split"] for x in rows}):
        sub = [x for x in rows if x["seg_split"] == s]
        n = sum(x["status"] == "confident" for x in sub)
        print(f"seg {s}: {n}/{len(sub)} confident")
    print(f"confident pairs: {len(conf)} | grading images reused: {len(keys) - len(set(keys))}")
    cross = sum(x["seg_split"] != x["grade_split"] for x in conf)
    print(f"confident pairs crossing train/test split: {cross}")
    print(f"mean pixel_mae over confident: {np.nanmean([x['pixel_mae'] for x in conf]):.3f}")
    print(f"Saved {a.out/'mapping.csv'}")


if __name__ == "__main__":
    main()
