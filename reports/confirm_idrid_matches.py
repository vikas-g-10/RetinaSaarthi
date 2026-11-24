"""
Content-based IDRiD correspondence: find, for each segmentation image, its best
grading image IGNORING file numbering, then check the candidates properly.

Outputs (in --out):
  score_matrix.npy         54 x N matrix of match scores (seg x grading)
  candidate_matches.csv    best/second match, margin, shift, pixel-level checks
  sheets/seg_XX_vs_grade_YYY.jpg   [segmentation | grading | |difference|] for visual review

Nothing is accepted automatically: 'status' is only a screening label.
  confident  : score >= HI and margin to 2nd-best >= MARGIN and mutual-best
  uncertain  : everything else (inspect visually, or probably has no partner in this split)
Run training and testing splits separately.
"""
import argparse, csv, re
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

W = 256
HI, MARGIN = 0.99, 0.05


def num_id(p):
    m = re.search(r"IDRiD_(\d+)", p.stem)
    return int(m.group(1)) if m else None


def index_dir(d):
    return {num_id(p): p for p in sorted(d.iterdir())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff") and num_id(p) is not None}


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
    ap.add_argument("--seg-dir", required=True, type=Path)
    ap.add_argument("--grade-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/idrid_confirm_train"))
    a = ap.parse_args()
    (a.out / "sheets").mkdir(parents=True, exist_ok=True)

    seg, grade = index_dir(a.seg_dir), index_dir(a.grade_dir)
    sids, gids = sorted(seg), sorted(grade)
    print(f"{len(sids)} segmentation x {len(gids)} grading images")
    G = {j: prep(grade[j]) for j in gids}

    S = np.zeros((len(sids), len(gids)), np.float32)
    shift = {}
    for r, i in enumerate(sids):
        t, (ex, ey) = crop(prep(seg[i]))
        for c, j in enumerate(gids):
            res = cv2.matchTemplate(G[j], t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            S[r, c] = mx
            shift[(r, c)] = (loc[0] - ex, loc[1] - ey)
    np.save(a.out / "score_matrix.npy", S)

    best_seg_for_grade = {c: int(np.argmax(S[:, c])) for c in range(len(gids))}
    rows = []
    for r, i in enumerate(sids):
        order = np.argsort(-S[r])
        c, c2 = int(order[0]), int(order[1])
        score, margin = float(S[r, c]), float(S[r, c] - S[r, c2])
        mutual = best_seg_for_grade[c] == r
        status = "confident" if (score >= HI and margin >= MARGIN and mutual) else "uncertain"

        mae = ss = float("nan")
        if score >= 0.9:  # pixel-level look at full-ish resolution
            a_img, b_img = load_mid(seg[i]), load_mid(grade[gids[c]])
            diff = cv2.absdiff(a_img, b_img)
            mae = float(diff.mean())
            ss = float(ssim(cv2.cvtColor(a_img, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(b_img, cv2.COLOR_BGR2GRAY), data_range=255))
            sheet = np.hstack([a_img, b_img, diff])
            sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2))
            cv2.imwrite(str(a.out / "sheets" / f"seg_{i:02d}_vs_grade_{gids[c]:03d}.jpg"), sheet)

        rows.append(dict(seg_id=i, best_grade_id=gids[c], score=round(score, 4),
                         second_id=gids[c2], second_score=round(float(S[r, c2]), 4),
                         margin=round(margin, 4), mutual_best=mutual,
                         shift_px_at256=shift[(r, c)], pixel_mae=round(mae, 2),
                         pixel_ssim=round(ss, 4), same_number=(i == gids[c]), status=status))
        print(rows[-1])

    with open(a.out / "candidate_matches.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    conf = [x for x in rows if x["status"] == "confident"]
    used = [x["best_grade_id"] for x in conf]
    print("\n===== SUMMARY =====")
    print(f"confident: {len(conf)}/{len(rows)} | uncertain: {len(rows)-len(conf)}")
    print(f"confident matches reusing the same grading image: {len(used) - len(set(used))}")
    print(f"confident where numbering agrees (seg N -> grade N): {sum(x['same_number'] for x in conf)}")
    zero = sum(1 for x in conf if x["shift_px_at256"] == (0, 0))
    print(f"confident with zero estimated shift: {zero}/{len(conf)}")
    print(f"Saved to {a.out}")


if __name__ == "__main__":
    main()
