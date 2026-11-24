import argparse, csv, math, re, sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

WORK_WIDTH = 512          # registration is done at reduced size for speed/robustness
MAX_SHIFT_FRAC = 0.10     # flag if translation > 10% of width
MAX_ROT_DEG = 10.0        # flag if rotation > 10 degrees


def num_id(p: Path):
    m = re.search(r"IDRiD_(\d+)", p.stem)
    return int(m.group(1)) if m else None


def index_dir(d: Path):
    out = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            n = num_id(p)
            if n is not None:
                out[n] = p
    return out


def load_prepped(path: Path):
    """Return (resized colour image, normalised green-channel float32 for ECC)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    h, w = img.shape[:2]
    scale = WORK_WIDTH / w
    small = cv2.resize(img, (WORK_WIDTH, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    g = small[:, :, 1]                                   # green channel: best vessel contrast
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    g = cv2.GaussianBlur(g, (0, 0), 2.0).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)                # reduces illumination/contrast differences
    return small, g, (w, h)


def retina_mask(small):
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    m = (gray > 15).astype(np.uint8)
    m = cv2.erode(m, np.ones((15, 15), np.uint8))
    return m * 255


def run_ecc(tmpl, inp, mask, motion):
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(tmpl, inp, warp, motion, crit, mask, 5)
        return float(cc), warp, True
    except cv2.error:
        return float("nan"), warp, False


def describe(warp):
    tx, ty = float(warp[0, 2]), float(warp[1, 2])
    rot = math.degrees(math.atan2(warp[1, 0], warp[0, 0]))
    return tx, ty, rot


def apply(inp_color, warp, shape):
    h, w = shape
    return cv2.warpAffine(inp_color, warp, (w, h),
                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


def gray_ssim(a, b, mask):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    s, smap = ssim(ga, gb, data_range=255, full=True)
    sel = mask > 0
    return float(smap[sel].mean()) if sel.any() else float(s)


def overlay(tmpl_color, seg_color, path):
    """Left: grading | middle: seg (aligned) | right: red/green overlay of green channels."""
    a, b = tmpl_color[:, :, 1], seg_color[:, :, 1]
    ov = np.dstack([np.zeros_like(a), b, a])  # B=0, G=seg, R=grading
    cv2.imwrite(str(path), np.hstack([tmpl_color, seg_color, ov]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", required=True, type=Path)
    ap.add_argument("--grade-dir", required=True, type=Path)
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--out", type=Path, default=Path("reports/idrid_registration_ecc"))
    args = ap.parse_args()

    seg, grade = index_dir(args.seg_dir), index_dir(args.grade_dir)
    ids = args.ids if args.ids else sorted(set(seg) & set(grade))
    ids = [i for i in ids if i in seg and i in grade]
    if not ids:
        sys.exit("No matching IDs found. Check the two directories (same split only).")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Pairs to process: {len(ids)}")

    cache = {}
    def get(kind, i):
        k = (kind, i)
        if k not in cache:
            cache[k] = load_prepped((seg if kind == "seg" else grade)[i])
        return cache[k]

    motions = {"translation": cv2.MOTION_TRANSLATION, "euclidean": cv2.MOTION_EUCLIDEAN}
    rows = []
    for n, i in enumerate(ids):
        g_small, g_ecc, g_size = get("grade", i)
        s_small, s_ecc, s_size = get("seg", i)
        if s_small.shape != g_small.shape:
            s_small = cv2.resize(s_small, (g_small.shape[1], g_small.shape[0]))
            s_ecc = cv2.resize(s_ecc, (g_small.shape[1], g_small.shape[0]))
        mask = retina_mask(g_small)
        wrong_i = ids[(n + 1) % len(ids)] if len(ids) > 1 else None

        base = gray_ssim(g_small, s_small, mask)
        for name, mot in motions.items():
            cc, warp, ok = run_ecc(g_ecc, s_ecc, mask, mot)
            tx, ty, rot = describe(warp)
            plausible = ok and abs(tx) < MAX_SHIFT_FRAC * g_small.shape[1] \
                and abs(ty) < MAX_SHIFT_FRAC * g_small.shape[1] and abs(rot) < MAX_ROT_DEG
            aligned = apply(s_small, warp, g_small.shape[:2])
            after = gray_ssim(g_small, aligned, mask)

            # control: same procedure against a WRONG image from the same split
            wrong_cc = float("nan")
            if wrong_i is not None:
                w_small, w_ecc, _ = get("seg", wrong_i)
                w_ecc = cv2.resize(w_ecc, (g_small.shape[1], g_small.shape[0]))
                wrong_cc, _, _ = run_ecc(g_ecc, w_ecc, mask, mot)

            overlay(g_small, aligned, args.out / f"IDRiD_{i:03d}_{name}.png")
            rows.append(dict(id=i, model=name, converged=ok, ecc_cc=round(cc, 4),
                             wrong_pair_cc=round(wrong_cc, 4), tx_px=round(tx, 2),
                             ty_px=round(ty, 2), rot_deg=round(rot, 2),
                             ssim_before=round(base, 4), ssim_after=round(after, 4),
                             plausible_flag=plausible,
                             orig_size_seg=f"{s_size[0]}x{s_size[1]}",
                             orig_size_grade=f"{g_size[0]}x{g_size[1]}"))
            print(rows[-1])

    with open(args.out / "ecc_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {args.out/'ecc_results.csv'} and overlay PNGs. "
          "Inspect overlays visually: vessels/optic disc should coincide, not double.")


if __name__ == "__main__":
    main()
