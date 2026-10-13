#!/usr/bin/env python3
"""Validate the manual tooth-bbox adjustment files against annotation.txt rules.

Part of the YOLO11 fine-tuning data preparation: run this *before* building
``tooth_bbox_adj`` so malformed annotation files are caught early.

Inputs
------
ADJUST_DIR : folder of manual corrections, one sub-folder per case_id, each
             containing  <VIEW>.png / <VIEW>.txt / <VIEW>.json  (VIEW in D/F/L/R/U).
             In the reorganised dataset this is ``intraoral_adjust/sample``.
ZEYU_DIR   : original model output root, containing  <batch>_process/tooth_bbox/<case_id>/<VIEW>.json

Outputs
-------
- console summary
- <out>/validation_report.csv   (one row per (case, view))
- <out>/validation_report.json  (structured, grouped by issue type)

Only the Python standard library is used (no numpy / opencv / PIL).
"""

import argparse
import csv
import json
import os
import re
import glob
import struct
import collections

VIEWS = ["D", "F", "L", "R", "U"]
FDI_RE = re.compile(r"^\d{2}$")
TXT_RENAME_RE = re.compile(r"^(\d{2}),(\d{2})$")
TXT_SINGLE_RE = re.compile(r"^(\d{2})$")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def png_size(path):
    """Return (width, height) read from the PNG IHDR chunk."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def valid_fdi(fdi):
    """Two-digit FDI: quadrant 1-4, tooth 1-8 (allow 0 for supernumerary? no)."""
    if not isinstance(fdi, int):
        return False
    q, t = fdi // 10, fdi % 10
    return q in (1, 2, 3, 4) and 1 <= t <= 8


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_txt(path):
    """Parse a correction txt file.

    Returns (renames, deletes, issues).
      renames : {old_fdi: new_fdi}
      deletes : set(fdi)
      issues  : list of (kind, detail)
    """
    renames, deletes, issues = {}, set(), []
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    for lineno, line in enumerate(raw.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        if "." in s and "," not in s:
            issues.append(("txt_dot_separator", "line %d: %r" % (lineno, s)))
            s = s.replace(".", ",")
        m = TXT_RENAME_RE.match(s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a in renames and renames[a] != b:
                issues.append(("txt_conflicting_rename", "line %d: %r" % (lineno, s)))
            renames[a] = b
            continue
        m = TXT_SINGLE_RE.match(s)
        if m:
            deletes.add(int(m.group(1)))
            continue
        issues.append(("txt_bad_line", "line %d: %r" % (lineno, s)))
    return renames, deletes, issues


def parse_labelme(path):
    """Parse a labelme rectangle json.

    Returns (shapes, issues) where shapes = [(fdi_int_or_None, box_xyxy_labelme)].
    """
    d = load_json(path)
    shapes, issues = [], []
    w, h = d.get("imageWidth"), d.get("imageHeight")
    for i, sh in enumerate(d.get("shapes", [])):
        if sh.get("shape_type") != "rectangle":
            issues.append(("labelme_non_rectangle", "shape %d: %s" % (i, sh.get("shape_type"))))
            continue
        label = sh.get("label")
        if not (isinstance(label, str) and FDI_RE.match(label.strip())):
            issues.append(("labelme_bad_label", "shape %d: %r" % (i, label)))
            fdi = None
        else:
            fdi = int(label.strip())
            if not valid_fdi(fdi):
                issues.append(("labelme_bad_label", "shape %d: %r" % (i, label)))
        pts = sh.get("points", [])
        if len(pts) != 2:
            issues.append(("labelme_bad_points", "shape %d: %d points" % (i, len(pts))))
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        box = [min(xs), min(ys), max(xs), max(ys)]
        if w and h:
            if box[0] < 0 or box[1] < 0 or box[2] > w or box[3] > h:
                issues.append(("labelme_box_out_of_bounds", "shape %d: %s in %dx%d" % (i, box, w, h)))
        shapes.append((fdi, box))
    return shapes, issues


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adjust", default="intraoral_adjust/sample")
    ap.add_argument("--zeyu", default="intraoral/zeyu")
    ap.add_argument("--out", default="intraoral_adjust/reports")
    ap.add_argument("--relax-scale", action="store_true",
                    help="do not report json annotated on preview (scale != 1) as an issue")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # map case_id -> tooth_bbox dir of the original predictions
    zmap = {}
    for base in glob.glob(os.path.join(args.zeyu, "*_process", "tooth_bbox")):
        for c in os.listdir(base):
            zmap.setdefault(c, base)

    cases = sorted(d for d in os.listdir(args.adjust)
                   if os.path.isdir(os.path.join(args.adjust, d)))

    issues = collections.defaultdict(list)   # kind -> list of "case/view: detail"
    rows = []

    def add(kind, case, view, detail=""):
        issues[kind].append("%s/%s: %s" % (case, view, detail) if view else "%s: %s" % (case, detail))

    for case in cases:
        cdir = os.path.join(args.adjust, case)
        files = os.listdir(cdir)

        # -------- filename check --------
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext not in (".png", ".txt", ".json") or stem not in VIEWS:
                add("stray_file", case, "", f)

        if case not in zmap:
            add("case_not_in_zeyu", case, "")
            continue

        for view in VIEWS:
            fp = os.path.join(cdir, view + ".png")
            ft = os.path.join(cdir, view + ".txt")
            fj = os.path.join(cdir, view + ".json")
            zf = os.path.join(zmap[case], case, view + ".json")

            if not os.path.exists(zf):
                add("zeyu_json_missing", case, view)
                continue
            z = load_json(zf)
            orig_fdis = [t["fdi"] for t in z["teeth"]]
            orig_set = set(orig_fdis)
            ow, oh = z["orig_size"]

            n_files = sum(map(os.path.exists, (fp, ft, fj)))
            if n_files == 0:
                continue  # untouched view: original prediction is fine

            # -------- rule 4: png presence / single txt+json --------
            if (os.path.exists(ft) or os.path.exists(fj)) and not os.path.exists(fp):
                add("correction_without_png", case, view,
                    "txt=%s json=%s" % (os.path.exists(ft), os.path.exists(fj)))

            row = {"case": case, "view": view,
                   "has_png": os.path.exists(fp), "has_txt": os.path.exists(ft),
                   "has_json": os.path.exists(fj)}

            # -------- txt --------
            if os.path.exists(ft):
                renames, deletes, t_issues = parse_txt(ft)
                for k, det in t_issues:
                    add(k, case, view, det)
                for a in renames:
                    if a not in orig_set:
                        add("rename_source_not_in_orig", case, view, "%d" % a)
                for x in deletes:
                    if x not in orig_set:
                        add("delete_not_in_orig", case, view, "%d" % x)
                row.update(renames=len(renames), deletes=len(deletes))
            else:
                renames, deletes = {}, set()

            # -------- json --------
            if os.path.exists(fj):
                a = load_json(fj)
                lw, lh = a.get("imageWidth"), a.get("imageHeight")
                shapes, j_issues = parse_labelme(fj)
                for k, det in j_issues:
                    add(k, case, view, det)
                # duplicate labels
                labs = [f for f, _ in shapes if f is not None]
                dup = [f for f, c in collections.Counter(labs).items() if c > 1]
                if dup:
                    add("labelme_duplicate_label", case, view, str(sorted(dup)))
                # scale
                sx = (ow / lw) if lw else 1.0
                sy = (oh / lh) if lh else 1.0
                if not args.relax_scale and (abs(sx - 1) > 0.02 or abs(sy - 1) > 0.02):
                    add("json_annotated_on_preview", case, view,
                        "scale=%.3f/%.3f (%dx%d -> %dx%d)" % (sx, sy, lw, lh, ow, oh))
                row.update(json_shapes=len(shapes), json_scale=round(sx, 3))
            else:
                row.update(json_shapes=0, json_scale=None)

            # -------- content / view-alignment check --------
            if os.path.exists(ft):
                refs = set(renames) | set(renames.values()) | set(deletes)
                # ignore renames that came from elsewhere; only flag when NONE of
                # the referenced fdi exist in this view but do in a sibling view
                if refs and not (refs & orig_set):
                    add("txt_refs_not_in_this_view", case, view, str(sorted(refs)))

            rows.append(row)

    # ------------------------------------------------------------------ #
    # summary
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("cases in adjust: %d   matched in zeyu: %d" % (len(cases), sum(c in zmap for c in cases)))
    print("-" * 70)
    if not issues:
        print("No issues found.")
    for kind in sorted(issues):
        print("%-32s %d" % (kind, len(issues[kind])))
    print("=" * 70)

    with open(os.path.join(args.out, "validation_report.json"), "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in sorted(issues.items())}, fh, indent=1, ensure_ascii=False)

    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(args.out, "validation_report.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    print("report written to %s/" % args.out)


if __name__ == "__main__":
    main()
