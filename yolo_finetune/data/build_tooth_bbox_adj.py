#!/usr/bin/env python3
"""Build a corrected tooth-bbox set (``tooth_bbox_adj``) from the manual
adjustment files + the original model predictions.

This is the ground truth used to fine-tune the YOLO11 tooth detector.

Dataset layout (run with cwd = the datasets root, i.e. the repo's
``.datasets/`` symlink directory, or /disk1/work/zychen/Datasets)
------------------------------------------------------------------------
    intraoral_adjust/
        sample/annotation.txt                       global correction spec
        sample/<case>/{D,F,L,R,U}.{png,txt,json}    manual corrections, per case
        unfiltered_bbox/<case>/<view>.json          model predictions (no view filter)
        tooth_bbox_adj/<case>/<view>.json           <-- this script's output
        vis_adj/<case>/<view>.png                   visualisation (visualize_adjust.py)
        reports/                                    QC reports

Which predictions are the starting point?
-----------------------------------------
**Unfiltered** ones -- ``intraoral_adjust/unfiltered_bbox/<case>/<view>.json``,
produced by ``build_tooth_bbox_predict.py`` by re-running the same YOLO detector
as the ``preview/`` PNGs, i.e. *without* the per-view FDI whitelist
(``TARGET_TEETH_PER_VIEW``).

This matters because ``tooth_bbox`` (the filtered product) only keeps the
teeth that belong to that view, while the annotator reviewed the ``preview``
image which shows **every** detected tooth.  Starting from the unfiltered set
makes the annotator's instructions applicable: no legitimate ``16,15`` rename
in an ``F`` view gets dropped just because ``tooth_bbox`` never listed 16.

The output therefore covers the full arch exactly like ``preview`` does, and
is directly comparable with the manual corrections.

Merge rules (confirmed with the annotator)
------------------------------------------
Per (case, view), starting from the unfiltered prediction:

1.  txt ``old,new``  -> rename the FDI of that tooth (box kept).
2.  txt single ``fdi`` -> delete that tooth (only when it exists in the original).
3.  labelme json boxes -> a *final* box for the labelled FDI:
        * label already exists   -> REPLACE its box (refinement)
        * label not present      -> ADD a new tooth
4.  json takes priority over txt when both touch the same resulting FDI.
5.  any original tooth not touched by the above is KEPT unchanged.

Renames are resolved against the **original** labels, i.e. all instructions are
applied *simultaneously*, not chained line by line.  This is deliberate, and
matches how the annotator writes them.  Chains such as::

    s20251149/D.txt:  45 / 31,32 / 32,33 / 33,34 / 34,35 / 35

mean "shift the 31..34 group up by one (31->32, 32->33, 33->34, 34->35) and
drop the original 45 and 35" -- NOT "34->35" applied to a tooth that was just
moved into 34.  The same idiom appears 34 times, always as the annotator
freeing up a tooth number (delete ``f``) and then relabelling another tooth
*into* it (``g,f``); both spellings therefore coexist in one file by design and
are not a contradiction.

The only genuinely unresolvable instructions are renames/deletes whose *source*
FDI was never predicted in that view; those are reported in
``build_warnings.txt`` as ``unmatched ...`` and can only be fixed by hand.

Labelme points are annotated either on the full-resolution ``process`` image
(scale 1.0) or on the down-scaled ``preview`` image (scale ~1.30); they are
always rescaled into the original pixel space via ``orig_size / labelme_size``.

Only the standard library is used.
"""

import argparse
import glob
import json
import os
import re

VIEWS = ["D", "F", "L", "R", "U"]
FDI_RE = re.compile(r"^\d{2}$")


def valid_fdi(fdi):
    q, t = fdi // 10, fdi % 10
    return q in (1, 2, 3, 4) and 1 <= t <= 8


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_txt(path):
    renames, deletes = {}, set()
    if not os.path.exists(path):
        return renames, deletes
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh.read().splitlines():
            s = line.strip().replace(".", ",")
            if not s:
                continue
            if "," in s:
                a, b = s.split(",", 1)
                if a.strip().isdigit() and b.strip().isdigit():
                    renames[int(a)] = int(b)   # later line wins (last write)
            elif s.isdigit():
                deletes.add(int(s))
    return renames, deletes


def parse_labelme_boxes(path, ow, oh):
    """Return list of (fdi, [x1,y1,x2,y2]) rescaled to the original image."""
    a = load_json(path)
    lw, lh = a.get("imageWidth"), a.get("imageHeight")
    sx = ow / lw if lw else 1.0
    sy = oh / lh if lh else 1.0
    out = []
    for sh in a.get("shapes", []):
        lab = sh.get("label", "")
        if not (isinstance(lab, str) and FDI_RE.match(lab.strip())):
            continue
        fdi = int(lab.strip())
        if not valid_fdi(fdi):
            continue
        pts = sh.get("points", [])
        if len(pts) != 2:
            continue
        xs = [p[0] * sx for p in pts]
        ys = [p[1] * sy for p in pts]
        out.append((fdi, [min(xs), min(ys), max(xs), max(ys)]))
    return out


def clamp_box(box, ow, oh):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(x1, ow))
    x2 = max(0.0, min(x2, ow))
    y1 = max(0.0, min(y1, oh))
    y2 = max(0.0, min(y2, oh))
    return [x1, y1, x2, y2]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def make_corrected_tooth(fdi, box, case, view):
    """A tooth whose box comes from a manual labelme annotation (confidence unknown)."""
    return {
        "fdi": fdi,
        "bbox_yolo": [round(v, 4) for v in box],
        "confidence": None,
        "pad_ratio": 0.0,
        "bbox_padded": [int(round(v)) for v in box],
        "clamped": False,
        "crop_path": "single_tooth/%s/%s/%d.png" % (case, view, fdi),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adjust", default="intraoral_adjust/sample",
                    help="manual corrections root (<adjust>/<case>/<view>.{txt,json})")
    ap.add_argument("--pred", default="intraoral_adjust/unfiltered_bbox",
                    help="unfiltered predictions (<pred>/<case>/<view>.json); "
                         "the starting point that the adjustments are applied to")
    ap.add_argument("--zeyu", default="intraoral/zeyu",
                    help="original filtered tooth_bbox tree, used only for "
                         "orig_size / sextants / image_path metadata")
    ap.add_argument("--out", default="intraoral_adjust/tooth_bbox_adj")
    ap.add_argument("--reports", default="intraoral_adjust/reports",
                    help="folder for the provenance log")
    args = ap.parse_args()

    # case -> unfiltered prediction file dir
    pmap = {}
    for cdir in sorted(glob.glob(os.path.join(args.pred, "*"))):
        if os.path.isdir(cdir):
            pmap.setdefault(os.path.basename(cdir), cdir)
    # case -> filtered tooth_bbox dir (metadata source)
    zmap = {}
    for base in glob.glob(os.path.join(args.zeyu, "*_process", "tooth_bbox")):
        for c in os.listdir(base):
            zmap.setdefault(c, base)

    cases = sorted(d for d in os.listdir(args.adjust)
                   if os.path.isdir(os.path.join(args.adjust, d)))

    os.makedirs(args.reports, exist_ok=True)
    provenance = []          # per-tooth status rows
    warnings = []

    for case in cases:
        if case not in zmap:
            warnings.append("case %s not found in zeyu, skipped" % case)
            continue
        if case not in pmap:
            warnings.append("case %s has no unfiltered prediction, skipped" % case)
            continue
        for view in VIEWS:
            zpath = os.path.join(zmap[case], case, view + ".json")
            if not os.path.exists(zpath):
                warnings.append("missing original %s/%s" % (case, view))
                continue
            ppath = os.path.join(pmap[case], view + ".json")
            if not os.path.exists(ppath):
                warnings.append("missing unfiltered prediction %s/%s" % (case, view))
                continue
            Z = load_json(zpath)      # metadata: orig_size, sextants, image_path
            P = load_json(ppath)      # boxes: unfiltered detection
            ow, oh = Z["orig_size"]
            if list(P.get("orig_size", [ow, oh])) != [ow, oh]:
                warnings.append("orig_size mismatch %s/%s: unfiltered %s vs tooth_bbox %s"
                                % (case, view, P.get("orig_size"), [ow, oh]))

            renames, deletes = parse_txt(os.path.join(args.adjust, case, view + ".txt"))
            jpath = os.path.join(args.adjust, case, view + ".json")
            lm_boxes = parse_labelme_boxes(jpath, ow, oh) if os.path.exists(jpath) else []

            # result[fdi] = tooth_dict ; origin[fdi] = (status, orig_fdi)
            result = {}
            origin = {}
            zfdis = {t["fdi"] for t in P["teeth"]}

            # explicit reference checks (for the report)
            for src in renames:
                if src not in zfdis:
                    warnings.append("unmatched rename source %s/%s: %d" % (case, view, src))
            for src in deletes:
                if src not in zfdis:
                    warnings.append("unmatched delete %s/%s: %d" % (case, view, src))

            kept, renamed = [], []
            for t in P["teeth"]:
                f = t["fdi"]
                if f in deletes:
                    origin[f] = ("deleted", f)
                    continue
                nf = renames.get(f, f)
                if nf != f and not valid_fdi(nf):
                    warnings.append("invalid rename target %s/%s: %d -> %d (ignored)"
                                    % (case, view, f, nf))
                    nf = f
                (renamed if nf != f else kept).append((t, nf))

            # 1. place untouched teeth first ...
            for t, nf in kept:
                result[nf] = dict(t)
                origin[nf] = ("kept", nf)
            # 2. ... then explicit renames win over any kept tooth sharing the target
            for t, nf in renamed:
                if nf in result and origin[nf][0] == "kept":
                    # Normally a duplicate detection: the model emitted two boxes for
                    # one physical tooth under different FDIs, and the annotator cured
                    # it by relabelling one of them.  A high IoU with the box being
                    # replaced confirms that; a low IoU deserves a manual look.
                    score = iou(t["bbox_yolo"], result[nf]["bbox_yolo"])
                    warnings.append("rename %s/%s: %d -> %d overwrites predicted %d (IoU %.2f)"
                                    % (case, view, t["fdi"], nf, nf, score))
                tooth = dict(t)
                tooth["fdi"] = nf
                tooth["confidence"] = None
                tooth["crop_path"] = "single_tooth/%s/%s/%d.png" % (case, view, nf)
                result[nf] = tooth
                origin[nf] = ("renamed", t["fdi"])

            # 3. apply labelme boxes (json priority: overwrite)
            for fdi, box in lm_boxes:
                existed = fdi in result
                result[fdi] = make_corrected_tooth(fdi, box, case, view)
                origin[fdi] = ("relocated" if existed else "added", fdi)

            # emit sorted output
            teeth = [result[f] for f in sorted(result)]
            out = {
                "case_id": Z.get("case_id", case),
                "view": view,
                "image_path": Z.get("image_path", "process/%s/%s.png" % (case, view)),
                "orig_size": [ow, oh],
                "teeth": teeth,
                "sextants": Z.get("sextants", []),     # kept as-is
            }
            odir = os.path.join(args.out, case)
            os.makedirs(odir, exist_ok=True)
            with open(os.path.join(odir, view + ".json"), "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1)

            for f in sorted(result):
                status, src = origin[f]
                provenance.append({"case": case, "view": view, "fdi": f,
                                   "status": status, "orig_fdi": src})

    # provenance log (used by the accuracy estimator for reference)
    with open(os.path.join(args.reports, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=1)
    if warnings:
        with open(os.path.join(args.reports, "build_warnings.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(warnings))

    import collections
    c = collections.Counter(p["status"] for p in provenance)
    print("tooth status counts: %s" % dict(c))
    print("warnings: %d (see %s/build_warnings.txt)" % (len(warnings), args.reports))
    print("output ->", args.out)


if __name__ == "__main__":
    main()
