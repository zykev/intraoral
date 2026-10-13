#!/usr/bin/env python3
"""Visualise the manual tooth-bbox corrections.

Part of the YOLO11 fine-tuning data preparation: this is the human-review
overlay of every change the annotator made, so the GT can be eyeballed before
training on it.

For every (case, view) whose boxes were actually changed, draws on top of the
full-resolution ``process`` image:

    RED  rectangle : original model prediction, i.e. the boxes the annotator
                     saw in ``preview/`` -- these come from the **unfiltered**
                     detections in ``intraoral_adjust/unfiltered_bbox`` (no
                     per-view FDI whitelist), matching what ``preview/`` draws
    BLUE rectangle : corrected box  (``intraoral_adjust/tooth_bbox_adj``)

Output layout::

    <out>/<case>/<view>.png     e.g.  intraoral_adjust/vis_adj/1106/D.png

Only views with at least one change (added / removed / relabelled / moved box)
are rendered, so the folder doubles as a to-do list of what was touched.
Views whose predictions were left completely untouched are not rendered.

Requires Pillow (run with the ``intraoral`` conda env).
"""

import argparse
import glob
import json
import os
import struct

from PIL import Image, ImageDraw, ImageFont

VIEWS = ["D", "F", "L", "R", "U"]
RED = (255, 0, 0)
BLUE = (0, 80, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# --------------------------------------------------------------------------- io
def png_size(path):
    with open(path, "rb") as fh:
        fh.read(16)
        w, h = struct.unpack(">II", fh.read(8))
    return w, h


def load_teeth(path):
    with open(path, "r", encoding="utf-8") as fh:
        j = json.load(fh)
    return j, {t["fdi"]: list(map(float, t["bbox_yolo"])) for t in j["teeth"]}


def boxes_differ(a, b, tol=1.0):
    return any(abs(x - y) > tol for x, y in zip(a, b))


def view_changed(orig, adj, iou_eps=0.999):
    if set(orig) != set(adj):
        return True
    for fdi, box in orig.items():
        if boxes_differ(box, adj[fdi]):
            return True
    return False


# ----------------------------------------------------------------------- drawing
def get_font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_label(d, xy, text, fill, font, anchor="lt"):
    x, y = xy
    try:
        l, t, r, b = d.textbbox((x, y), text, font=font, anchor=anchor)
    except Exception:
        l, t, r, b = x, y, x + 8 * len(text), y + 12
    pad = 3
    d.rectangle([l - pad, t - pad, r + pad, b + pad], fill=fill)
    d.text((x, y), text, fill=WHITE, font=font, anchor=anchor)
    return b - t + 2 * pad


def draw_box(d, box, color, width, label=None, font=None, fh=0, label_at="top"):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    d.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label is None:
        return
    if label_at == "top":
        draw_label(d, (x1 + 2, max(0, y1 - fh - 6)), label, color, font, "lt")
    else:
        draw_label(d, (x1 + 2, min(y2 + 4, y1)), label, color, font, "lt")


def render(proc_png, orig, adj, font, fh, line_w, out_path, title, max_side=0):
    img = Image.open(proc_png).convert("RGB")
    W, H = img.width, img.height
    d = ImageDraw.Draw(img)

    # original predictions first (so corrected boxes sit on top)
    for fdi, box in sorted(orig.items()):
        draw_box(d, box, RED, line_w, "o%d" % fdi, font, fh, "top")
    for fdi, box in sorted(adj.items()):
        draw_box(d, box, BLUE, line_w, "j%d" % fdi, font, fh, "bottom")

    # legend
    h = fh + 10
    d.rectangle([0, 0, 250, 2 * h + 6], fill=(0, 0, 0))
    d.rectangle([6, 6, 6 + h, 6 + h], outline=RED, width=max(2, line_w // 2))
    d.text((12 + h, 6), "original (preview detection)", fill=WHITE, font=font)
    d.rectangle([6, 12 + h, 6 + h, 12 + 2 * h], outline=BLUE, width=max(2, line_w // 2))
    d.text((12 + h, 12 + h), "adjusted (tooth_bbox_adj)", fill=WHITE, font=font)

    # view caption bottom-right
    d.text((W - 8, H - 8), title, fill=WHITE, font=font, anchor="rb",
           stroke_width=2, stroke_fill=BLACK)

    if max_side and max(W, H) > max_side:
        s = max_side / max(W, H)
        img = img.resize((round(W * s), round(H * s)), Image.LANCZOS)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default="intraoral_adjust/unfiltered_bbox",
                    help="unfiltered predictions (<pred>/<case>/<view>.json); "
                         "the boxes preview/ draws and the annotator reviewed")
    ap.add_argument("--zeyu", default="intraoral/zeyu", help="folder holding *_process/process for the images")
    ap.add_argument("--adj", default="intraoral_adjust/tooth_bbox_adj", help="corrected bbox set")
    ap.add_argument("--out", default="intraoral_adjust/vis_adj",
                    help="output root (<out>/<case>/<view>.png)")
    ap.add_argument("--max-side", type=int, default=0,
                    help="downscale output so the longest side is at most this (0 = keep full resolution)")
    args = ap.parse_args()

    # index of unfiltered predictions + process images per batch
    omap, pmap = {}, {}
    for cdir in sorted(glob.glob(os.path.join(args.pred, "*"))):
        if os.path.isdir(cdir):
            omap.setdefault(os.path.basename(cdir), cdir)
    for base in glob.glob(os.path.join(args.zeyu, "*_process", "tooth_bbox")):
        proc_root = os.path.join(os.path.dirname(base), "process")
        for c in os.listdir(base):
            pmap.setdefault(c, proc_root)

    n_done = n_skip = n_missing = 0
    changed_views = []
    mismatches = []

    for ap_json in sorted(glob.glob(os.path.join(args.adj, "*", "*.json"))):
        case = os.path.basename(os.path.dirname(ap_json))
        view = os.path.splitext(os.path.basename(ap_json))[0]
        if view not in VIEWS or case not in omap:
            continue

        o_json = os.path.join(omap[case], view + ".json")
        proc_png = os.path.join(pmap.get(case, ""), case, view + ".png")
        if not os.path.exists(o_json) or not os.path.exists(proc_png):
            n_missing += 1
            continue

        Zj, orig = load_teeth(o_json)
        Aj, adj = load_teeth(ap_json)

        # ---- coordinate-space guard: adj must live in process-image space
        W, H = png_size(proc_png)
        if [W, H] != list(Aj["orig_size"]):
            mismatches.append("%s/%s adj orig_size %s != process png %s"
                              % (case, view, Aj["orig_size"], [W, H]))

        if not view_changed(orig, adj):
            n_skip += 1
            continue

        # relative line width so it looks the same at any resolution
        line_w = max(3, round(max(W, H) / 500))
        font_size = max(16, round(max(W, H) / 85))
        font = get_font(font_size)

        out_path = os.path.join(args.out, case, view + ".png")
        title = "%s / %s   orig:%d  adj:%d" % (case, view, len(orig), len(adj))
        render(proc_png, orig, adj, font, font_size, line_w, out_path, title,
               args.max_side)

        changed_views.append(case + "/" + view)
        n_done += 1

    print("rendered ............ %d view(s)" % n_done)
    print("unchanged (skipped) . %d" % n_skip)
    print("missing inputs ...... %d" % n_missing)
    if mismatches:
        print("!! coordinate-space mismatches: %d" % len(mismatches))
        for m in mismatches[:10]:
            print("   ", m)
    else:
        print("coordinate-space check: OK (all adj boxes are in process-image space)")
    print("output -> %s/<case>/<view>.png" % args.out)


if __name__ == "__main__":
    main()
