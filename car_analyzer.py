#!/usr/bin/env python3
"""Advanced vehicle analytics with annotation.

This script detects vehicles in a video, tracks them with DeepSORT and
annotates each track with:
    * car brand (if a model is supplied)
    * dominant colour
    * licence plate text and the German city inferred from its prefix
    * estimated speed relative to the camera vehicle.

Compared to a vanilla implementation it now:
    * considers cars, buses and trucks (COCO ids 2, 5, 7)
    * draws annotations using a larger font for readability
    * overlays the total number of unique vehicles seen so far
    * defaults to a camera speed of 100 km/h

Run ``python car_analyzer.py --help`` for full CLI usage.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------- Ultralytics YOLO -------------------------------------------------
try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - import failure messaging
    sys.exit(
        f"[FATAL] Ultralytics failed to import → {exc}\n"
        "        Run: pip install ultralytics"
    )

# ---------- EasyOCR ----------------------------------------------------------
try:
    import easyocr
except Exception as exc:  # pragma: no cover
    sys.exit(
        f"[FATAL] easyocr failed to import → {exc}\n"
        "        Run: pip install easyocr"
    )

# ---------- DeepSORT (modern & legacy namespaces) ---------------------------
DeepSort, Track = None, None
for mod_name in ("deep_sort_realtime.deepsort_tracker",  # ≥ 1.3
                 "deep_sort_realtime"):                  # ≤ 1.2
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        continue
    DeepSort = getattr(mod, "DeepSort", None)
    Track = getattr(mod, "Track", None)
    if DeepSort:
        break

if DeepSort is None:  # pragma: no cover
    sys.exit(
        "[FATAL] Cannot import DeepSort\n"
        "        • Activate the venv where you installed it\n"
        "        • Or (re-)install:  pip install -U deep_sort_realtime"
    )

# Silence torch pickle warnings when loading Re-ID model
import torch  # pylint: disable=wrong-import-position  # noqa: E402
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

# ---------- Colour lookup utilities ----------------------------------------
_COLOR_TABLE = {
    (255, 0, 0): "red",
    (0, 255, 0): "green",
    (0, 0, 255): "blue",
    (255, 255, 255): "white",
    (0, 0, 0): "black",
    (128, 128, 128): "gray",
    (0, 255, 255): "yellow",
    (255, 0, 255): "magenta",
    (255, 255, 0): "cyan",
}

_PLATE_PREFIX = {
    "B": "Berlin",
    "M": "München",
    "HH": "Hamburg",
    "S": "Stuttgart",
    "K": "Köln",
    "F": "Frankfurt a.M.",
}

def _closest_colour(bgr: tuple[int, int, int]) -> str:
    """Return nearest colour name for a BGR triple."""
    dist, name = min(
        (
            (np.linalg.norm(np.array(bgr) - np.array(rgb)), n)
            for rgb, n in _COLOR_TABLE.items()
        ),
        key=lambda t: t[0],
    )
    return name

def _dominant_colour(img_bgr: np.ndarray) -> str | None:
    """Return dominant colour name of an image region."""
    if img_bgr.size == 0:
        return None
    img_small = cv2.resize(img_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    pixels = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    try:
        from sklearn.cluster import KMeans  # pylint: disable=import-error
    except ImportError:
        return None
    kmeans = KMeans(n_clusters=3, n_init=3, random_state=0).fit(pixels)
    rgb = kmeans.cluster_centers_[np.bincount(kmeans.labels_).argmax()]
    return _closest_colour(tuple(int(c) for c in rgb[::-1]))

def _bbox_from_track(trk) -> tuple[int, int, int, int, int, int, int, int]:
    """Return (x1,y1,x2,y2,w,h,cx,cy) for any DeepSORT Track variant."""
    if hasattr(trk, "to_tlwh"):
        x, y, w, h = trk.to_tlwh()
    elif hasattr(trk, "to_ltwh"):
        x, y, w, h = trk.to_ltwh()
    elif hasattr(trk, "to_xyah"):
        cx, cy, w, h = trk.to_xyah()
        x, y = cx - w / 2, cy - h / 2
    elif hasattr(trk, "tlbr"):
        x1, y1, x2, y2 = trk.tlbr
        w, h = x2 - x1, y2 - y1
        x, y = x1, y1
    else:  # pragma: no cover - defensive branch
        raise AttributeError("Unsupported Track API — please update deep_sort_realtime")
    x1, y1, w, h = map(int, [x, y, w, h])
    return x1, y1, x1 + w, y1 + h, w, h, x1 + w // 2, y1 + h // 2

class CarAnalyzer:
    """Video processor performing per-vehicle analytics."""

    PIXEL_TO_METRE = 0.05  # adjust for camera geometry

    def __init__(
        self,
        video_path: str | Path,
        *,
        output_path: str | Path = "annotated_video.mp4",
        own_speed_kmh: float = 100.0,
        car_weights: str | Path = "yolov8n.pt",
        plate_weights: str | Path | None = None,
        make_weights: str | Path | None = None,
        embedder: bool = False,
    ) -> None:
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.own_speed_kmh = float(own_speed_kmh)

        # YOLO models ---------------------------------------------------------
        self.det = YOLO(car_weights)
        self.plate_det = (
            YOLO(str(plate_weights))
            if plate_weights and Path(plate_weights).is_file() else None
        )
        self.make_det = (
            YOLO(str(make_weights))
            if make_weights and Path(make_weights).is_file() else None
        )
        if not self.plate_det:
            print("[WARN] No licence-plate weights → plate OCR disabled.")
        if not self.make_det:
            print("[WARN] No car-brand weights → brand detection disabled.")

        # DeepSORT tracker ---------------------------------------------------
        self.embedder_on = bool(embedder)
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            embedder="mobilenet" if self.embedder_on else None,
            embedder_gpu=False,
        )

        # OCR ----------------------------------------------------------------
        self.ocr = easyocr.Reader(["de", "en"], gpu=False)

        # per-ID database ----------------------------------------------------
        self._info: dict[int, dict] = defaultdict(
            lambda: {"brand": None, "colour": None, "plate": None, "city": None, "speeds": []}
        )

    def process(self) -> dict:
        """Run analytics and produce annotated video and JSON summary."""
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
        )

        prev_cent: dict[int, np.ndarray] = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pred = self.det.predict(frame, classes=[2, 5, 7], conf=0.30, verbose=False)[0]
            dets, embeds = [], []
            for *xyxy, conf, cls in pred.boxes.data.tolist():
                x1, y1, x2, y2 = xyxy
                dets.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))
                embeds.append(
                    np.array(
                        [
                            (x1 + x2) / 2 / W,
                            (y1 + y2) / 2 / H,
                            (x2 - x1) / W,
                            (y2 - y1) / H,
                        ],
                        dtype=np.float32,
                    )
                )

            if self.embedder_on:
                tracks = self.tracker.update_tracks(dets, frame=frame)
            else:
                tracks = self.tracker.update_tracks(dets, embeds=embeds)

            for trk in tracks:
                if hasattr(trk, "is_confirmed") and not trk.is_confirmed():
                    continue
                tid = trk.track_id
                x1, y1, x2, y2, w, h, cx, cy = _bbox_from_track(trk)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)
                crop = frame[y1:y2, x1:x2]
                info = self._info[tid]

                if self.make_det and info["brand"] is None and crop.size:
                    mk = self.make_det.predict(crop, conf=0.25, verbose=False)[0]
                    if len(mk.boxes):
                        info["brand"] = mk.names[int(mk.boxes[0].cls)]

                if info["colour"] is None and crop.size:
                    info["colour"] = _dominant_colour(crop)

                if self.plate_det and info["plate"] is None and crop.size:
                    p = self.plate_det.predict(crop, conf=0.25, verbose=False)[0]
                    if len(p.boxes):
                        px1, py1, px2, py2 = map(int, p.boxes[0].xyxy[0])
                        px1, py1 = max(0, px1), max(0, py1)
                        px2, py2 = min(crop.shape[1] - 1, px2), min(crop.shape[0] - 1, py2)
                        plate_crop = crop[py1:py2, px1:px2]
                        if plate_crop.size:
                            txt = self.ocr.readtext(plate_crop)
                            if txt:
                                plate = txt[0][1].replace(" ", "").upper()
                                info["plate"] = plate
                                prefix = "".join(c for c in plate if c.isalpha())
                                for L in range(len(prefix), 0, -1):
                                    if prefix[:L] in _PLATE_PREFIX:
                                        info["city"] = _PLATE_PREFIX[prefix[:L]]
                                        break

                this_cent = np.array([cx, cy])
                if tid in prev_cent:
                    pix_per_frame = np.linalg.norm(this_cent - prev_cent[tid])
                    metres = pix_per_frame * self.PIXEL_TO_METRE
                    kmh = metres * fps * 3.6 + self.own_speed_kmh
                    info["speeds"].append(kmh)
                prev_cent[tid] = this_cent

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                parts = [f"ID:{tid}"]
                if info["brand"]:
                    parts.append(info["brand"])
                if info["colour"]:
                    parts.append(info["colour"])
                if info["plate"]:
                    parts.append(info["plate"])
                if info["city"]:
                    parts.append(info["city"])
                if info["speeds"]:
                    parts.append(f"{np.mean(info['speeds']):.1f} km/h")
                txt = " | ".join(parts)
                cv2.putText(
                    frame,
                    txt,
                    (x1, max(35, y1 - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    3,
                )

            cv2.putText(
                frame,
                f"Total vehicles: {len(self._info)}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )

            vw.write(frame)

        cap.release()
        vw.release()

        summary = {}
        for tid, d in self._info.items():
            summary[tid] = {k: (np.mean(v) if k == "speeds" and v else v) for k, v in d.items()}
        json_path = self.output_path.with_suffix(".json")
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[DONE] Video → {self.output_path} | JSON → {json_path}")
        return summary

def _parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Annotate a dash-cam video with car brand, dominant colour, "
            "(optional) licence-plate and speed."
        )
    )
    ap.add_argument("video", help="input video file (e.g. video.MOV)")
    ap.add_argument("--out", default="annotated_video.mp4", help="output MP4 path")
    ap.add_argument(
        "--speed",
        type=float,
        default=100,
        help="your own car's speed in km/h (for relative speed)",
    )
    ap.add_argument("--pt", dest="plate_wts", help="YOLO weights for licence-plate detector")
    ap.add_argument("--mt", dest="make_wts", help="YOLO weights for car-brand detector")
    ap.add_argument(
        "--embedder",
        action="store_true",
        help="enable Mobilenet appearance-based tracking (CPU gets slower; GPU recommended)",
    )
    return ap.parse_args()

if __name__ == "__main__":  # pragma: no cover - CLI entry
    args = _parse_cli()
    CarAnalyzer(
        args.video,
        output_path=args.out,
        own_speed_kmh=args.speed,
        plate_weights=args.plate_wts,
        make_weights=args.make_wts,
        embedder=args.embedder,
    ).process()
