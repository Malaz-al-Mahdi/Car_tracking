import argparse
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from pymongo import MongoClient

# Simple mapping from YOLO class IDs to human friendly labels
COCO_LABELS = {
    2: "car",
    5: "bus",
    7: "truck",
}


def detect_brand(img):
    """Placeholder brand classifier.

    This function can be replaced with a proper car brand classifier.
    For now, it always returns "unknown".
    """
    return "unknown"


def detect_color(img):
    data = img.reshape((-1, 3))
    data = np.float32(data)
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, palette = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant = palette[0].astype(int)
    return "#%02x%02x%02x" % tuple(dominant)


def car_type(yolo_id):
    return COCO_LABELS.get(yolo_id, "unknown")


def parse_args():
    parser = argparse.ArgumentParser(description="Car tracking and analysis")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--mongo-uri", default="mongodb+srv://skymayfly:8SI9QPMdIjuUi0m1@cluster0.ehchxvi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0", help="MongoDB URI")
    parser.add_argument("--collection", default="car_tracking.results", help="MongoDB collection")
    return parser.parse_args()


def main():
    args = parse_args()

    client = MongoClient(args.mongo_uri)
    db_name, coll_name = args.collection.split(".", 1)
    collection = client[db_name][coll_name]

    yolo_model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tracks = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)[0]
        detections = []
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            label = int(cls)
            if label in COCO_LABELS:
                x1, y1, x2, y2 = box
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

        track_outputs = tracker.update_tracks(detections, frame=frame)

        for trk in track_outputs:
            if not trk.is_confirmed():
                continue
            tid = trk.track_id
            l, t, w, h = trk.to_ltwh()
            center = (int(l + w / 2), int(t + h / 2))
            crop = frame[int(t):int(t + h), int(l):int(l + w)]
            ctype = car_type(trk.det_class)
            brand = detect_brand(crop)
            color = detect_color(crop) if crop.size else "unknown"

            prev = tracks.get(tid)
            speed = 0
            if prev:
                dist = np.linalg.norm(np.array(center) - np.array(prev["center"]))
                time = (trk.age - prev["age"]) / fps
                speed = dist / time if time > 0 else 0
            tracks[tid] = {"center": center, "age": trk.age}

            entry = {
                "track_id": tid,
                "bbox": [l, t, w, h],
                "type": ctype,
                "brand": brand,
                "color": color,
                "speed_px_s": speed,
            }
            collection.insert_one(entry)

    cap.release()


if __name__ == "__main__":
    main()
