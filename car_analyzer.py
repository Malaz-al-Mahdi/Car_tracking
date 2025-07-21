# Car Analyzer class for video car detection and tracking
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Mapping YOLO class IDs to human labels
COCO_LABELS = {
    2: "car",
    5: "bus",
    7: "truck",
}

def detect_brand(img):
    """Placeholder brand classifier."""
    return "unknown"

def detect_color(img):
    data = img.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, palette = cv2.kmeans(data, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant = palette[0].astype(int)
    return "#%02x%02x%02x" % tuple(dominant)

def drive_assessment(speed_px_s, threshold=100):
    """Return a simple speed assessment."""
    return "fast" if speed_px_s > threshold else "normal"

def car_type(yolo_id):
    return COCO_LABELS.get(yolo_id, "unknown")

class CarAnalyzer:
    """Detects, tracks and annotates cars in a video."""

    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.tracker = DeepSort(max_age=30)
        self.tracks = {}
        self.results = []
        self.stats = {
            "brands": {},
            "colors": {},
        }

    def _annotate(self, frame, bbox, text):
        l, t, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
        cv2.putText(frame, text, (l, t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def _draw_stats(self, frame):
        lines = [
            f"Cars: {sum(self.stats['brands'].values())}",
            "Brands: " + ", ".join(f"{b}:{c}" for b, c in self.stats['brands'].items()),
            "Colors: " + ", ".join(f"{c}:{n}" for c, n in self.stats['colors'].items()),
        ]
        x, y = 5, frame.shape[0] - 40
        for line in lines:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 15

    def process_frame(self, frame, fps):
        results = self.model(frame)[0]
        detections = []
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            label = int(cls)
            if label in COCO_LABELS:
                x1, y1, x2, y2 = box
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))
        track_outputs = self.tracker.update_tracks(detections, frame=frame)

        annotated = frame.copy()
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

            prev = self.tracks.get(tid)
            speed = 0
            if prev:
                dist = np.linalg.norm(np.array(center) - np.array(prev["center"]))
                time = (trk.age - prev["age"]) / fps
                speed = dist / time if time > 0 else 0
            self.tracks[tid] = {"center": center, "age": trk.age}

            assessment = drive_assessment(speed)

            entry = {
                "track_id": tid,
                "bbox": [int(l), int(t), int(w), int(h)],
                "type": ctype,
                "brand": brand,
                "color": color,
                "speed_px_s": float(speed),
                "assessment": assessment,
            }
            self.results.append(entry)

            # Update stats
            self.stats["brands"][brand] = self.stats["brands"].get(brand, 0) + 1
            self.stats["colors"][color] = self.stats["colors"].get(color, 0) + 1

            label = f"{ctype} {speed:.1f}px/s {assessment}"
            self._annotate(annotated, [l, t, w, h], label)

        self._draw_stats(annotated)
        return annotated

    def analyze(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated = self.process_frame(frame, fps)
            if writer:
                writer.write(annotated)
        cap.release()
        if writer:
            writer.release()
        return self.results

    def analyze_dashboard(self, video_path, output_path=None, show=True):
        """Analyze video and optionally show a dashboard window."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated = self.process_frame(frame, fps)
            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("Dashboard", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        return self.results

    def summary(self):
        return {
            "count": len(self.tracks),
            "brands": dict(self.stats["brands"]),
            "colors": dict(self.stats["colors"]),
        }
