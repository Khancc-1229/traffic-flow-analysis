"""
越线计数器 — 基于虚拟线的双向车流量计数
"""

import numpy as np


class LineCounter:
    def __init__(self, line_p1, line_p2):
        self.line_p1 = np.array(line_p1, dtype=float)
        self.line_p2 = np.array(line_p2, dtype=float)
        self.line_vector = self.line_p2 - self.line_p1

        self.total_count = 0
        self.forward_count = 0
        self.backward_count = 0
        self.class_counts: dict[str, dict] = {}
        self.time_series: list[dict] = []
        self._counted_tracks: set[int] = set()
        self._prev_positions: dict[int, np.ndarray] = {}

    def update(self, detections, frame_idx, timestamp_s):
        events = []
        for det in detections:
            tid = det.get("track_id", -1)
            if tid < 0 or tid in self._counted_tracks:
                continue
            centroid = np.array(det["centroid"], dtype=float)

            if tid in self._prev_positions:
                prev = self._prev_positions[tid]
                cross_prev = np.cross(self.line_vector, prev - self.line_p1)
                cross_curr = np.cross(self.line_vector, centroid - self.line_p1)
                if cross_prev * cross_curr < 0 and \
                   self._on_segment(centroid) and \
                   self._on_segment(prev):
                    direction = "forward" if cross_curr > 0 else "backward"
                    cls_name = det["cls_name"]
                    self._counted_tracks.add(tid)
                    self.total_count += 1
                    if direction == "forward":
                        self.forward_count += 1
                    else:
                        self.backward_count += 1
                    if cls_name not in self.class_counts:
                        self.class_counts[cls_name] = {"forward": 0, "backward": 0}
                    self.class_counts[cls_name][direction] += 1
                    events.append({
                        "track_id": tid, "direction": direction,
                        "cls_name": cls_name, "frame": frame_idx,
                        "timestamp": timestamp_s,
                    })
            self._prev_positions[tid] = centroid

        return events

    def _on_segment(self, point, margin=80):
        """点必须在画面上与线段垂直距离 <= margin 像素, 才算在线段上"""
        ap = point - self.line_p1
        ab = self.line_vector
        t = np.dot(ap, ab) / np.dot(ab, ab) if np.dot(ab, ab) > 0 else 0
        t = np.clip(t, 0.0, 1.0)
        closest = self.line_p1 + t * ab
        return np.linalg.norm(point - closest) <= margin

    def set_line(self, p1, p2):
        self.line_p1 = np.array(p1, dtype=float)
        self.line_p2 = np.array(p2, dtype=float)
        self.line_vector = self.line_p2 - self.line_p1

    def reset(self):
        self.total_count = 0
        self.forward_count = 0
        self.backward_count = 0
        self.class_counts.clear()
        self._counted_tracks.clear()
        self._prev_positions.clear()
        self.time_series.clear()

    def get_summary(self):
        return {
            "total": self.total_count,
            "forward": self.forward_count,
            "backward": self.backward_count,
            "by_class": self.class_counts,
        }
