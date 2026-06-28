"""
鸟瞰图速度估计器 — 透视变换后算速度, 消除梯形画面影响
"""

import cv2
import numpy as np
import config as cfg


class SpeedEstimator:
    def __init__(self, road_width_points=None, real_width=7.0, smooth_window=6):
        self.real_width = real_width
        self.smooth_window = smooth_window
        self.calibrated = False

        self.M = None           # 图像→鸟瞰图
        self.M_inv = None       # 鸟瞰图→图像
        self._be_w = 600        # 鸟瞰图分辨率(宽)
        self._be_h = 1200       # 鸟瞰图分辨率(高)
        self._m_per_px = 1.0    # 鸟瞰图中 1px = ?米

        self._speed_history: dict[int, list[float]] = {}
        self.track_speeds: dict[int, float] = {}
        self._positions: dict[int, list[tuple[float, float, int]]] = {}

        if road_width_points and len(road_width_points) == 4:
            self.set_calibration(road_width_points, real_width)

    def set_calibration(self, road_width_points, real_width):
        self.real_width = real_width
        fl, fr, nl, nr = [np.array(p, dtype=float) for p in road_width_points]

        # 源: 画面梯形 (左上, 右上, 右下, 左下)
        src = np.float32([fl, fr, nr, nl])

        # 真实路宽 = real_width 米, 鸟瞰图宽 = _be_w 像素
        # 鸟瞰图长按比例: _be_h = _be_w * (近处 y - 远处 y) / (远处左右像素差)
        px_far = np.linalg.norm(fr - fl)
        px_near = np.linalg.norm(nr - nl)
        # 用远/近 y 差估算鸟瞰图高度
        dy = abs((nl[1] + nr[1]) / 2 - (fl[1] + fr[1]) / 2)
        self._be_h = max(400, int(dy * 2))

        dst = np.float32([
            [0, 0], [self._be_w - 1, 0],
            [self._be_w - 1, self._be_h - 1], [0, self._be_h - 1]
        ])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

        # 鸟瞰图比例尺: 宽对应 real_width 米
        self._m_per_px = real_width / self._be_w
        self.calibrated = True

        print(f"[速度标定] 透视变换已建立")
        print(f"  鸟瞰图 {self._be_w}x{self._be_h} px")
        print(f"  比例尺: {self._m_per_px:.4f} m/px")

    def to_bird(self, xy):
        if self.M is None: return xy
        p = np.array([[xy]], dtype=np.float32)
        q = cv2.perspectiveTransform(p, self.M)
        return (float(q[0][0][0]), float(q[0][0][1]))

    def update(self, track_id, frame_idx, image_centroid, fps=60.0):
        if not self.calibrated:
            return 0.0
        bx, by = self.to_bird(image_centroid)

        if track_id not in self._positions:
            self._positions[track_id] = []
        self._positions[track_id].append((bx, by, frame_idx))
        while len(self._positions[track_id]) > self.smooth_window * 3:
            self._positions[track_id].pop(0)

        hist = self._positions[track_id]
        if len(hist) < self.smooth_window:
            return 0.0

        recent = hist[-self.smooth_window:]
        x1, y1, f1 = recent[0]
        x2, y2, f2 = recent[-1]

        dp_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        df = f2 - f1
        dt = df / fps if fps > 0 else 0
        if dt <= 0 or dp_px < 1.0:
            return 0.0

        dp_m = dp_px * self._m_per_px
        m_s = dp_m / dt
        kmh = m_s * 3.6

        if kmh < 2.0 or kmh > 200.0:
            kmh = 0.0

        if track_id not in self._speed_history:
            self._speed_history[track_id] = []
        hist = self._speed_history[track_id]
        hist.append(kmh)
        while len(hist) > self.smooth_window:
            hist.pop(0)

        smoothed = float(np.mean(hist))
        self.track_speeds[track_id] = smoothed
        return smoothed

    def get_speed(self, track_id):
        return self.track_speeds.get(track_id, 0.0)

    def get_all_speeds(self):
        return dict(self.track_speeds)

    def get_average_speed(self):
        speeds = [s for s in self.track_speeds.values() if s > 2.0]
        return float(np.mean(speeds)) if speeds else 0.0

    @staticmethod
    def get_speed_color(speed_kmh):
        if speed_kmh < 40: return (0, 255, 0)
        elif speed_kmh < 80: return (0, 255, 255)
        else: return (0, 0, 255)

    def reset(self):
        self._speed_history.clear()
        self.track_speeds.clear()
        self._positions.clear()
