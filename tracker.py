"""
目标跟踪器 — 维护每个 track 的历史轨迹, 用于速度估计和可视化
"""

from collections import defaultdict, OrderedDict
import numpy as np


class Tracker:
    """
    轨迹管理器。
    存储每个 track_id 的历史位置, 提供轨迹查询和位移计算。
    """

    def __init__(self, max_history: int = 60, lost_timeout: int = 30):
        """
        参数:
            max_history:  每个 track 最大保留帧数
            lost_timeout: track 丢失多少帧后移除
        """
        self.max_history = max_history
        self.lost_timeout = lost_timeout

        # track_id -> deque of (frame_idx, cx, cy, timestamp_seconds)
        self._history: dict[int, list] = defaultdict(list)

        # 记录每个 track 最后出现的帧
        self._last_seen_frame: dict[int, int] = {}

        # 当前帧号
        self._current_frame = 0

    def update(self, detections: list[dict], timestamp_s: float) -> None:
        """
        用当前帧的检测结果更新轨迹。

        参数:
            detections:  detector.detect_with_tracking() 的返回值
            timestamp_s: 当前帧的时间戳 (秒)
        """
        self._current_frame += 1
        active_ids = set()

        for det in detections:
            tid = det.get("track_id", -1)
            if tid < 0:
                continue

            active_ids.add(tid)
            cx, cy = det["centroid"]
            self._history[tid].append((self._current_frame, cx, cy, timestamp_s))

            # 限制历史长度
            while len(self._history[tid]) > self.max_history:
                self._history[tid].pop(0)

            self._last_seen_frame[tid] = self._current_frame

        # 清理超时 track
        lost_ids = []
        for tid in list(self._last_seen_frame.keys()):
            if tid not in active_ids:
                frames_since_last = self._current_frame - self._last_seen_frame[tid]
                if frames_since_last > self.lost_timeout:
                    lost_ids.append(tid)

        for tid in lost_ids:
            del self._history[tid]
            del self._last_seen_frame[tid]

    def get_trail(self, track_id: int, length: int = 20) -> list[tuple[int, int]]:
        """
        获取 track 最近 N 帧的轨迹点列表。

        返回:
            [(cx, cy), ...]  最近的在前面
        """
        if track_id not in self._history:
            return []
        points = [(cx, cy) for (_, cx, cy, _) in self._history[track_id]]
        return points[-length:]

    def get_displacement(self, track_id: int, window: int = 6) -> tuple[float, float]:
        """
        计算 track 在最近 window 帧内的像素位移。

        返回:
            (dx_pixels, dt_seconds) — 像素位移和对应的时间差
            如果历史不够则返回 (0.0, 0.0)
        """
        hist = self._history.get(track_id, [])
        if len(hist) < window:
            return 0.0, 0.0

        # 取 window 帧的范围
        recent = hist[-window:]
        first_frame, first_cx, first_cy, first_ts = recent[0]
        last_frame, last_cx, last_cy, last_ts = recent[-1]

        dx = np.sqrt((last_cx - first_cx) ** 2 + (last_cy - first_cy) ** 2)
        dt = last_ts - first_ts

        if dt <= 0:
            return 0.0, 0.0

        return float(dx), float(dt)

    def get_speed_history(self, track_id: int, window: int = 6) -> float:
        """
        获取最近 window 帧内的平均像素速度 (px/s)。
        """
        dx, dt = self.get_displacement(track_id, window)
        if dt <= 0:
            return 0.0
        return dx / dt

    def is_active(self, track_id: int) -> bool:
        """检查 track 是否仍然活跃"""
        return (track_id in self._last_seen_frame and
                self._current_frame - self._last_seen_frame[track_id] <= self.lost_timeout)

    @property
    def active_track_ids(self) -> list[int]:
        """返回当前活跃的 track ID 列表"""
        return [tid for tid in self._last_seen_frame if self.is_active(tid)]

    @property
    def total_tracks_seen(self) -> int:
        """历史中出现过的总 track 数"""
        return len(self._history)
