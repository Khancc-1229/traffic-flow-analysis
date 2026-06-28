"""
可视化引擎 — 在每一帧上绘制检测、跟踪、速度、计数等所有叠加元素
"""

import cv2
import numpy as np
from datetime import timedelta
import config as cfg


class Visualizer:
    """帧绘制器: 将分析结果叠加到帧上"""

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_w = frame_width
        self.frame_h = frame_height

        # 预定义颜色
        self.color_white = (255, 255, 255)
        self.color_black = (0, 0, 0)
        self.color_red   = (0, 0, 255)
        self.color_green = (0, 255, 0)
        self.color_yellow = (0, 255, 255)
        self.color_cyan  = (255, 255, 0)
        self.color_gray  = (128, 128, 128)

    def draw_all(self, frame: np.ndarray, detections: list[dict],
                 tracker, speed_estimator, counter: "LineCounter",
                 frame_idx: int, timestamp_s: float, processing_fps: float) -> np.ndarray:
        """
        在帧上绘制所有叠加元素。

        参数:
            frame:            原始帧 (BGR)
            detections:       当前帧检测结果
            tracker:          Tracker 实例
            speed_estimator:  SpeedEstimator 实例
            counter:          LineCounter 实例
            frame_idx:        当前帧号
            timestamp_s:      时间戳 (秒)
            processing_fps:   实际处理速度 (FPS)

        返回:
            绘制后的帧 (BGR)
        """
        result = frame.copy()

        if cfg.SHOW_TRAILS:
            self._draw_trails(result, detections, tracker)

        if cfg.SHOW_BOXES:
            self._draw_boxes(result, detections, tracker, speed_estimator)

        if cfg.SHOW_COUNTING_LINE:
            self._draw_counting_line(result, counter)

        if cfg.SHOW_DASHBOARD:
            self._draw_dashboard(result, counter, speed_estimator, detections,
                                 frame_idx, timestamp_s, processing_fps)

        self._draw_status_bar(result, frame_idx, timestamp_s, processing_fps)

        return result

    # ---- 各元素绘制 ----

    def _draw_boxes(self, frame, detections, tracker, speed_estimator):
        """绘制检测框 + 标签 + ID + 速度"""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["cls_name"]
            conf = det["conf"]
            tid = det.get("track_id", -1)
            speed = speed_estimator.get_speed(tid) if tid >= 0 else 0.0

            # 类别颜色
            box_color = cfg.CLASS_COLORS.get(cls_name, self.color_green)

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, cfg.BOX_THICKNESS)

            # 组合标签文本
            label_parts = []
            if cfg.SHOW_LABELS:
                label_parts.append(cls_name)
            if cfg.SHOW_CONFIDENCE:
                label_parts.append(f"{conf:.2f}")
            if cfg.SHOW_TRACK_ID and tid >= 0:
                label_parts.append(f"ID:{tid}")
            if cfg.SHOW_SPEED and speed > 2.0:
                label_parts.append(f"{speed:.0f}km/h")

            if not label_parts:
                continue

            label = " | ".join(label_parts)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, cfg.FONT_SCALE, 2)

            # 标签背景
            label_y = max(y1 - th - 6, 0)
            cv2.rectangle(frame, (x1, label_y), (x1 + tw + 6, y1), box_color, -1)

            # 标签文字
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        font, cfg.FONT_SCALE, self.color_white, 2)

            # 速度颜色标注 (在框右上角画小色块)
            if cfg.SHOW_SPEED and speed > 2.0:
                speed_color = speed_estimator.get_speed_color(speed)
                sw, sh = 20, 6
                cv2.rectangle(frame, (x2 - sw, y1), (x2, y1 + sh), speed_color, -1)

    def _draw_trails(self, frame, detections, tracker):
        """绘制车辆轨迹线"""
        for det in detections:
            tid = det.get("track_id", -1)
            if tid < 0:
                continue

            trail = tracker.get_trail(tid, cfg.TRAIL_LENGTH)
            if len(trail) < 2:
                continue

            cls_name = det["cls_name"]
            color = cfg.CLASS_COLORS.get(cls_name, self.color_green)

            # 画轨迹折线 (渐透明效果: 远→近 由暗到亮)
            for i in range(1, len(trail)):
                alpha = i / len(trail)  # 0→1
                c = tuple(int(v * alpha) for v in color)
                cv2.line(frame, trail[i - 1], trail[i], c, 2, cv2.LINE_AA)

            # 在当前位置画圆点
            cx, cy = det["centroid"]
            cv2.circle(frame, (cx, cy), 4, color, -1)

    def _draw_counting_line(self, frame, counter):
        """绘制计数线和方向箭头"""
        p1 = tuple(map(int, counter.line_p1))
        p2 = tuple(map(int, counter.line_p2))

        # 虚线
        self._draw_dashed_line(frame, p1, p2, self.color_red, thickness=3,
                               dash_length=30)

        # 中点画方向指示箭头
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        line_vec = np.array(counter.line_vector, dtype=float)
        norm = np.linalg.norm(line_vec)
        if norm > 0:
            perp = np.array([-line_vec[1], line_vec[0]]) / norm * 40  # 垂线方向
            arrow_tip = (int(mid[0] + perp[0]), int(mid[1] + perp[1]))
            cv2.arrowedLine(frame, mid, arrow_tip, self.color_red, 2, tipLength=0.4)

    def _draw_dashboard(self, frame, counter, speed_estimator, detections,
                        frame_idx, timestamp_s, processing_fps):
        """绘制左上角信息面板"""
        # 半透明背景
        overlay = frame.copy()
        panel_w, panel_h = 380, 220
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h),
                      self.color_black, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h),
                      self.color_gray, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.55
        y = 32

        # 标题
        cv2.putText(frame, "Traffic Flow Analysis", (18, y), font, 0.65,
                    self.color_cyan, 2)
        y += 28

        # 分隔线
        cv2.line(frame, (18, y), (18 + panel_w - 10, y), self.color_gray, 1)
        y += 12

        # 车辆计数
        summary = counter.get_summary()
        cv2.putText(frame, f"Total Count: {summary['total']}", (18, y),
                    font, fs, self.color_white, 2)
        cv2.putText(frame, f"  Up: {summary['forward']}  |  Down: {summary['backward']}",
                    (18, y + 22), font, fs, self.color_white, 1)
        y += 48

        # 分类计数
        for cls_name, counts in summary.get("by_class", {}).items():
            total_cls = counts["forward"] + counts["backward"]
            cv2.putText(frame, f"  {cls_name}: {total_cls}", (18, y),
                        font, fs, self.color_white, 1)
            y += 20

        y += 10
        # 当前帧车流量 (辆/分钟)
        flow_rate = self._estimate_flow_rate(counter, timestamp_s)
        cv2.putText(frame, f"Flow Rate: {flow_rate:.1f} veh/min", (18, y),
                    font, fs, self.color_white, 2)
        y += 24

        # 平均速度
        avg_speed = speed_estimator.get_average_speed()
        speed_color = speed_estimator.get_speed_color(avg_speed)
        cv2.putText(frame, f"Avg Speed: {avg_speed:.1f} km/h", (18, y),
                    font, fs, speed_color, 2)

    def _draw_status_bar(self, frame, frame_idx, timestamp_s, processing_fps):
        """底部状态栏"""
        ts_str = str(timedelta(seconds=int(timestamp_s)))
        text = (f"Frame: {frame_idx}  |  Time: {ts_str}  |  "
                f"Processing: {processing_fps:.1f} FPS")

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.5
        (tw, th), _ = cv2.getTextSize(text, font, fs, 1)

        # 底部半透明条
        bar_y = self.frame_h - th - 12
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (self.frame_w, self.frame_h),
                      self.color_black, -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, text, (10, self.frame_h - 8),
                    font, fs, self.color_gray, 1)

    # ---- 工具方法 ----

    @staticmethod
    def _draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=20):
        """绘制虚线"""
        pt1 = np.array(pt1, dtype=float)
        pt2 = np.array(pt2, dtype=float)
        vec = pt2 - pt1
        total_len = np.linalg.norm(vec)
        if total_len == 0:
            return
        unit = vec / total_len
        pos = 0.0
        draw = True
        while pos < total_len:
            end = min(pos + dash_length, total_len)
            if draw:
                s = tuple(map(int, pt1 + unit * pos))
                e = tuple(map(int, pt1 + unit * end))
                cv2.line(img, s, e, color, thickness, cv2.LINE_AA)
            pos = end
            draw = not draw

    @staticmethod
    def _estimate_flow_rate(counter, current_ts: float) -> float:
        """估算当前车流量 (辆/分钟)"""
        if current_ts <= 0:
            return 0.0
        minutes = current_ts / 60.0
        if minutes < 0.1:
            return 0.0
        return counter.total_count / minutes
