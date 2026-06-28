"""
交通流量分析 — 主流程
=====================
完整 pipeline: 标定 → 检测 → 跟踪 → 计数 → 测速 → 可视化 → 统计

运行模式:
    python main.py              正常模式 (使用已有标定)
    python main.py --calibrate  交互标定模式
"""

import os
import sys
import time
import json
import argparse
import cv2
import numpy as np
from collections import defaultdict

import config as cfg
from detector import VehicleDetector
from tracker import Tracker
from counter import LineCounter
from speed_estimator import SpeedEstimator
from visualizer import Visualizer


# ============================================================
# 交互标定
# ============================================================

class Calibrator:
    """鼠标交互标定:
    步骤1: 画计数线 (2 点)
    步骤2: 标路宽 — 远处 2 点 + 近处 2 点 + 输入宽度
    步骤3: 圈 ROI — 顺时针圈出你要检测的车道区域 (4+ 点)
    """

    def __init__(self, frame: np.ndarray):
        self.original = frame.copy()
        self.current_points: list[tuple[int, int]] = []
        self.counting_line_points: list[tuple[int, int]] = []
        self.road_width_points: list[tuple[int, int]] = []
        self.roi_points: list[tuple[int, int]] = []
        self.step = 1
        self.real_width = 7.0
        self.done = False
        self.message = ""

    def run(self) -> bool:
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration", 1280, 720)
        cv2.setMouseCallback("Calibration", self._mouse_callback)

        print("\n" + "=" * 55)
        print("  标定模式")
        print("=" * 55)
        print("  步骤1: 点 2 个点画横穿马路的计数线")
        print("  步骤2: 点 4 个点标道路宽度 + 输入宽度(米)")
        print("  步骤3: 顺时针点 4+ 个点圈出你要的车道")
        print("         (框外的车不检测, 过滤对面车道)")
        print("=" * 55)
        print("  操作: 左键=加点 | Enter=确认 | Backspace=撤销 | ESC=取消")
        print("=" * 55 + "\n")

        while not self.done:
            display = self._render()
            cv2.imshow("Calibration", display)
            key = cv2.waitKey(50) & 0xFF

            if key == 27:
                cv2.destroyWindow("Calibration")
                print("[标定] 已取消")
                return False
            elif key == 13:
                self._advance()
            elif key == 8 or key == 127:
                if self.current_points:
                    pt = self.current_points.pop()
                    print(f"  撤销: {pt}")
            elif key == ord('r'):
                self.current_points.clear()
                print("  清除当前步骤所有点")

        cv2.destroyAllWindows()
        cv2.waitKey(1)

        if len(self.counting_line_points) == 2 and len(self.road_width_points) == 4 and len(self.roi_points) >= 3:
            cfg.COUNTING_LINE = self.counting_line_points
            cfg.ROAD_WIDTH_POINTS = self.road_width_points
            cfg.ROAD_REAL_WIDTH = self.real_width
            cfg.ROI_POINTS = self.roi_points
            cfg.save_calibration()
            print(f"\n[标定] 完成!")
            print(f"  计数线: {self.counting_line_points[0]} → {self.counting_line_points[1]}")
            print(f"  道路宽度: {self.real_width} 米")
            print(f"  ROI: {len(self.roi_points)} 个点")
            return True
        return False

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            max_map = {1: 2, 2: 4, 3: 99}  # 步骤3不限点数
            max_pts = max_map.get(self.step, 4)
            if len(self.current_points) < max_pts:
                self.current_points.append((x, y))
                if self.step == 1:
                    print(f"  步骤1 — 点{len(self.current_points)}: ({x}, {y})")
                elif self.step == 2:
                    labels = ["远处左", "远处右", "近处左", "近处右"]
                    i = len(self.current_points) - 1
                    print(f"  步骤2 — {labels[i]}: ({x}, {y})")
                else:
                    print(f"  步骤3 — ROI点{len(self.current_points)}: ({x}, {y})")

    def _advance(self):
        if self.step == 1:
            if len(self.current_points) < 2: return
            self.counting_line_points = self.current_points.copy()
            print(f"  计数线: {self.counting_line_points[0]} → {self.counting_line_points[1]}")
            print(f"  → 步骤2")
            self.step = 2
            self.current_points = []
            self.message = ""

        elif self.step == 2:
            if len(self.current_points) < 4: return
            self.road_width_points = self.current_points.copy()
            try:
                w = input(f"\n  这是几条车道? 道路宽度多少米? [默认 {self.real_width}m]: ")
                if w.strip(): self.real_width = float(w)
            except (ValueError, EOFError): pass
            print(f"  道路宽度: {self.real_width} 米")
            print(f"  → 步骤3: 顺时针圈出你要检测的车道 (点完 4+ 个点按 Enter)")
            self.step = 3
            self.current_points = []
            self.message = ""

        elif self.step == 3:
            if len(self.current_points) < 3: return
            self.roi_points = self.current_points.copy()
            self.done = True

    def _render(self) -> np.ndarray:
        display = self.original.copy()
        h, w = display.shape[:2]

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        titles = {
            1: ("Step 1/3: COUNTING LINE — 2 points across the road", ""),
            2: ("Step 2/3: ROAD WIDTH — FL,FR = far edge | NL,NR = near edge", ""),
            3: ("Step 3/3: ROI POLYGON — Clockwise around YOUR lane (4+ points)", ""),
        }
        title, _ = titles.get(self.step, ("", ""))
        cv2.putText(display, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2)
        cv2.putText(display, "L-click:add | Enter:confirm | Backspace:undo | ESC:quit",
                    (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # 步骤1: 计数线 (红色)
        if self.step >= 1:
            pts = self.counting_line_points if self.step > 1 else self.current_points
            if self.step == 1:
                for i, pt in enumerate(pts):
                    cv2.circle(display, pt, 8, (0, 0, 255), -1)
                    cv2.putText(display, f"P{i+1}", (pt[0]+12, pt[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if len(pts) >= 2:
                    cv2.line(display, pts[0], pts[1], (0, 0, 255), 3)

        # 步骤2: 路宽 (橙色)
        if self.step >= 2:
            pts = self.road_width_points if self.step > 2 else self.current_points
            if self.step == 2:
                labels = ["FL", "FR", "NL", "NR"]
                for i, pt in enumerate(pts):
                    cv2.circle(display, pt, 8, (255, 150, 0), -1)
                    cv2.putText(display, labels[i], (pt[0]+12, pt[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)
                if len(pts) >= 2: cv2.line(display, pts[0], pts[1], (255, 150, 0), 2)
                if len(pts) >= 4: cv2.line(display, pts[2], pts[3], (255, 150, 0), 2)

        # 步骤3: ROI (绿色多边形)
        if self.step >= 3:
            pts = self.roi_points if self.step > 3 else self.current_points
            if self.step == 3:
                for pt in pts:
                    cv2.circle(display, pt, 6, (0, 255, 0), -1)
                if len(pts) >= 3:
                    cv2.polylines(display, [np.array(pts, dtype=np.int32)],
                                  True, (0, 255, 0), 2)

        if self.message:
            cv2.putText(display, self.message, (12, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return display


# ============================================================
# 数据收集 (用于离线统计)
# ============================================================

class DataCollector:
    """收集逐帧统计数据, 供 analytics 使用"""

    def __init__(self):
        # 时间序列: [{timestamp, total_count, forward, backward, avg_speed, detections_count}]
        self.time_series: list[dict] = []

        # 所有计数事件
        self.count_events: list[dict] = []

        # 所有车辆的速度数据: {track_id: [speed_samples]}
        self.speed_samples: dict[int, list[float]] = defaultdict(list)

        self.frame_count = 0
        self.total_detections = 0

        # 已越线的 track ID (导出时过滤未越线的)
        self.counted_track_ids: set[int] = set()

    def record_frame(self, timestamp_s: float, counter: LineCounter,
                     speed_estimator: SpeedEstimator, detections: list[dict]):
        self.frame_count += 1
        self.total_detections += len(detections)

        # 记录速度样本
        for tid, speed in speed_estimator.get_all_speeds().items():
            if speed > 2.0:
                self.speed_samples[tid].append(speed)

    def add_count_events(self, events: list[dict]):
        self.count_events.extend(events)

    def get_snapshot(self, timestamp_s: float, counter: LineCounter,
                     speed_estimator: SpeedEstimator) -> dict:
        """生成当前时刻统计快照"""
        return {
            "timestamp": timestamp_s,
            "total_count": counter.total_count,
            "forward": counter.forward_count,
            "backward": counter.backward_count,
            "avg_speed": speed_estimator.get_average_speed(),
            "class_counts": {
                cls: {"forward": c["forward"], "backward": c["backward"]}
                for cls, c in counter.class_counts.items()
            },
        }

    def export(self) -> dict:
        """导出: 剔除异常 track (max>150 = 跟踪跳变垃圾), 其余全保留"""
        valid_speeds = {}
        for tid, speeds in self.speed_samples.items():
            if not speeds: continue
            if max(speeds) <= 150:
                valid_speeds[str(tid)] = speeds
        all_speeds = []
        for speeds in valid_speeds.values():
            all_speeds.extend(speeds)

        return {
            "frame_count": self.frame_count,
            "total_detections": self.total_detections,
            "count_events": self.count_events,
            "speed_samples": valid_speeds,
            "all_speeds": all_speeds,
            "time_series": self.time_series,
        }


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="交通流量分析")
    parser.add_argument("--calibrate", action="store_true", help="强制进入标定模式")
    parser.add_argument("--model", choices=["v8s", "v8m", "v8x"], default=None,
                        help="模型选择: v8s / v8m / v8x")
    parser.add_argument("--no-charts", action="store_true", help="不生成图表")
    args = parser.parse_args()

    # ---- 1. 标定 (标定时跳过模型选择) ----
    if args.calibrate or not cfg.load_calibration()[0]:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cap = cv2.VideoCapture(cfg.VIDEO_PATH)
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            print("[错误] 无法读取视频")
            return

        # 缩放到处理分辨率
        first_frame = cv2.resize(first_frame, (cfg.PROCESS_WIDTH, cfg.PROCESS_HEIGHT))

        cal = Calibrator(first_frame)
        if not cal.run():
            print("[标定] 标定已取消")
            return

        # 重新加载标定
        cfg.load_calibration()
        print("[标定] 标定完成，退出。运行 python main.py --model v8s 开始处理")
        return

    # ---- 2. 选择模型 ----
    model_tag = args.model
    if model_tag is None:
        print("\n请选择模型:")
        print("  1. v8s (轻量快速)")
        print("  2. v8m (精度更高)")
        print("  3. v8x (最强精度)")
        try:
            choice = input("输入 1/2/3: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消")
            return
        model_tag = {"1": "v8s", "2": "v8m", "3": "v8x"}.get(choice, "v8s")
    if model_tag == "v8m":
        cfg.MODEL_NAME = "yolov8m.pt"
    elif model_tag == "v8x":
        cfg.MODEL_NAME = "yolov8x.pt"
    else:
        cfg.MODEL_NAME = "yolov8s.pt"
    cfg.OUTPUT_VIDEO_PATH = f"output/result_{model_tag}.mp4"
    cfg.CHARTS_DIR = f"output/charts_{model_tag}"
    cfg.REPORT_PATH = f"output/report_{model_tag}.json"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.CHARTS_DIR, exist_ok=True)

    print(f"[主流程] 模型: {model_tag}")

    # ---- 3. 初始化各模块 ----
    print("\n" + "=" * 60)
    print("  交通流量分析系统")
    print("=" * 60)

    detector = VehicleDetector()
    tracker = Tracker(max_history=60, lost_timeout=30)
    counter = LineCounter(cfg.COUNTING_LINE[0], cfg.COUNTING_LINE[1])
    speed_est = SpeedEstimator(
        road_width_points=cfg.ROAD_WIDTH_POINTS,
        real_width=cfg.ROAD_REAL_WIDTH,
        smooth_window=cfg.SPEED_SMOOTH_WINDOW,
    )
    collector = DataCollector()

    # 打开视频
    cap = cv2.VideoCapture(cfg.VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # 输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(
        cfg.OUTPUT_VIDEO_PATH, fourcc, original_fps,
        (cfg.PROCESS_WIDTH, cfg.PROCESS_HEIGHT)
    )

    visualizer = Visualizer(cfg.PROCESS_WIDTH, cfg.PROCESS_HEIGHT)

    print(f"[主流程] 视频: {cfg.VIDEO_PATH}")
    print(f"[主流程] 总帧数: {total_frames}, FPS: {original_fps:.2f}")
    print(f"[主流程] 处理分辨率: {cfg.PROCESS_WIDTH}x{cfg.PROCESS_HEIGHT}")
    print(f"[主流程] 输出: {cfg.OUTPUT_VIDEO_PATH}")
    print("=" * 60)

    # ---- 3. 逐帧处理 ----
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 跳帧
        if frame_idx % cfg.SKIP_FRAMES != 0:
            frame_idx += 1
            continue

        # 缩放
        frame = cv2.resize(frame, (cfg.PROCESS_WIDTH, cfg.PROCESS_HEIGHT))
        timestamp_s = frame_idx / original_fps

        # 检测 + 跟踪
        detections = detector.detect_with_tracking(frame)

        # 更新跟踪器
        tracker.update(detections, timestamp_s)

        # 更新速度估计 (路宽 y 轴插值)
        for det in detections:
            tid = det.get("track_id", -1)
            if tid < 0:
                continue
            speed_est.update(tid, frame_idx, det["centroid"], fps=original_fps)

        # 越线计数
        events = counter.update(detections, frame_idx, timestamp_s)
        if events:
            collector.add_count_events(events)

        # 收集数据
        collector.record_frame(timestamp_s, counter, speed_est, detections)

        # 定期记录时间序列快照 (每 STATS_INTERVAL_SECONDS 秒)
        if frame_idx % max(1, int(cfg.STATS_INTERVAL_SECONDS * original_fps)) == 0:
            snapshot = collector.get_snapshot(timestamp_s, counter, speed_est)
            collector.time_series.append(snapshot)

        # 可视化
        elapsed = time.time() - start_time
        proc_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        annotated = visualizer.draw_all(
            frame, detections, tracker, speed_est, counter,
            frame_idx, timestamp_s, proc_fps
        )

        # 写入输出
        out_writer.write(annotated)

        frame_idx += 1

        # 进度
        if frame_idx % 100 == 0:
            pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            print(f"\r[进度] {frame_idx}/{total_frames} ({pct:.1f}%)  "
                  f"| 检测: {counter.total_count} 辆  |  处理速度: {proc_fps:.1f} FPS",
                  end="", flush=True)

    # ---- 4. 收尾 ----
    total_time = time.time() - start_time
    cap.release()
    out_writer.release()

    print(f"\n\n[完成] 处理了 {frame_idx} 帧, 耗时 {total_time:.0f} 秒 "
          f"({frame_idx / total_time:.1f} FPS)")
    print(f"[完成] 输出视频: {cfg.OUTPUT_VIDEO_PATH}")

    # 导出原始数据
    data = collector.export()
    data["summary"] = counter.get_summary()
    data["processing_time"] = total_time
    data["avg_processing_fps"] = frame_idx / total_time if total_time > 0 else 0

    with open(cfg.REPORT_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[完成] 报告: {cfg.REPORT_PATH}")

    # ---- 5. 生成图表 ----
    if not args.no_charts:
        print("\n生成分析图表...")
        try:
            from analytics import generate_charts
            generate_charts(collector, counter, speed_est)
            print(f"[完成] 图表: {cfg.CHARTS_DIR}/")
        except Exception as e:
            print(f"[警告] 图表生成失败: {e}")

    # ---- 6. 打印摘要 ----
    print_summary(counter, speed_est, collector, total_time)


def print_summary(counter: LineCounter, speed_est: SpeedEstimator,
                  collector: DataCollector, total_time: float):
    """打印分析摘要到控制台"""
    print("\n" + "=" * 60)
    print("  交 通 流 量 分 析 摘 要")
    print("=" * 60)

    summary = counter.get_summary()
    print(f"\n  总计数: {summary['total']} 辆")
    print(f"    正向: {summary['forward']} 辆")
    print(f"    反向: {summary['backward']} 辆")

    print(f"\n  分类统计:")
    for cls_name, counts in summary.get("by_class", {}).items():
        total_cls = counts["forward"] + counts["backward"]
        print(f"    {cls_name}: {total_cls} 辆 "
              f"(↑{counts['forward']} ↓{counts['backward']})")

    # 速度统计
    all_speeds = [s for s in collector.export()["all_speeds"] if s > 2.0]
    if all_speeds:
        print(f"\n  速度统计:")
        print(f"    平均速度: {np.mean(all_speeds):.1f} km/h")
        print(f"    最高速度: {np.max(all_speeds):.1f} km/h")
        print(f"    最低速度: {np.min(all_speeds):.1f} km/h")
        print(f"    中位速度: {np.median(all_speeds):.1f} km/h")

    # 流量
    minutes = total_time / 60.0
    if minutes > 0:
        print(f"\n  平均车流量: {summary['total'] / minutes:.1f} 辆/分钟")

    print(f"\n  总处理时间: {total_time:.0f} 秒")
    print("=" * 60)


if __name__ == "__main__":
    main()
