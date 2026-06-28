"""
车辆检测器 — 封装 YOLO 模型, 提供车辆检测接口
"""

import numpy as np
import cv2
from ultralytics import YOLO
import config as cfg


class VehicleDetector:
    """YOLO 车辆检测器"""

    def __init__(self):
        print(f"[检测器] 加载模型: {cfg.MODEL_NAME} (device={cfg.DEVICE})")
        self.model = YOLO(cfg.MODEL_NAME)
        self.model.to(cfg.DEVICE)
        # ROI 多边形 (过滤对面车道)
        self.roi = np.array(cfg.ROI_POINTS, dtype=np.int32) if cfg.ROI_POINTS else None
        if self.roi is not None and len(self.roi) >= 3:
            print(f"[检测器] ROI 已启用: {len(self.roi)} 个点")
        # 预热
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print(f"[检测器] 模型加载完成, 车辆类别: {list(cfg.VEHICLE_CLASSES.values())}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        检测一帧中的车辆。

        参数:
            frame: BGR 图像 (H, W, 3)

        返回:
            检测结果列表, 每项 dict:
                bbox:     (x1, y1, x2, y2) — 像素坐标
                centroid: (cx, cy)         — 中心点
                conf:     float            — 置信度
                cls_id:   int              — 类别 ID
                cls_name: str              — 类别名称
        """
        results = self.model(frame, conf=cfg.CONFIDENCE_THRESHOLD,
                             iou=cfg.IOU_THRESHOLD, verbose=False)

        detections = []
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())

            # 只保留车辆类别
            if cls_id not in cfg.VEHICLE_CLASS_IDS:
                continue

            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = xyxy.astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                "bbox":     (int(x1), int(y1), int(x2), int(y2)),
                "centroid": (int(cx), int(cy)),
                "conf":     conf,
                "cls_id":   cls_id,
                "cls_name": cfg.VEHICLE_CLASSES[cls_id],
            })

        return detections

    def detect_with_tracking(self, frame: np.ndarray):
        """
        检测 + 跟踪一帧 (使用 Bot-SORT)。

        返回:
            (detections, tracking_data)
            tracking_data: list[dict] 额外包含 track_id
        """
        results = self.model.track(
            frame,
            conf=cfg.CONFIDENCE_THRESHOLD,
            iou=cfg.IOU_THRESHOLD,
            persist=True,
            verbose=False,
        )

        detections = []
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id not in cfg.VEHICLE_CLASS_IDS:
                continue

            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = xyxy.astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # ROI 过滤: 中心点不在 ROI 内 → 跳过
            if self.roi is not None and len(self.roi) >= 3:
                if cv2.pointPolygonTest(self.roi, (float(cx), float(cy)), False) < 0:
                    continue

            # track_id: Bot-SORT 分配的 ID, 如果没有则为 -1
            track_id = -1
            if boxes.id is not None:
                track_id = int(boxes.id[i].item())

            detections.append({
                "bbox":     (int(x1), int(y1), int(x2), int(y2)),
                "centroid": (int(cx), int(cy)),
                "conf":     conf,
                "cls_id":   cls_id,
                "cls_name": cfg.VEHICLE_CLASSES[cls_id],
                "track_id": track_id,
            })

        return detections
