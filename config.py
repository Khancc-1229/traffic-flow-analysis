"""
交通流量分析 — 集中配置文件
交通流量分析 — 集中配置文件
"""

import json
import os

# ============================================================
# 模型与设备
# ============================================================
MODEL_NAME = "yolov8m.pt"              # 检测模型: yolov8n / yolov8s / yolov8m
DEVICE = "cuda"                         # 推理设备: "cuda" / "cpu"
CONFIDENCE_THRESHOLD = 0.4              # 检测置信度阈值
IOU_THRESHOLD = 0.5                     # NMS IOU 阈值

# ============================================================
# 车辆类别 — COCO 数据集中的车辆类
# ============================================================
# COCO class ids for vehicles
# 2=car, 3=motorcycle, 5=bus, 7=truck (COCO 2017)
VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

# 过滤后保留的类别 (只在车辆类别中)
VEHICLE_CLASS_IDS = set(VEHICLE_CLASSES.keys())

# 类别显示颜色 BGR (用于绘制框)
CLASS_COLORS = {
    "car":        (0, 255, 0),        # 绿色
    "motorcycle": (255, 255, 0),      # 青色
    "bus":        (255, 0, 0),        # 蓝色
    "truck":      (0, 0, 255),        # 红色
}

# ============================================================
# 视频处理
# ============================================================
VIDEO_PATH = "Traffic1.mp4"
OUTPUT_VIDEO_PATH = "output/result.mp4"
OUTPUT_DIR = "output"
CHARTS_DIR = "output/charts"
REPORT_PATH = "output/report.json"

PROCESS_WIDTH = 1920                   # 处理分辨率宽
PROCESS_HEIGHT = 1080                  # 处理分辨率高
SKIP_FRAMES = 1                        # 跳帧间隔, 1=不跳

# ============================================================
# 计数线 (两点的像素坐标, 在标定模式下通过鼠标点击设置)
# 格式: [(x1, y1), (x2, y2)]
# ============================================================
COUNTING_LINE = [(0, 0), (0, 0)]

# ============================================================
# 速度标定 — 基于路宽 y 轴比例插值
# ============================================================
# 原理: 路在任何距离宽度一样, 但画面里远处比近处窄
# 在远处和近处各标一次路宽 → 系统对任意 y 坐标插值 px/m
#
# 标定点: 4 个 (远处左, 远处右, 近处左, 近处右)
ROAD_WIDTH_POINTS = []           # [(far_left), (far_right), (near_left), (near_right)]

# 道路实际宽度 (米)
ROAD_REAL_WIDTH = 7.0            # 如 2 车道 ≈ 7m, 1 车道 ≈ 3.5m

# 检测 ROI: 只检测这个多边形内的车辆 (过滤对面车道)
# 顺时针 4 个点, 标定时用鼠标圈选
ROI_POINTS = []

# 速度平滑窗口 (帧数)
SPEED_SMOOTH_WINDOW = 6

# 速度显示颜色阈值 (km/h)
SPEED_COLOR_GREEN = 40                 # < 40 绿色
SPEED_COLOR_YELLOW = 80                # 40~80 黄色, >80 红色

# ============================================================
# 统计区间
# ============================================================
STATS_INTERVAL_SECONDS = 30            # 统计时间片 (秒)

# ============================================================
# 可视化选项
# ============================================================
SHOW_BOXES = True                      # 显示检测框
SHOW_LABELS = True                     # 显示类别标签
SHOW_CONFIDENCE = True                 # 显示置信度
SHOW_TRACK_ID = True                   # 显示跟踪 ID
SHOW_SPEED = True                      # 显示速度标注
SHOW_TRAILS = True                     # 显示轨迹线
SHOW_COUNTING_LINE = True              # 显示计数线
SHOW_DASHBOARD = True                  # 显示叠加面板
TRAIL_LENGTH = 20                      # 轨迹线长度 (帧数)
BOX_THICKNESS = 2                      # 检测框线宽
FONT_SCALE = 0.6                       # 字体大小

# ============================================================
# 标定文件路径
# ============================================================
CALIBRATION_FILE = "calibration.json"


def save_calibration():
    """保存标定参数到 JSON 文件"""
    data = {
        "counting_line": COUNTING_LINE,
        "road_width_points": ROAD_WIDTH_POINTS,
        "road_real_width": ROAD_REAL_WIDTH,
        "roi_points": ROI_POINTS,
    }
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[标定] 参数已保存到 {CALIBRATION_FILE}")


def load_calibration():
    """从 JSON 文件加载标定参数, 返回 (成功, 标定数据字典)"""
    if not os.path.exists(CALIBRATION_FILE):
        print("[标定] 未找到标定文件, 将进入交互标定模式")
        return False, {}
    with open(CALIBRATION_FILE, "r") as f:
        data = json.load(f)
    global COUNTING_LINE, ROAD_WIDTH_POINTS, ROAD_REAL_WIDTH, ROI_POINTS
    COUNTING_LINE = data["counting_line"]
    ROAD_WIDTH_POINTS = data["road_width_points"]
    ROAD_REAL_WIDTH = data["road_real_width"]
    ROI_POINTS = data.get("roi_points", [])
    print(f"[标定] 已从 {CALIBRATION_FILE} 加载标定参数 (路宽标定)")
    return True, data
