"""
统计分析 — 从收集的数据生成图表

图表列表:
  1. 车流量随时间变化折线图
  2. 车辆类型分布饼图
  3. 速度分布直方图
  4. 行驶方向分布柱状图
  5. 综合仪表盘 (4 合 1)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import config as cfg

# ============================================================
# 中文字体设置
# ============================================================

def _setup_chinese_font():
    """尝试设置中文字体"""
    # 尝试常见中文字体
    chinese_fonts = [
        "Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei",
        "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC",
        "STSong", "SimSun", "KaiTi",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in chinese_fonts:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name, "Arial"]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    # 如果没有中文字体, 使用英文备用
    return None

_CN_FONT = _setup_chinese_font()

# 如果没找到中文字体, 标签用英文
def _label(cn: str, en: str) -> str:
    return cn if _CN_FONT else en


# ============================================================
# 图表生成
# ============================================================

def generate_charts(collector, counter, speed_estimator):
    """
    生成所有图表。

    参数:
        collector:        DataCollector 实例
        counter:          LineCounter 实例
        speed_estimator:  SpeedEstimator 实例
    """
    os.makedirs(cfg.CHARTS_DIR, exist_ok=True)

    data = collector.export()

    # 图表 1: 车流量随时间变化
    _chart_flow_over_time(collector)

    # 图表 2: 车辆类型分布
    _chart_class_distribution(counter)

    # 图表 3: 速度分布
    _chart_speed_distribution(data["all_speeds"])

    # 图表 4: 方向分布
    _chart_direction(counter)

    # 图表 5: 综合仪表盘
    _chart_dashboard(collector, counter, data["all_speeds"])

    print(f"[图表] 5 张图表已生成到 {cfg.CHARTS_DIR}/")


def _chart_flow_over_time(collector):
    """图表 1: 车流量随时间变化折线图"""
    ts_data = collector.time_series
    if not ts_data:
        print("[图表] 无时间序列数据, 跳过流量图")
        return

    times = [d["timestamp"] / 60.0 for d in ts_data]  # 分钟
    totals = [d["total_count"] for d in ts_data]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, totals, color="#2196F3", linewidth=2, marker="o", markersize=3)
    ax.fill_between(times, 0, totals, alpha=0.15, color="#2196F3")

    ax.set_xlabel(_label("时间 (分钟)", "Time (minutes)"), fontsize=12)
    ax.set_ylabel(_label("累计车流量 (辆)", "Cumulative Vehicle Count"), fontsize=12)
    ax.set_title(_label("车流量随时间变化", "Traffic Flow Over Time"), fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 标注最大值
    if totals:
        max_idx = np.argmax(np.diff([0] + totals))  # 最大增量
        if max_idx < len(times):
            ax.annotate(
                f'Peak: +{totals[max_idx] - (totals[max_idx - 1] if max_idx > 0 else 0)}',
                xy=(times[max_idx], totals[max_idx]),
                xytext=(times[max_idx] + 0.5, totals[max_idx] + 5),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red",
            )

    plt.tight_layout()
    fig.savefig(os.path.join(cfg.CHARTS_DIR, "1_flow_over_time.png"), dpi=150)
    plt.close(fig)


def _chart_class_distribution(counter):
    """图表 2: 车辆类型分布饼图"""
    summary = counter.get_summary()
    by_class = summary.get("by_class", {})
    if not by_class:
        print("[图表] 无分类数据, 跳过类型分布图")
        return

    labels = []
    sizes = []
    colors_list = []
    for cls_name, counts in by_class.items():
        total = counts["forward"] + counts["backward"]
        if total > 0:
            labels.append(cls_name)
            sizes.append(total)
            color_bgr = cfg.CLASS_COLORS.get(cls_name, (0, 255, 0))
            # BGR → RGB hex
            colors_list.append(f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}")

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors_list, startangle=90,
        textprops={"fontsize": 12},
    )
    for at in autotexts:
        at.set_fontweight("bold")

    ax.set_title(_label("车辆类型分布", "Vehicle Type Distribution"),
                 fontsize=14, fontweight="bold")

    # 图例
    legend_labels = [f"{l}: {s}" for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title="Types", loc="lower right",
              fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(cfg.CHARTS_DIR, "2_class_distribution.png"), dpi=150)
    plt.close(fig)


def _chart_speed_distribution(all_speeds: list):
    """图表 3: 速度分布直方图"""
    speeds = [s for s in all_speeds if 2.0 < s < 200.0]
    if not speeds:
        print("[图表] 无速度数据, 跳过速度分布图")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.arange(0, max(speeds) + 11, 10)  # 每 10 km/h 一个 bin
    counts, bins, patches = ax.hist(speeds, bins=bins, edgecolor="white",
                                     color="#4CAF50", alpha=0.8)

    # 速度颜色带: 绿→黄→红
    for patch, bin_center in zip(patches, (bins[:-1] + bins[1:]) / 2):
        if bin_center < cfg.SPEED_COLOR_GREEN:
            patch.set_facecolor("#4CAF50")  # 绿
        elif bin_center < cfg.SPEED_COLOR_YELLOW:
            patch.set_facecolor("#FFC107")  # 黄
        else:
            patch.set_facecolor("#F44336")  # 红

    avg_speed = np.mean(speeds)
    ax.axvline(avg_speed, color="blue", linestyle="--", linewidth=2,
               label=f'Avg: {avg_speed:.1f} km/h')
    ax.axvline(np.median(speeds), color="purple", linestyle=":",
               linewidth=2, label=f'Median: {np.median(speeds):.1f} km/h')

    ax.set_xlabel(_label("速度 (km/h)", "Speed (km/h)"), fontsize=12)
    ax.set_ylabel(_label("频次", "Frequency"), fontsize=12)
    ax.set_title(_label("车辆速度分布", "Vehicle Speed Distribution"),
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(cfg.CHARTS_DIR, "3_speed_distribution.png"), dpi=150)
    plt.close(fig)


def _chart_direction(counter):
    """图表 4: 行驶方向分布"""
    summary = counter.get_summary()
    forward = summary.get("forward", 0)
    backward = summary.get("backward", 0)

    if forward == 0 and backward == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        [_label("正向 (↑)", "Forward"), _label("反向 (↓)", "Backward")],
        [forward, backward],
        color=["#2196F3", "#FF9800"],
        width=0.5,
        edgecolor="white",
        linewidth=1.5,
    )

    for bar, val in zip(bars, [forward, backward]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", fontsize=14, fontweight="bold")

    ax.set_ylabel(_label("车辆数", "Vehicle Count"), fontsize=12)
    ax.set_title(_label("行驶方向分布", "Traffic Direction Distribution"),
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(forward, backward) * 1.2 + 5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(cfg.CHARTS_DIR, "4_direction.png"), dpi=150)
    plt.close(fig)


def _chart_dashboard(collector, counter, all_speeds):
    """图表 5: 综合仪表盘 — 4 合 1 大图"""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(_label("交通流量分析仪表盘", "Traffic Flow Analysis Dashboard"),
                 fontsize=18, fontweight="bold", y=0.98)

    # ---- 左上: 流量曲线 ----
    ax1 = fig.add_subplot(2, 2, 1)
    ts_data = collector.time_series
    if ts_data:
        times = [d["timestamp"] / 60.0 for d in ts_data]
        totals = [d["total_count"] for d in ts_data]
        ax1.plot(times, totals, color="#2196F3", linewidth=2)
        ax1.fill_between(times, 0, totals, alpha=0.1, color="#2196F3")
    ax1.set_title(_label("Cumulative Vehicle Count", "Cumulative Vehicle Count"), fontsize=12)
    ax1.set_xlabel(_label("分钟", "Minutes"))
    ax1.set_ylabel(_label("辆", "Count"))
    ax1.grid(True, alpha=0.3)

    # ---- 右上: 类型分布 ----
    ax2 = fig.add_subplot(2, 2, 2)
    summary = counter.get_summary()
    by_class = summary.get("by_class", {})
    if by_class:
        labels = []
        sizes = []
        colors_list = []
        for cls_name, counts in by_class.items():
            total = counts["forward"] + counts["backward"]
            if total > 0:
                labels.append(cls_name)
                sizes.append(total)
                c = cfg.CLASS_COLORS.get(cls_name, (0, 255, 0))
                colors_list.append(f"#{c[2]:02x}{c[1]:02x}{c[0]:02x}")
        if labels:
            ax2.pie(sizes, labels=labels, autopct="%1.1f%%",
                    colors=colors_list, startangle=90)
    ax2.set_title(_label("Vehicle Type Distribution", "Vehicle Type Distribution"), fontsize=12)

    # ---- 左下: 速度分布 ----
    ax3 = fig.add_subplot(2, 2, 3)
    speeds = [s for s in all_speeds if 2.0 < s < 200.0]
    if speeds:
        bins = np.arange(0, max(speeds) + 11, 10)
        ax3.hist(speeds, bins=bins, edgecolor="white", color="#4CAF50", alpha=0.8)
        ax3.axvline(np.mean(speeds), color="blue", linestyle="--", linewidth=2)
    ax3.set_title(_label("Speed Distribution (km/h)", "Speed Distribution (km/h)"), fontsize=12)
    ax3.set_xlabel("km/h")
    ax3.set_ylabel(_label("频次", "Frequency"))
    ax3.grid(axis="y", alpha=0.3)

    # ---- 右下: 统计摘要 (文本) ----
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    minutes = collector.frame_count / 12.0 / 60.0 if collector.frame_count > 0 else 1
    flow_rate = summary["total"] / minutes if minutes > 0 else 0
    avg_speed = np.mean(speeds) if speeds else 0

    lines = [
        f"  Total Vehicles:     {summary['total']}",
        f"  Forward (Up):       {summary['forward']}",
        f"  Backward (Down):    {summary['backward']}",
        f"  Flow Rate:          {flow_rate:.1f} veh/min",
        f"  Avg Speed:          {avg_speed:.1f} km/h",
        f"  Max Speed:          {max(speeds):.1f} km/h" if speeds else "  Max Speed:          N/A",
        f"  Median Speed:       {np.median(speeds):.1f} km/h" if speeds else "  Median Speed:       N/A",
        f"  Frames Processed:   {collector.frame_count}",
    ]

    for i, line in enumerate(lines):
        ax4.text(0.05, 0.9 - i * 0.1, line, fontsize=13, fontfamily="monospace",
                 transform=ax4.transAxes)

    ax4.set_title(_label("Summary Statistics", "Summary Statistics"), fontsize=12)

    plt.tight_layout()
    fig.savefig(os.path.join(cfg.CHARTS_DIR, "5_dashboard.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
