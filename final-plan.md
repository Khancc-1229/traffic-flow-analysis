# CV 学习冲刺 — 最终计划

## 已完成
- Day 1-2: CNN + LeNet + PyTorch 训练框架
- Day 3: 迁移学习 freeze/finetune/scratch
- Day 4: OpenCV 全基础
- Day 5: 过拟合五方案
- Day 6: labelme 标注 + JSON→YOLO 转换
- Day 7+: YOLO v1-v8 原理 + 训练 + 推理 + 调参对比
- Day 8: 分割（YOLO-seg + SAM + U-Net）+ 关键点（YOLO-pose）
- Day 9: 实时摄像头检测
- 数据增强 20+ 方法全览

## 学习期（剩余 4 天）

### Day 10 — 目标跟踪 + 视频分析
- `model.track(persist=True)` — Bot-SORT 原理
- 越线检测 + 去重逻辑（理解原理即可）
- 视频抽帧分析 + 计数

### Day 11 — 图像增强
- 去噪: 高斯/中值/双边/非局部均值 对比
- 弱光增强: CLAHE + Gamma 校正
- 去雾: 暗通道先验 手写
- 去模糊: 概念 + DeblurGAN 了解
- **产出**: 增强对比图，每种方法一个脚本

### Day 12 — FastAPI + Docker
- FastAPI 路由/文件上传/多端点
- Dockerfile + docker-compose
- **产出**: 可 curl 的 CV API + 容器化

### Day 13 — ONNX + 模型压缩 + Gradio
- ONNX 导出/推理/加速对比
- 量化/剪枝/蒸馏 概念 + 实验
- Gradio 前端 + 全链路整合
- **产出**: 压缩对比表 + Web UI

## 项目期（5 天）

### 项目 1: 猫狗品种识别（1.5 天）
- Stanford Dogs + Oxford Pets 数据集
- 细粒度分类: 120 种狗 + 37 种猫狗
- Gradio UI: 拍照→识别品种→Top-5 结果
- 面试讲: 细粒度分类难点 + 迁移学习方案

### 项目 2: 图像增强工具箱（1 天）
- Gradio UI + 4 种增强 + 原图/结果对比
- 面试讲: 每种算法的原理和适用场景

### 项目 3: 未戴头盔人脸识别（2 天）
- YOLO 检测 + 跟踪 + 人脸识别 + SQLite
- 面试讲: 多模型联动 + 感知→识别→记录

### 项目 4: 交通流量分析（1 天）
- 检测 + 跟踪 + 计数 + 统计
- 手机拍 5 分钟马路视频

### 项目 5: CV API 平台（0.5 天）
- FastAPI + Docker + 多模型端点
- 与 Day 12-13 重合，主要整理文档

---

**总计: 4 天学习 + 5 天项目 = 9 天。6/24 开始，7/2 结束。**
