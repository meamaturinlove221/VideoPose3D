# VideoPose3D 复现与项目说明

<p align="center">
  <img src="docs/figures/videopose3d_personal_improvement_route.svg" alt="VideoPose3D reproduction workflow" width="100%" />
</p>

<p align="center">
  <a href="README.md">English README</a> ·
  <a href="#这个仓库的定位">仓库定位</a> ·
  <a href="#整体流程">整体流程</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#贡献边界">贡献边界</a>
</p>

这个仓库是基于 **VideoPose3D** 的项目化整理版本。VideoPose3D 是单目 3D 人体姿态估计里很经典的一条路线：输入一段 2D 人体关键点序列，通过时间模型预测 3D 人体关节点坐标。

我保留原始项目的核心方法和代码边界，把这个 fork 整理成更适合复现、汇报、面试讲解和项目组合展示的形式。这里重点讲清楚任务是什么、数据怎么流动、模型解决什么问题、结果如何评估，以及这个仓库里哪些内容属于我自己的整理工作。

## 这个仓库的定位

VideoPose3D 的输入输出关系很清楚：

```text
连续帧 2D 关键点  ->  时间模型  ->  3D 人体关节点
```

这个仓库围绕这条路线做项目化整理。它适合用来：

- 复习经典的 2D-to-3D 人体姿态估计流程；
- 跑通 Human3.6M 风格的数据、训练、评估和可视化链路；
- 解释时间上下文为什么能帮助单目 3D 姿态恢复；
- 作为 3D 视觉项目组合里的基础人体姿态估计 baseline；
- 在讲项目时明确区分原始论文贡献和个人 fork 的整理工作。

## 项目亮点

| 方向 | 原项目保留内容 | 这个 fork 的整理重点 |
| --- | --- | --- |
| 方法 | VideoPose3D 原始时间卷积姿态提升模型 | 说明它在个人 3D 视觉项目链路中的 baseline 位置 |
| 流程 | 数据读取、2D 关键点、训练、评估、渲染 | 按工程复现顺序重新梳理 README |
| 展示 | 原始代码与许可边界 | 增加架构图和中英文项目说明 |
| 可信度 | 核心模型归属于原始 VideoPose3D | 明确个人工作集中在复现、整理和展示层面 |

## 整体流程

```text
单目视频 / 连续帧图像
        |
        v
2D 人体关键点检测结果或预处理好的 2D 关键点
        |
        v
VideoPose3D 时间卷积姿态提升模型
        |
        v
预测 3D 人体关节点坐标
        |
        v
指标评估 + 可视化检查
```

实际讲解时，可以拆成五步：

1. 准备视频序列或 benchmark 数据集；
2. 得到符合 VideoPose3D 格式的 2D 人体关键点；
3. 用时间模型把 2D 关键点轨迹提升到 3D；
4. 使用标准 3D pose 指标评估预测结果；
5. 渲染结果，观察成功样例和失败样例。

## 仓库结构

```text
common/          模型、相机几何、loss、数据集 loader、generator
run.py           训练、评估、渲染的主入口
data/            放置预处理后的 2D / 3D 数据文件
checkpoint/      放置训练得到的 checkpoint
docs/figures/    README 顶部使用的项目流程图
```

主入口是 `run.py`。它会读取命令行参数，加载数据集和 2D 检测结果，构建时间姿态模型，并根据参数进入训练、评估或渲染流程。

## 快速开始

克隆仓库：

```bash
git clone https://github.com/meamaturinlove221/VideoPose3D.git
cd VideoPose3D
```

创建 Python 环境，并安装原始项目需要的依赖。通常需要 PyTorch、NumPy、Matplotlib，以及渲染视频时用到的相关依赖。

按照原始 VideoPose3D 的数据命名方式，把数据放到 `data/` 目录。例如默认 Human3.6M 配置会查找：

```text
data/data_3d_h36m.npz
data/data_2d_h36m_cpn_ft_h36m_dbb.npz
```

使用默认 Human3.6M 风格配置训练：

```bash
python run.py -d h36m -k cpn_ft_h36m_dbb
```

评估 checkpoint：

```bash
python run.py -d h36m -k cpn_ft_h36m_dbb --evaluate checkpoint.bin
```

渲染一段姿态结果：

```bash
python run.py -d h36m -k cpn_ft_h36m_dbb --render \
  --viz-subject S9 \
  --viz-action Walking \
  --viz-camera 0 \
  --viz-output output.mp4
```

数据集和预训练权重没有直接打包在仓库里。Human3.6M 等数据需要按原数据集和原项目要求在本地准备。

## 为什么保留这个 baseline

在 3D 视觉项目组合里，VideoPose3D 的价值在于它足够经典，也足够清楚：

- 输入是 2D 人体关键点，不直接处理 RGB 图像；
- 模型核心是时间上下文建模，关注连续帧中的运动信息；
- 输出是 3D 骨架关节点，不是稠密人体表面；
- 评估指标明确，适合作为后续更复杂人体几何项目的对照基线。

这条路线可以自然连接到后续更复杂的方向，例如 SMPL / SMPL-X 人体先验、多视角人体重建、人体点云补全和 full-scene human reconstruction。

## 我在这个 fork 里做了什么

这个 fork 不把原始 VideoPose3D 模型包装成新的个人算法。我的整理重点是：

- 重写 README，让任务、流程和仓库用途更清楚；
- 增加项目流程图，方便报告和 GitHub 首页展示；
- 按数据准备、训练、评估、可视化的顺序梳理使用方式；
- 明确原始方法与个人整理工作的边界；
- 增加中文版说明，方便中文场景下复习和面试准备。

## 贡献边界

这个仓库应该这样描述：

- **原始方法**：VideoPose3D 的时间模型和 3D 姿态提升主流程。
- **个人 fork 工作**：复现路线整理、文档重写、流程说明、架构图和项目展示组织。
- **不应声称**：原始模型结构、论文贡献或原项目 benchmark 结果是个人原创。

这个边界很重要。它能让项目讲起来更可信，也能避免把 fork 的 baseline 说成自研模型。

## 面试或汇报时可以这样讲

> 我把 VideoPose3D 作为一个经典的单目 3D 人体姿态估计 baseline 来复现和整理。它的核心流程是从连续帧 2D 关键点恢复 3D 人体关节点。我在这个仓库里主要做的是复现路线梳理、流程说明、可视化入口和中英文文档整理，没有把原始模型包装成自己的新算法。

可以继续展开的问题：

- 为什么时间模型比单帧模型更适合 3D pose lifting；
- 2D 关键点检测质量会怎样影响 3D 预测；
- Human3.6M 风格指标能说明什么，又不能说明什么；
- 3D 骨架和稠密人体几何有什么区别；
- 这个 baseline 如何连接到后续 SMPL-X、人体点云和场景级人体重建项目。

## 架构图

README 顶部的流程图位于：

```text
docs/figures/videopose3d_personal_improvement_route.svg
```

## 致谢

本仓库基于原始 **VideoPose3D** 项目。核心方法和代码归属于原项目作者。这个 fork 的重点是把该 baseline 整理成更容易阅读、复现、汇报和讨论的项目形态。
