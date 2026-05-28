# VideoPose3D Reproduction & Project Notes

<p align="center">
  <img src="docs/figures/videopose3d_personal_improvement_route.svg" alt="VideoPose3D reproduction workflow" width="100%" />
</p>

<p align="center">
  <a href="README_CN.md">中文说明</a> ·
  <a href="#what-this-repository-is-for">Project role</a> ·
  <a href="#pipeline">Pipeline</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#contribution-boundary">Contribution boundary</a>
</p>

This repository is a project-oriented fork of **VideoPose3D**, the well-known temporal baseline for monocular 3D human pose estimation.

The original project answers a clean research question: given a sequence of 2D human keypoints, how far can a temporal model go in recovering 3D human joint positions? This fork keeps that baseline intact and makes the repository easier to use as a readable reproduction project: the task, data flow, evaluation loop, visualization path, and contribution boundary are laid out in one place.

## What this repository is for

VideoPose3D is useful because the input-output contract is simple and still relevant:

```text
2D keypoints over time  ->  temporal pose lifting  ->  3D human joints
```

This fork is organized around that contract. It is meant to be read as a compact 3D pose estimation project rather than only as a copied research-code release.

It is suitable for:

- reviewing a classical 2D-to-3D pose lifting pipeline;
- reproducing the Human3.6M-style evaluation route;
- explaining how temporal context helps monocular 3D pose estimation;
- preparing a clean project narrative for reports, interviews, and portfolio review;
- keeping upstream method ownership and fork-level work clearly separated.

## Highlights

| Area | What is kept | What is added in this fork |
| --- | --- | --- |
| Method | Original VideoPose3D temporal lifting baseline | Clearer explanation of where the baseline fits in a personal project stack |
| Workflow | Dataset loading, 2D keypoints, training, evaluation, rendering | A README structure that follows the actual engineering loop |
| Presentation | Original code and license boundary | Architecture figure and bilingual project documentation |
| Credibility | Upstream authors remain credited for the core model | Fork-specific contribution is described without overstating model authorship |

## Pipeline

```text
Monocular video / frame sequence
        |
        v
2D keypoint detection or precomputed 2D detections
        |
        v
Temporal convolutional pose lifting model
        |
        v
3D human joint coordinates
        |
        v
Metric evaluation + qualitative pose visualization
```

In practice, this means the project can be explained through five steps:

1. prepare a video sequence or a benchmark dataset split;
2. obtain 2D keypoints in the format expected by VideoPose3D;
3. run the temporal model to lift 2D trajectories into 3D joint coordinates;
4. evaluate the predicted 3D poses with standard pose metrics;
5. render examples and inspect where the model succeeds or fails.

## Repository map

```text
common/          model, camera geometry, losses, dataset loaders, generators
run.py           main training / evaluation / rendering entry point
data/            expected location for prepared 2D and 3D dataset files
checkpoint/      expected location for trained checkpoints
docs/figures/    project-level workflow figure used by this README
```

The main entry point is `run.py`. It loads the dataset, reads 2D detections, builds the temporal pose model, and then follows the selected mode: training, evaluation, or rendering.

## Quick start

Clone the repository:

```bash
git clone https://github.com/meamaturinlove221/VideoPose3D.git
cd VideoPose3D
```

Create a Python environment and install the packages required by the original project. A typical environment needs PyTorch, NumPy, Matplotlib, and the video / visualization dependencies used for rendering.

Prepare the dataset files under `data/`. The default argument setup expects files following the original VideoPose3D naming convention, for example:

```text
data/data_3d_h36m.npz
data/data_2d_h36m_cpn_ft_h36m_dbb.npz
```

Run training with the default Human3.6M-style setting:

```bash
python run.py -d h36m -k cpn_ft_h36m_dbb
```

Evaluate a checkpoint:

```bash
python run.py -d h36m -k cpn_ft_h36m_dbb --evaluate checkpoint.bin
```

Render a pose sequence:

```bash
python run.py -d h36m -k cpn_ft_h36m_dbb --render \
  --viz-subject S9 \
  --viz-action Walking \
  --viz-camera 0 \
  --viz-output output.mp4
```

The exact dataset files and checkpoints are not bundled here. Keep benchmark data and pretrained weights in local storage according to the original project and dataset license requirements.

## Why keep this baseline

For a 3D vision portfolio, VideoPose3D is a good classical baseline because it sits between two larger topics:

- **human pose estimation**: it uses 2D body keypoints as the observation;
- **3D geometry from video**: it recovers a temporally coherent 3D joint sequence from monocular input.

That makes it a useful reference point before moving to heavier 3D human reconstruction projects such as dense point clouds, SMPL / SMPL-X priors, multi-view geometry, or scene-aware human reconstruction.

## What I changed

This fork does not claim a new pose-estimation architecture. The changes are project-facing:

- rewrote the repository entry point so the task and route are easier to understand;
- added a report-style workflow figure for the 2D-to-3D lifting process;
- documented the engineering loop from data preparation to rendering;
- clarified how to describe the repository honestly in a portfolio context;
- added a Chinese README for local review and interview preparation.

## Contribution boundary

This boundary is intentional:

- **Original method**: VideoPose3D temporal model and core 3D pose lifting pipeline.
- **This fork**: reproduction-oriented organization, project documentation, workflow explanation, and presentation assets.
- **Not claimed**: authorship of the original architecture, paper contribution, or benchmark results that belong to the upstream project.

Keeping this separation makes the repository easier to defend technically. It also avoids the common problem of mixing a forked baseline with personal contribution.

## Notes for interviews and reports

A concise way to describe this repository:

> I used VideoPose3D as a classical monocular 3D pose estimation baseline. The project helped me trace the full path from 2D keypoint sequences to temporal 3D joint prediction, evaluation, and visualization. My work on this fork is mainly reproduction, workflow organization, documentation, and project presentation, rather than claiming the original model as my own architecture.

Good follow-up discussion points:

- why a temporal model helps compared with single-frame lifting;
- how 2D keypoint quality affects 3D prediction;
- what Human3.6M-style evaluation measures and what it misses;
- why a sparse 3D skeleton is different from dense human geometry;
- how this baseline connects to later SMPL-X, point cloud, and full-scene human reconstruction work.

## Figure

The workflow figure used at the top of this README is stored at:

```text
docs/figures/videopose3d_personal_improvement_route.svg
```

## Credits

This repository is based on the original **VideoPose3D** project. The upstream authors deserve credit for the core method and codebase. This fork focuses on making the baseline easier to read, reproduce, present, and discuss as part of a broader 3D human understanding project stack.
