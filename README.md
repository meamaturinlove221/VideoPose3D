# VideoPose3D Fork: Improvements and Workflow

<p align="center">
  <img src="docs/figures/videopose3d_personal_improvement_route.svg" alt="VideoPose3D fork improvements and workflow" width="100%" />
</p>

## Route Position

This repository is a project-oriented fork of **VideoPose3D**, a classic baseline for **monocular 3D human pose estimation**.

The upstream VideoPose3D project provides the core method: lifting 2D keypoint sequences into 3D human joint coordinates with a temporal model. This fork keeps that baseline role clear, while reorganizing the repository description around reproduction, workflow clarity, result interpretation, and project presentation.

The goal of this README is therefore different from the original upstream documentation. It does not repeat the full paper-level introduction. Instead, it explains what this fork adds on top of the upstream baseline and how the project is used as a reproducible 3D pose estimation route.

## What the Upstream VideoPose3D Baseline Provides

The original VideoPose3D codebase is valuable because it gives a strong and widely recognized 3D pose lifting pipeline:

- **Input**: 2D joint trajectories extracted from monocular video.
- **Model**: a temporal convolutional architecture that uses motion context across frames.
- **Output**: 3D human joint coordinates.
- **Evaluation**: standard 3D pose metrics on benchmark-style datasets.

This makes VideoPose3D a good foundation for a personal research repository: the method is well known, the task is clear, and the input-output structure is easy to explain in a technical interview or project report.

## What This Fork Adds

Compared with the original repository, this fork focuses on **projectization** rather than claiming a new pose-estimation architecture.

The main repository-specific contributions are:

### 1. Clearer project framing

The original repository is primarily a research-code release. This fork rewrites the project entry point so that the repository is easier to read as a complete project:

- what task is being solved,
- what the upstream method contributes,
- where the fork-specific work sits,
- how the pipeline should be understood from input to output.

This makes the repository easier to present and easier to reuse later.

### 2. Explicit experiment workflow

The fork emphasizes a practical experiment loop:

1. prepare video or frame-sequence input,
2. obtain or load 2D human keypoints,
3. run the temporal 2D-to-3D lifting pipeline,
4. evaluate predicted 3D joints with standard metrics,
5. visualize the predicted poses and inspect failure cases.

This workflow turns the repository from a code snapshot into a more repeatable engineering route.

### 3. Better documentation for contribution boundaries

A common issue with forked research repositories is that the upstream method and the fork-specific contribution can become mixed together. This README keeps that boundary explicit:

- the original VideoPose3D method remains the core baseline,
- this fork contributes documentation, workflow framing, and project-facing organization,
- any future model or training changes should be described as separate extensions rather than as part of the original method.

### 4. Presentation-ready architecture figure

The SVG figure at the top of this README summarizes the repository in a report-friendly way:

- upstream baseline,
- fork contribution,
- reproducible experiment loop,
- repository output and portfolio value.

This is meant to make the project easier to explain at a glance without overstating the technical ownership of the original method.

## Practical Pipeline

The repository can be understood as the following pipeline:

```text
Monocular video / frame sequence
        ↓
2D keypoint detection or precomputed 2D keypoints
        ↓
VideoPose3D temporal lifting model
        ↓
Predicted 3D human joint coordinates
        ↓
Metric evaluation and qualitative visualization
```

## Contribution Boundary

This fork should be described carefully:

- **Original method**: VideoPose3D temporal 3D pose lifting baseline.
- **Fork-specific contribution**: project documentation, clearer workflow framing, reproducible project route, and presentation-oriented organization.
- **Not claimed**: authorship of the original VideoPose3D architecture or paper contribution.

This distinction makes the repository more credible in a portfolio context and avoids overstating the work.

## Repository Role

In a project portfolio, this repository is best positioned as a **classical monocular 3D human pose estimation baseline and reproduction project**.

It demonstrates the ability to:

- understand a well-known computer vision baseline,
- trace the full 2D-to-3D pose estimation pipeline,
- organize a research codebase into a clearer project workflow,
- interpret both numerical and visual pose-estimation results,
- present the project in a way that separates upstream work from fork-level contribution.

## Figure

The README architecture figure is stored at:

```text
docs/figures/videopose3d_personal_improvement_route.svg
```

## Credits

This repository is based on the original **VideoPose3D** project. The upstream authors deserve credit for the core method and codebase. This fork focuses on project-level organization, documentation, and workflow presentation around that baseline.
