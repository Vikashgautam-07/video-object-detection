# Video Object Detection

In the contemporary era of rapid technological advancement, **video** object detection has become central to intelligent surveillance, real-time monitoring, and automated behavioral analysis. This repository implements a three-phase pipeline that transforms raw video streams into structured, actionable metadata using SSD-based detection, OpenCV, and COCO label maps.

## 1. Project Overview

This project is designed as a compact yet complete research pipeline for video object detection, bridging the gap between exploratory notebooks and near-production scripts. The work progressively evolves from basic frame-level detection to class-specific entity statistics suitable for scene understanding and analytics.

### Motivation

Traditional static image classification fails to address temporal consistency, frame-by-frame variability, and the latency demands of real-time video streams.

- Using a single-shot detector (SSD) for fast, real-time inference on video streams.
- Integrating OpenCV’s video pipeline with TensorFlow’s frozen inference graphs for efficient deployment.
- Progressively enriching outputs from raw detections to interpretable scene-level summaries (e.g., “3 cars and 2 person”).

### Objectives

- Implement and verify SSD-based object detection on video frames.  
- Design scripts that move from simple detection to total object counting and finally granular entity counting.  
- Provide a clear, reproducible template for video object detection research projects.

## 2. Core Technologies

The implementation balances high-speed inference with robust frame manipulation and metadata extraction.

- **SSD (Single Shot MultiBox Detector)**  
  - Single-stage detector that performs localization and classification in one pass, enabling near real-time video inference.

- **TensorFlow Frozen Inference Graph**  
  - `frozen_inference_graph.pb` contains the serialized network architecture and trained SSD weights for efficient loading via OpenCV’s DNN module.

- **OpenCV (cv2) & DNN Module**  
  - `cv2.VideoCapture` is used for frame acquisition, while `cv2.dnn.readNetFromTensorflow` loads and runs the frozen graph on each frame.

- **COCO Label Mapping**  
  - COCO `.names` files map numeric class IDs to human-readable labels, enabling classification across 80 object categories.

## 3. Project Phases

The project is structured to mirror a typical computer vision research lifecycle, from exploratory analysis to deployable scripts.

### Phase 1 – Simple Object Detection (Notebook)

- File: `Phase 1_ Simple object detection.ipynb`  
- Role: Experimental baseline to validate that the SSD model and frozen graph load correctly and produce reasonable detections. 
- Environment: Jupyter Notebook, supporting rapid iteration, visualization, and hyperparameter tuning on static frames. 

### Phase 2 – Total Object Count (Script)

- File: `Phase 2_Total Object count.py`  
- Role: Standalone Python script that extends detection to maintain a running tally of all objects detected per frame. 
- Goal: Provide a holistic measure of scene density and object activity in real time. 

### Phase 3 – Entity Count (Script)

- File: `Phase 3_Entity count.py`  
- Role: Final phase where the system transitions from raw counts to class-specific statistics using the COCO label map. 
- Output: Instead of “N objects detected,” the script reports granular statistics such as “3 cars and 2 persons,” turning visual signals into actionable metadata.


## 4. Installation & Setup

### Prerequisites

- Python 3.x  
- pip  
- A working webcam or video file as input  
- Basic familiarity with OpenCV and Python scripting

## 5. Research Perspective

Although this repository is lightweight, it is structured to reflect a typical computer vision research pipeline.

- **Exploratory stage**: Phase 1 Notebook for model sanity checks, hyperparameter exploration, and visualization.  
- **Prototyping stage**: Phase 2 script for real-time deployment and memory-aware execution.  
- **Analysis stage**: Phase 3 script for extracting semantic-level statistics that can feed higher-level tasks (e.g., traffic analysis, occupancy monitoring).

This makes the repository a useful starting point for more advanced topics such as tracking, temporal consistency, or integrating fairness and robustness metrics over time.

## 6. License

This project is licensed under the **Apache-2.0 License**.  
See the [`LICENSE`](LICENSE) file for details.
