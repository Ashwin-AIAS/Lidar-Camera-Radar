## Overview

This repository contains a collection of scripts and utilities for experimenting with sensor data processing, object detection, and basic tracking algorithms. The focus is on handling datasets like KITTI, performing data conversions, testing object detection outputs, and demonstrating multi-sensor tracking concepts.

Each module is written to be self-contained but interoperable, making it easier to build, test, and extend components for autonomous system perception pipelines.

---

## File Descriptions

- **`baby_kalman.py`**  
  A minimal Kalman filter implementation for tracking object positions over time. Useful for understanding how state estimation works before applying it to complex multi-sensor setups.

- **`dataloaders.py`**  
  Contains data loading utilities for structured datasets like KITTI. Handles reading annotations, calibration files, and sensor data for quick prototyping.

- **`kitty_conversion.py`**  
  Scripts for converting KITTI-format data into custom or intermediate formats used in later experiments (e.g., transforming coordinates or reformatting bounding boxes).

- **`kitty_detect_classes.py`**  
  Defines or extracts object classes (car, pedestrian, cyclist, etc.) from KITTI annotations for training, testing, or evaluation.

- **`kitty_bounding_box_test.py`**  
  A testing and visualization tool for validating bounding box predictions against ground truth. Helpful for debugging detection algorithms.

- **`multisensor_fusion_demo.py`**  
  A demonstration script showing how multiple sensing modalities can be combined for better perception and tracking. Includes basic visualization and synchronization logic.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Ashwin-AIAS/Lidar-Camera-Fusion.git

