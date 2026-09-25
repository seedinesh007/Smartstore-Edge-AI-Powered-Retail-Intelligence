# SmartStore Edge AI — Powered Retail Intelligence

> An AI-powered retail monitoring system for **real-time queue management and shelf monitoring** using computer vision and edge AI.

## Overview

SmartStore Edge AI is a computer-vision-based retail intelligence prototype designed to help small and medium-sized stores monitor customer queues and shelf activity in real time.

The system uses a camera feed and an **YOLO-based object detection model** to analyze the store environment and generate actionable alerts without requiring continuous cloud processing.

### Core Idea

```text
 Camera
    ↓
 YOLO Object Detection
    ↓
Edge Processing
    ↓
Queue Monitoring + Shelf Monitoring
    ↓
Alert / Action
