# Radar-Guided Vision Transformer with Cross-Attention Fusion for Object Detection and Depth Estimation

Author: Padmanabh Butala

Overview
This repository hosts the source code for Radar-Guided Vision Transformer with Cross-Attention Fusion for object detection and depth estimation in autonomous vehicles. The approach combines radar and camera inputs through a Transformer-based fusion framework, achieving robust detection under adverse weather conditions such as fog and rain.
The design is informed by a 3-month Capstone timeline, covering literature review, baseline CNN fusion, and advanced Transformer-based radar-vision attention.


Project Figure
Below is a conceptual diagram of our multi-modal fusion framework. The radar input is processed into feature maps, which guide the Vision Transformer’s attention layers for object detection and depth estimation:
         ┌────────────────────┐        ┌────────────────────┐
         │   Radar Encoder    │        │  Camera Encoder     │
         └────────────────────┘        └────────────────────┘
                    │                          │
                    ▼                          ▼
            Radar Feature Maps          Image Feature Maps
                    │                          │
                    \________     _____________/
                             \   /
                   Cross-Attention Fusion
                             /   \
                    ┌───────────────┐
                    │ Multi-Task     │
                    │  Transformer   │
                    └───────────────┘
                             │
                             ▼
             Object Bounding Boxes & Depth Maps


