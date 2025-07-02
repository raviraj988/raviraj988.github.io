---
title: "🤖 Robotics"
layout: page
permalink: /robotics/
---

## 🤖 Overview

This section features my robotics work spanning multi-agent SLAM, autonomous navigation, perception, and robotic manipulation. I focus on blending classical algorithms with modern AI to build robust, scalable, and real-time robotic systems for exploration, mapping, and interaction. Projects are built using **ROS2**, **RTABMap**, **ZED SDK**, **YOLO**, and simulation environments like **Isaac Gym** and **Gazebo**.

---

## 🧭 Multi-Agent SLAM for Quadruped Robot Fleet  
[🔗 GitHub Repo](https://github.com/raviraj988)  
![SLAM for Quadruped Robots](../images/robotics/quad-slam.png)

- Developed SLAM pipelines for Unitree GO1 (ZED+IMU) and GO2 (RTABMap+LiDAR), enabling real-time localization and mapping.
- Applied voxel filtering, deskewing, and point cloud clustering to improve grid accuracy and robustness.

---

## 🐝 Multi-Robot Swarm Exploration and Mapping  
[🔗 GitHub Repo](https://github.com/raviraj988)  
![Swarm Exploration](../images/robotics/swarm-map.png)

- Designed a ROS2-based autonomous swarm for dynamic exploration using stereo depth and instance segmentation.
- Built a semantic database task manager and chatbot interface for distributed swarm control and monitoring.

---

## 🛰️ 3D Point Cloud Segmentation via 2D Image Segmentation  
[🔗 GitHub Repo](https://github.com/raviraj988/3D-POINT-CLOUD-SEGMENTATION-USING-2D-IMAGE-SEGMENTATION)  
![3D Segmentation](../images/robotics/pointcloud.png)

- Fused RGB, depth, and LiDAR data with OneFormer 2D masks to segment 3D point clouds using a voting strategy.
- Achieved 96.5% segmentation accuracy with significant gains in compute efficiency vs. PointFormer.

---

## 🏎️ Autonomous Racing Buggy (ROS2 + YOLOv5)  
[🔗 GitHub Repo](https://github.com/raviraj988)  
![Autonomous Buggy](../images/robotics/buggy-race.png)

- Built a ROS2-based racing buggy using LIDAR and camera fusion for lane tracking and object avoidance.
- Used INT8-quantized YOLOv5s for 7Hz real-time inference; achieved a lap time of 1:42 with Ackermann control.

---

## 🦾 Precision Motion Planning for Franka Emika Arm  
[🔗 GitHub Repo](https://github.com/raviraj988)  
![Franka Arm Planning](../images/robotics/franka-golf.png)

- Implemented precise trajectory planning using MoveIt and vision-pipeline synchronization to strike golf balls with the Franka arm.
- Designed a ROS2 node for real-time ball and target localization, coordinating motion execution and object detection.

---

🧠 More robotic platforms, RL-based planning, and SLAM research in progress...

