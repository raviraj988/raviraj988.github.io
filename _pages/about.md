---
permalink: /
title: " 👋 About Me"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

I’m Ravi Raj Kumar, a graduate student in Computer Science at **Case Western Reserve University**, specializing in AI and Machine Learning. 

I have taken graduate-level courses in **Machine Learning**, **Natural Language Processing**, **Probabilistic Graphical Models**, **Computer Vision**, **High-Performance Systems for AI**, and **Reinforcement Learning**, which have built a strong foundation for both my academic research and applied AI work.

I have four years of professional experience from **Tata Consultancy Services**, where I worked on designing and deployment of end-to-end machine learning systems in the banking and finance sector.

My work included building and fine-tuning scalable varous NLP pipelines using transformer-based models such as BERT, RoBERTa  as well as implementing production-grade** MLOps workflows** using MLflow, Docker, Kubernetes, and Jenkins. I’ve handled massive, high-velocity data from sources like Hadoop, Snowflake, and MongoDB, integrating continuous model validation, drift detection, and real-time monitoring with Prometheus and Grafana.


---
## AI & Machine Learning
### 🧪 Research Interests  
My research lies at the intersection of **generative modeling**, **multimodal vision-language learning**, and **reinforcement learning for structured reasoning**. I have explored methods to improve sample-efficient PPO-based optimization for multi-agent setups, and designing modular generative pipelines that operate across modalities. My work also extends to semi-supervised learning with deep generative models and efficient 3D perception using 2D vision cues—bridging foundational AI research with practical deployment challenges.

#### 📚 Research Highlights

- **Optimizing RAG with Multi-Agent Reinforcement Learning**  
  Designed a Multi-Agent Reinforcement Learning framework for RAG, modeling query design, document retrieval, and answer generation as cooperative agents jointly optimized via PPO under a unified F1-based reward signal.
  
- **Multimodal Transformer for Image-Conditioned Generation**  
  Designed and implemented a multimodal transformer integrating a SigLIP-style Vision Transformer with a Gemma-based causal decoder for image-grounded generation tasks such as captioning and visual question answering.

- **Latent Diffusion for Text-to-Image Synthesis**  
  Reproduced Stable Diffusion v1.5 with classifier-free guidance, achieving FID: 12.4 and CLIPScore: 0.32; enabled inpainting, image-to-image, and captioning tasks.

- **Semi-Supervised Learning with Deep Generative Models**  
  Extended the M2 architecture with entropy regularization and mutual information maximization, yielding a +4% accuracy gain on CIFAR-10 benchmarks.
  
- **🛰️ 3D Point Cloud Segmentation via 2D Voting**  
  Developed a novel 3D point cloud segmentation framework leveraging state-of-the-art 2D image segmentation models (OneFormer) and a voting-based approach to project 2D semantic and panoptic labels onto 3D point clouds, achieving real-time segmentation with reduced computational overhead.

---

### 💼 Professional Experience

**Machine Learning Engineer**  
*Tata Consultancy Services* (2019–2023)  
- Built robust and scalable end-to-end ML pipelines for a Bank Member Complaint Distribution System on cloud as-well-as on-prem
with components like data ingestion, data validation, feature engineering, model training, prediction, and monitoring.
- Leveraged advanced NLP tokenizers, such as BytePair Encoding (BPE) and SentencePiece for tokenization, trained and
finetuned several transformer-based models like BERT, RoBERTa, and Longformer on tasks such as Complaint Categorization
and Prioritization, Named Entity Recognition (NER) for automated information extraction, and Complaint Severity Prediction.
- Built CI/CD workflows with Jenkins, MLflow, Docker, and Kubernetes to support reproducible ML lifecycle and scalable model deployment across hybrid environments.  
- Designed real-time monitoring and drift detection dashboards using Prometheus and Grafana.  

**Data Engineer**  
- Architected ingestion and transformation pipelines for Hadoop, Snowflake, and MongoDB sources.  
- Delivered production-grade data pipelines using DBT, Hive, and shell scripting with automated validation and security controls.

 --- 
## 🤖 Robotics  
I work on **real-time autonomous systems** that integrate perception, planning, and control using **ROS2**, **SLAM**, and visual-inertial pipelines. My robotics projects span multi-agent navigation, racing systems, and robotic arms—blending classical robotics with modern deep learning for robust, adaptive, and deployable solutions.

- **🤖 Multi-Agent SLAM for Quadruped Robots**  
  Built distributed SLAM pipelines for Unitree GO1/GO2 using ZED SDK and RTABMap with LiDAR, improving mapping via voxel filtering and cloud clustering.

- **🧠 Swarm-Based Exploration and Semantic Mapping**  
  Developed scalable ROS2-based swarm navigation using stereo vision and instance segmentation with a custom semantic task allocator and chatbot interface.

- **🛰️ 3D Point Cloud Segmentation via 2D Voting**  
  Achieved 96.5% segmentation accuracy by projecting OneFormer 2D masks onto 3D LiDAR scans, enabling real-time, resource-efficient 3D understanding.

- **🏁 Autonomous Racing Buggy (ROS2 + YOLO)**  
  Integrated LIDAR and camera with INT8-quantized YOLOv5s for real-time object detection; achieved 1:42 track time with Ackermann-steering controller.


---

## 🔧 Backend & API Development
- Designed and deployed RESTful APIs using **FastAPI** and Python for modular AI services
- Built microservice architectures leveraging **Docker**, **Kubernetes**, and **NGINX** for scalable deployments
- Implemented **asynchronous processing** and **background task queues** using `asyncio` and `Celery` for real-time API responsiveness
- Used **CI/CD pipelines** with Jenkins and GitHub Actions to automate testing, linting, and container deployments
- Ensured **secure secret management** and environment isolation via Docker secrets, `.env` configs, and role-based access
- Delivered interactive model interfaces using **Streamlit**, **Gradio**, and **AJAX/REST** frontends for live ML demos
- Followed infrastructure-as-code principles using **Terraform** and **Docker Compose** for reproducible environments


---


