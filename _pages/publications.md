---
title: "🧠 AI & Machine Learning Projects"
layout: page
permalink: /ai-ml/
---

## 🚀 Overview

Welcome to my AI/ML portfolio. This section showcases my work in machine learning, generative AI, large language models, and MLOps. My focus lies in building scalable systems that combine state-of-the-art modeling with production-grade deployment. Projects here span across **Reinforcement Learning**, **Diffusion-based Generation**, **Multimodal Transformers**, and **LLM Tooling**, developed through both research and industry settings.

---

## 🔁 Optimizing RAG with Multi-Agent Reinforcement Learning  
[🔗 GitHub Repo](https://github.com/raviraj988)  
![RAG + PPO](../images/projects/rag-ppo.png)

- Designed a multi-agent PPO-based framework where query rewriting, document retrieval, and generation are jointly optimized using shared rewards.
- Used LoRA adapters for warm-start fine-tuning on SQuAD and built a reproducible pipeline with TRL’s PPOTrainer, improving factual consistency and generation relevance.

---

## 🖼️ Multimodal Transformer for Image-Conditioned Generation  
[🔗 GitHub Repo](https://github.com/yourusername/siglip-gemma)  
![Multimodal Transformer](../images/projects/siglip-gemma.png)

- Built a transformer combining SigLIP ViT encoder with a Gemma-style decoder to perform captioning and VQA.
- Engineered KV caching and autoregressive decoding for performance and BLEU-4 improvement; used temperature-scaled top-p sampling for diversity.

---

## 🎨 Latent Diffusion Model for Text-to-Image Synthesis  
[🔗 GitHub Repo](https://github.com/hkproj/pytorch-stable-diffusion)  
![Stable Diffusion](../images/projects/latent-diffusion.png)

- Reproduced Stable Diffusion from scratch with PyTorch, integrating CLIP for conditioning, VAE for latent encoding, and U-Net for denoising.
- Achieved FID: 12.4 and CLIPScore: 0.32 on COCO; optimized 12-layer Transformer with 4000+ tokens/sec throughput on A100.

---

## 📊 Semi-Supervised Learning with Deep Generative Models  
[🔗 GitHub Repo](https://github.com/raviraj988/Enhancing_SSL_Using_Deep_Gen_Models)  
![Semi-supervised Learning](../images/projects/ssl-dgm.png)

- Implemented Kingma et al.’s M2 architecture on MNIST and CIFAR-10; extended ELBO with entropy regularization and mutual info gain.
- Achieved +4% accuracy gain and 15% entropy reduction in predictions, boosting performance under low-label conditions.

---

## 🛰️ 3D Point Cloud Segmentation via 2D Voting  
[🔗 GitHub Repo](https://github.com/raviraj988/3D-POINT-CLOUD-SEGMENTATION-USING-2D-IMAGE-SEGMENTATION)  
![Point Cloud](../images/projects/pointcloud.png)

- Projected segmentation masks from 2D images (OneFormer outputs) onto 3D point clouds captured via LiDAR and iPhone depth.
- Achieved 96.5% accuracy—comparable to PointFormer—with dramatically lower resource usage.

---

## 🩺 Multi-Agent Medical Appointment System  
[🔗 GitHub Repo](https://github.com/raviraj988/Multi_Agent_System_For_Doctor-s_Appointment)  
![Medical Appointment AI](../images/projects/doctor-agent.png)

- Created a LangChain + LangGraph-based AI assistant with supervisor-agent structure to handle doctor queries and booking.
- Backend with FastAPI and frontend with Streamlit reduced appointment latency by ~60% across 4K simulated entries.

---

## 💬 RAG-Powered Customer Support Agent  
[🔗 GitHub Repo](https://github.com/raviraj988/RAG_Customer_support_System)  
![RAG Chatbot](../images/projects/rag-chatbot.png)

- Built an end-to-end chatbot using LangChain, AstraDB vector search, and Google Gemini embeddings.
- Supports semantic product recommendations and real-time chat UI with AJAX and FastAPI backend.

---

🧠 More projects coming soon...

Let me know if you'd like these added to your site automatically or converted into individual portfolio cards!
