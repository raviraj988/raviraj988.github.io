---
title: "🧠 AI & Machine Learning"
layout: page
permalink: /ai-ml/
---

## 🚀 Overview

Welcome to my AI/ML portfolio. This section showcases my work in machine learning, generative AI, large language models, and MLOps. My focus lies in building scalable systems that combine state-of-the-art modeling with production-grade deployment. Projects here span across **Reinforcement Learning**, **Diffusion-based Generation**, **Multimodal Transformers**, and **LLM Tooling**, developed through both research and industry settings.

---

## 🔁 Optimizing RAG with Multi-Agent Reinforcement Learning  
[🔗 GitHub Repo](https://github.com/raviraj988)  
![RAG + PPO](../images/projects/rag-ppo.png)
<span style="font-size:12px;">
- Designed a Multi-Agent Reinforcement Learning framework for RAG, modeling query design, document retrieval, and answer
generation as cooperative agents jointly optimized via PPO under a unified F1-based reward signal.
- Implemented FAISS search with sentence-transformers/all-MiniLM-L6-v2 embeddings to enable efficient top-K passage retrieval
over a custom SQuAD corpus, all within a reproducible Conda environment.
- Implemented Warm-start for each agent by employing PEFT’s LoRA by freezing 96.65% of weights, to fine-tuning LoRA adapters
on 5,000 SQuAD QA pairs before any RL, which improved sample efficiency and stabilized the subsequent PPO loop.
- Implemented a PPO loop using TRL’s PPOTrainer that fine-tunes only the LoRA adapters and value head by iterating query
rewrite, retrieve, generate with a unified reward signal, significantly improving QA performance over SFT.- Designed a multi-agent PPO-based framework where query rewriting, document retrieval, and generation are jointly optimized using shared rewards.
- Used LoRA adapters for warm-start fine-tuning on SQuAD and built a reproducible pipeline with TRL’s PPOTrainer, improving factual consistency and generation relevance.
</span>
---

## 🖼️ Multimodal Transformer for Image-Conditioned Generation  
[🔗 GitHub Repo](https://github.com/yourusername/siglip-gemma)  
![Multimodal Transformer](../images/projects/siglip-gemma.png)
<span style="font-size:12px;">
- Designed and implemented a multimodal transformer integrating a SigLIP-style Vision Transformer with a Gemma-based causal
decoder for image-grounded generation tasks such as captioning and visual question answering.
- Engineered autoregressive decoding with KV caching and Rotary Positional Embeddings, reducing inference latency by 40% per
token on long sequences.
- Built a PaLI-inspired tokenizer pipeline with image-token prefixing and robust image normalization, improving input alignment and
reducing token mismatch errors by 23%.
- Enabled diverse generation using temperature-scaled top-p sampling; achieved a +3.7 BLEU-4 score improvement over greedy
decoding on benchmark prompts.
</span>

---

## 🎨 Latent Diffusion Model for Text-to-Image Synthesis  
[🔗 GitHub Repo](https://github.com/hkproj/pytorch-stable-diffusion)  
![Stable Diffusion](../images/projects/latent-diffusion.png)
<span style="font-size:12px;"> 
- Developed a complete multimodal pipeline integrating CLIP-based language encoder, variational autoencoder (VAE), and U-Net-
based denoising diffusion model for high-fidelity text-to-image generation.
- Achieved 4× faster inference over pixel-space models by operating in latent space (64×64 vs. 512×512), reducing memory and
compute by over 85%.
- Implemented Classifier-Free Guidance to enhance prompt alignment, improving semantic accuracy of generated images by 8%
(measured via CLIPScore).
- Enabled multimodal capabilities: text-to-image, image-to-image translation, and inpainting, through unified conditional
diffusion framework
</span>
- Reproduced Stable Diffusion from scratch with PyTorch, integrating CLIP for conditioning, VAE for latent encoding, and U-Net for denoising.
- Achieved FID: 12.4 and CLIPScore: 0.32 on COCO; optimized 12-layer Transformer with 4000+ tokens/sec throughput on A100.

---

## 📊 Semi-Supervised Learning with Deep Generative Models  
[🔗 GitHub Repo](https://github.com/raviraj988/Enhancing_SSL_Using_Deep_Gen_Models)  
![Semi-supervised Learning](../images/projects/ssl-dgm.png)
<span style="font-size:12px;">
- Conducted an in-depth survey of semi-supervised learning with deep generative models and implemented Kingma et al.’s M2
approach on MNIST and CIFAR-10. Matched paper results: 94.8% test accuracy on MNIST (1k labels) and 63.1% on CIFAR-10
(4k labels), establishing a reliable baseline.
- Conducted a rigorous mathematical and experimental analysis of the Evidence Lower Bound (ELBO) within a variational inference
framework, identifying and resolving three critical issues in the paper: entropy penalization for sharper decision boundaries; mutual
information maximization to strengthen input–label coupling; and smoothed-label integration into the classification loss.
- Delivered a 4% accuracy improvement over the original paper’s baseline on MNIST and a 2.5% gain on CIFAR-10 and 15%
reduction in classifier entropy.
</span>

---

## 🛰️ 3D Point Cloud Segmentation via 2D Voting  
[🔗 GitHub Repo](https://github.com/raviraj988/3D-POINT-CLOUD-SEGMENTATION-USING-2D-IMAGE-SEGMENTATION)  
![Point Cloud](../images/projects/pointcloud.png)
<span style="font-size:12px;">
- Developed a novel 3D point cloud segmentation framework leveraging state-of-the-art 2D image segmentation models (OneFormer)
and a voting-based approach to project 2D semantic and panoptic labels onto 3D point clouds, achieving real-time segmentation
with reduced computational overhead.
- Utilized RGB images, depth maps, and LiDAR data captured with the iPhone 13 Pro, integrating segmentation masks generated by
OneFormer with a voting mechanism to accurately transfer semantic labels to 3D point clouds.
- Achieved segmentation accuracy of 96.5%, matching PointFormer, while significantly reducing computational overhead and memory
usage demonstrating the model’s scalability and efficiency
</span>

---

## 🩺 Multi-Agent Medical Appointment System  
[🔗 GitHub Repo](https://github.com/raviraj988/Multi_Agent_System_For_Doctor-s_Appointment)  
![Medical Appointment AI](../images/projects/doctor-agent.png)
<span style="font-size:12px;">
- Built a modular multi-agent AI system using LangChain and LangGraph, implementing a supervisor-agent architecture to
route user queries to specialized agents for doctor information and appointment management.
- Engineered a full-stack solution with FastAPI for scalable backend services and Streamlit for real-time, interactive user
interfaces, enabling seamless doctor appointment management via natural language queries.
- mproved query-to-response turnaround time by ˜60% through automated slot filtering and decision routing, reducing manual
filtering from ˜30 seconds to under 12 seconds across 4,000+ simulated booking records.
</span>

---

## 💬 RAG-Powered Customer Support Agent  
[🔗 GitHub Repo](https://github.com/raviraj988/RAG_Customer_support_System)  
![RAG Chatbot](../images/projects/rag-chatbot.png)
<span style="font-size:12px;">
- Built an end-to-end ETL pipeline with Pandas to parse Flipkart reviews into LangChain documents, then ingested them into
AstraDB Vector Store for high-performance, semantic top-k retrieval.
- Integrated Google Gemini-1.5-Pro embeddings and LangChain’s ChatPromptTemplate to orchestrate context-aware retrieval and
natural-language answer generation for real-time product recommendations.
- Developed a RESTful FastAPI backend with environment-driven configuration with PyYAML, python-dotenv and integrated secure
secret management, paired with a modular AJAX chat UI using Jinja2, Bootstrap 4 and jQuery.

</span>

---

🧠 More projects coming soon...

