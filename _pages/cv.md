---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

📄 **Download My Full Resume (PDF)**  
👉 [Click here to view/download](../files/Resume_Gen_AI_Research.pdf)

---

## 🎓 Education & Honors

**Case Western Reserve University**, Cleveland, Ohio  
*Master of Science in Computer Science* — *Expected May 2025*  
- Coursework: Machine Learning, Computer Vision, Robotics, High Performant Systems for AI, Probabilistic Graphical Models, Algorithms, Computer Networks  
- Observer: Statistical NLP, Deep Generative Models, Quantum Computing, Reinforcement Learning, ML on Graphs

---

## 🛠 Technical Skills

- **Languages & Frameworks**: Python, SQL, Java, C++, JavaScript, TensorFlow, PyTorch, HuggingFace Transformers, FastAPI, Django, Streamlit, Gradio, LangChain, LangGraph  
- **ML & RL**: Classification, Regression, Clustering, SVM, Random Forest, CNN, RNN, Transformers, PPO, A2C, DQN, SAC, Multi-Agent RL  
- **Generative AI**: Diffusion Models, GANs, VAEs, Fine-Tuning (LoRA, QLoRA, PEFT), Image Generation, Inpainting, Upscaling  
- **RAG & LLM**: FAISS, Pinecone, VectorDBs, LangChain Agents, Embedding Models, Prompt Engineering, Top-K Retrieval, Document Chunking  
- **MLOps & Infrastructure**: MLflow, Docker, Kubernetes, Helm, AWS (EC2, S3, ECR), GCP, Azure, CI/CD (GitHub Actions, GitLab, Jenkins), Monitoring (Prometheus, Grafana), ONNX Runtime, Quantization, Scalable Model Serving

---

## 💼 Professional Experience

**Tata Consultancy Services**, Hyderabad, India  
*Machine Learning Engineer (2019–2023)*  
- Designed scalable end-to-end ML pipelines for a Bank Member Complaint System  
- Built and fine-tuned BERT, RoBERTa, and Longformer for NLP tasks (NER, severity prediction)  
- Managed CI/CD workflows with Jenkins and MLflow, deployed using Docker/Kubernetes  
- Implemented automated validation, drift detection, and observability dashboards using Prometheus/Grafana  
- Ensured model reproducibility with MLflow artifacts, Docker image tags, and Conda environments  

*Data Engineer*  
- Created secure data pipelines using Hive, Snowflake, DBT, UNIX scripting  
- Designed a Data Quality Engine and subject-area marts  
- Delivered end-to-end pipelines for high-volume financial data systems

---

## 🧪 Research Experience

**Optimizing RAG with Multi-Agent RL** — *with Dr. Soumya Ray*  
- PPO-based cooperative agents for RAG with LoRA warm-start and reward shaping  
- Used MiniLM + FAISS retrieval over custom SQuAD corpus

**Multimodal Transformer for Image-Conditioned Generation**  
- Built SigLIP-Gemma architecture with KV caching, rotary embeddings, and PaLI-style tokenization  
- Achieved +3.7 BLEU-4 over greedy baseline

**Latent Diffusion Transformer (Stable Diffusion from scratch)**  
- Built full pipeline: CLIP encoder, VAE, DDPM, U-Net denoiser  
- Reduced inference steps from 100 to 50 with custom scheduler  
- Matched v1.5 baseline: FID 12.4, CLIPScore 0.32

**Semi-Supervised Learning with Generative Models** — *with Dr. Soumya Ray*  
- Implemented Kingma et al.'s M2 with enhancements to ELBO, entropy minimization, and mutual information  
- +4% accuracy on MNIST, +2.5% on CIFAR-10

**3D Point Cloud Segmentation Using 2D Voting** — *with Dr. Yu Yin*  
- Mapped OneFormer 2D semantic masks onto 3D LiDAR using RGB, depth, and voting  
- Achieved 96.5% accuracy with reduced overhead

---

## 🚀 Projects

**Multi-Agent Medical Appointment System**  
- Modular LangChain + LangGraph agents with FastAPI + Streamlit UI  
- Reduced latency ~60% across 4000+ simulated bookings

**RAG-Powered Customer Support Agent**  
- Gemini-1.5-Pro + LangChain over Flipkart reviews in AstraDB  
- Real-time QA via AJAX chat interface with secure FastAPI backend

---

Let me know if you want this resume content to be auto-synced from your LaTeX or Overleaf source, or broken into downloadable sections like research, experience, and skills.
