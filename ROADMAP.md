# 🗺️ BioSyn AI Development Roadmap

This document outlines the strategic milestones for the BioSyn AI project, moving from the current Proof-of-Concept (v0.1) to a production-grade Drug Discovery Platform (v1.0).

## 🟢 Phase 1: The Foundation (Completed)
> **Goal:** Establish end-to-end pipeline connectivity.
- [x] **Repo Structure:** Professional hybrid Python/TypeScript architecture.
- [x] **Data Ingestion:** Automated PDB fetching and parsing.
- [x] **Model Architecture:**
    - [x] GNN Encoder (PyTorch Geometric).
    - [x] 3D Diffusion Decoder (DDPM).
- [x] **Inference Logic:**
    - [x] Reverse diffusion sampling loop.
    - [x] "Crash-Proof" KNN Molecule Builder.
    - [x] SMILES string decoding.
- [x] **Environment:** Docker/Conda reproducible builds.

## 🟡 Phase 2: Intelligence Upgrade (Current Focus)
> **Goal:** Transition from "Mock Data" to "Real Chemistry".
- [ ] **Data Integration:**
    - [ ] Integrate **CrossDocked2020** dataset (50GB+).
    - [ ] Implement `ComplexDataset` loader for receptor-ligand pairs.
- [ ] **Training at Scale:**
    - [ ] Train for 100+ epochs on NVIDIA A100s.
    - [ ] Implement validation loss tracking (RMSE on coordinates).
- [ ] **Chemistry Refinement:**
    - [ ] Upgrade `MoleculeBuilder` to detect bond types (Single vs Double).
    - [ ] Add atom-type prediction (C, N, O, F, S, P).

## 🔵 Phase 3: Advanced Capabilities (v0.5)
> **Goal:** Improve drug viability (QED/Lipinski Rules).
- [ ] **Guided Diffusion:**
    - [ ] Implement classifier guidance to optimize for QED (Quantitative Estimation of Drug-likeness).
    - [ ] Optimize for SA (Synthetic Accessibility).
- [ ] **Visualization:**
    - [ ] Interactive 3D web dashboard (Streamlit/Py3Dmol).
    - [ ] Real-time docking score estimation.

## 🟣 Phase 4: Production (v1.0)
> **Goal:** Cloud deployment and lab validation.
- [ ] **Cloud API:** FastAPI wrapper for model inference.
- [ ] **Dockerization:** Full container support for Kubernetes deployment.
- [ ] **Lab Loop:** Integration with robotic wet-lab formats (CSV/SDF export).

---

### 📅 Release Schedule

| Version | Target Date | Key Feature |
| :--- | :--- | :--- |
| **v0.1-alpha** | *Released* | Functional Pipeline, Mock Weights |
| **v0.2-beta** | Q1 2026 | Real CrossDocked Training, Smart Bonds |
| **v0.5-rc** | Q2 2026 | Guided Diffusion, Web UI |
| **v1.0** | Q4 2026 | Production API, Cloud Scaling |

---
*To contribute to any of these milestones, please open a PR referencing the specific task.*