# BioSyn AI: Repurposing Life

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0) [![Status](https://img.shields.io/badge/Status-v0.1%20(Alpha)-orange.svg)]() [![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **"Repurposing Life through Geometric Deep Learning."**

**BioSyn AI** is an end-to-end generative pipeline designed to imagine novel drug candidates (ligands) that bind to specific protein targets. It combines **E(n)-Equivariant Graph Neural Networks (GNNs)** for protein structure encoding with **3D Denoising Diffusion Probabilistic Models (DDPMs)** for molecule generation.

---

## 🧬 Architecture

The pipeline follows a closed-loop generative process:

```mermaid
graph LR
    A[Protein PDB] -->|Ingestion Engine| B(Geometric Graph)
    B -->|GNN Encoder| C[Context Embedding]
    D[Gaussian Noise] -->|Diffusion Model| E{Reverse Process}
    C --> E
    E -->|Denoising| F[3D Atom Cloud]
    F -->|KNN Builder| G[SMILES Candidate]
   


 ```1. **Ingestion:** TypeScript engine fetches raw PDB/SDF files from biological databases.
2. **Encoder:** A GNN extracts geometric features (invariant to rotation/translation) from the protein pocket.
3. **Decoder:** A Diffusion model iteratively refines random noise into stable 3D molecular structures conditioned on the protein embedding.
4. **Inference:** A robust `MoleculeBuilder` reconstructs valid chemical graphs from 3D point clouds using K-Nearest Neighbors (KNN) logic.

---

## ⚡ Quick Start

### Prerequisites

* Python 3.10+
* Node.js (v16+)
* CUDA-enabled GPU (Recommended)

### 1. Installation

Clone the repository and set up the hybrid environment.

```bash
# Clone the repo
git clone [https://github.com/zumermalik/BioSyn-AI-Repurposing-Life.git](https://github.com/zumermalik/BioSyn-AI-Repurposing-Life.git)
cd BioSyn-AI-Repurposing-Life

# Set up Python Environment (Conda recommended for RDKit compatibility)
conda create -n biosyn python=3.10 -y
conda activate biosyn

# Install Core Dependencies
pip install -r requirements.txt

# Install Ingestion Engine (TypeScript)
npm install

```

### 2. Run the Pipeline (Zero to Hero)

You can run the entire inference stack with a single command. This will load the pre-trained checkpoint and generate candidates for the target protein `5R82`.

```bash
# Run Inference
python src/pipeline/inference_pipeline.py

```

*Expected Output:*

```text
🧪 Starting BioSyn Inference on cuda...
   >> Target Protein: 5R82.pdb
   >> Loading checkpoint: checkpoints/biosyn_epoch_5.pt
   >> Generating 5 drug candidates...
      🔹 Candidate 1: CC(=O)Nc1ccc(O)cc1
      🔹 Candidate 2: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
✅ Generation Complete. 5 candidates saved to results/

```

---

## 📂 Project Structure

```bash
BioSyn-AI-Repurposing-Life/
├── .github/              # GitHub Actions & Templates
├── .vscode/              # Editor Configuration
├── checkpoints/          # Trained Model Weights (.pt)
│   └── biosyn_epoch_5.pt
├── configs/              # Hyperparameter Configuration
│   └── training_config.yaml
├── data/                 # Data Storage
│   ├── processed/        # PyTorch Geometric Tensors
│   └── raw/              # Original PDB/SDF Files
├── notebooks/            # Jupyter Prototyping Environments
├── results/              # Generated SMILES (.smi) & Visualizations
├── src/                  # Source Code
│   ├── chemistry/        # RDKit Logic & Molecule Builders
│   ├── ingestion/        # TypeScript/Python Data Fetchers
│   ├── models/           # GNN Encoder & Diffusion Decoder
│   └── pipeline/         # Training & Inference Orchestration
├── tests/                # Unit Tests
├── .gitignore            # Git Ignore Rules
├── LICENSE               # Apache 2.0 License
├── package.json          # Node.js Dependencies
├── README.md             # Project Documentation
├── requirements.txt      # Python Dependencies
├── roadmap.md            # Future Development Plans
└── tsconfig.json         # TypeScript Configuration

```

## 🛠️ Development & Testing

We use `pytest` for unit testing the geometric logic and chemical validity.

```bash
# Run the full test suite
pytest tests/

```

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. **Fork the Project**
2. **Create your Feature Branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your Changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the Branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Contribution Standards

* **Code Style:** Please use `black` for Python formatting.
* **Testing:** Ensure all new modules have accompanying tests in `tests/`.
* **Data:** Do not commit large datasets (PDB/SDF files) to Git. Use the `data/` folder which is ignored by default.

---

## 📜 Citation & License

This project is licensed under the Apache 2.0 License. If you use this architecture in your research, please link back to this repository.

---

*Maintained by the Builders.*