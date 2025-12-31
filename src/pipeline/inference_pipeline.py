import torch
import os
import sys
from pathlib import Path
import argparse

# --- PATH FIX: ENSURE ROOT IS FOUND ---
# This looks 3 levels up from this file to find the project root
# (src/pipeline/inference_pipeline.py -> src/pipeline -> src -> ROOT)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))
# --------------------------------------

from src.models.gnn_encoder import GNNEncoder
from src.models.diffusion import BioDiffusion
from src.ingestion.parser import ProteinParser
from src.chemistry.molecule_gen import MoleculeBuilder

def generate_candidate(protein_pdb_path, checkpoint_path, output_dir, num_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧪 Starting BioSyn Inference on {device}...")
    
    # --- 1. Load Models ---
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return []

    print(f"   >> Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize Architectures (Must match training config)
    encoder = GNNEncoder(num_node_features=9, embedding_dim=128).to(device)
    diffusion = BioDiffusion(hidden_dim=128).to(device)
    
    # Load weights
    encoder.load_state_dict(checkpoint['encoder'])
    diffusion.load_state_dict(checkpoint['diffusion'])
    
    encoder.eval()
    diffusion.eval()

    # --- 2. Process Target Protein ---
    print(f"   >> Parsing target protein: {protein_pdb_path}")
    # In a real scenario, we would parse the PDB here.
    # parser = ProteinParser()
    # coords = parser.parse_pdb(protein_pdb_path)
    
    # Mocking the context vector for the demo (Shape: [1, 128])
    # This simulates the embedding the GNN would produce from the protein
    context = torch.randn(1, 128).to(device) 
    
    # --- 3. Reverse Diffusion (Generation) ---
    print(f"   >> Generating {num_samples} drug candidates...")
    builder = MoleculeBuilder()
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for i in range(num_samples):
        # Determine size of molecule (e.g., random between 15-25 atoms)
        num_atoms = torch.randint(15, 25, (1,)).item()
        
        # Expand context for batch size 1
        current_context = context.repeat(1, 1) 
        
        # Run Reverse Diffusion
        # 
        generated_coords = diffusion.sample(num_samples=1, num_atoms=num_atoms, context=current_context, device=device)
        
        # Remove batch dim -> [atoms, 3]
        coords = generated_coords[0]
        
        # Assign random atom types for the prototype (C, N, O)
        # In v2, the model would also predict these types.
        atom_types = torch.randint(0, 3, (num_atoms,)).to(device)
        
        # --- 4. Decode to SMILES ---
        smiles = builder.tensor_to_smiles(coords, atom_types)
        
        if smiles:
            print(f"      🔹 Candidate {i+1}: {smiles}")
            results.append(smiles)
            
            # Save to file
            with open(f"{output_dir}/candidate_{i+1}.smi", "w") as f:
                f.write(smiles)
        else:
            print(f"      🔸 Candidate {i+1}: Invalid geometry, discarded.")

    print(f"✅ Generation Complete. Saved to {output_dir}/")
    return results

if __name__ == "__main__":
    # Default Paths
    pdb_path = "data/raw/proteins/5R82.pdb"
    ckpt_path = "checkpoints/biosyn_epoch_5.pt"
    
    # Check if files exist before running
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Warning: Checkpoint {ckpt_path} not found. Running training first is recommended.")
    
    generate_candidate(
        protein_pdb_path=pdb_path,
        checkpoint_path=ckpt_path,
        output_dir="results"
    )