import torch
import os
import sys
from pathlib import Path
import argparse

# --- PATH FIX: ENSURE ROOT IS FOUND ---
# This critical block ensures Python finds the 'src' folder 
# regardless of where you run the script from (VS Code, Terminal, Colab).
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))
# --------------------------------------

# Import your custom modules
from src.models.gnn_encoder import GNNEncoder
from src.models.diffusion import BioDiffusion
from src.chemistry.molecule_gen import MoleculeBuilder

def generate_candidate(protein_pdb_path, checkpoint_path, output_dir, num_samples=5):
    """
    Main Inference Function:
    1. Loads the trained GNN and Diffusion models.
    2. Simulates a protein context (Mock or Real).
    3. Generates 3D coordinates for drug candidates.
    4. Converts 3D points -> Chemical Graphs (SMILES).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧪 Starting BioSyn Inference on {device}...")
    
    # --- 1. Load Checkpoint ---
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        print("   >> Please run the training pipeline first.")
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
    print(f"   >> Target Protein: {os.path.basename(protein_pdb_path)}")
    
    # NOTE: In v0.1 (Mock Mode), we generate a random context vector.
    # In v0.2, this will use 'encoder(protein_graph)' to get real features.
    context = torch.randn(1, 128).to(device) 
    
    # --- 3. Reverse Diffusion (Generation) ---
    print(f"   >> Generating {num_samples} drug candidates...")
    
    # Initialize the Builder (KNN Strategy is safer for early models)
    builder = MoleculeBuilder()
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for i in range(num_samples):
        # Random size between 15 and 25 atoms
        num_atoms = torch.randint(15, 25, (1,)).item()
        
        # Expand context for batch size 1
        current_context = context.repeat(1, 1) 
        
        # Run Reverse Diffusion Process
        generated_coords = diffusion.sample(
            num_samples=1, 
            num_atoms=num_atoms, 
            context=current_context, 
            device=device
        )
        
        # Remove batch dimension -> [atoms, 3]
        coords = generated_coords[0]
        
        # Assign random atom types for v0.1 (Carbon, Nitrogen, Oxygen)
        atom_types = torch.randint(0, 3, (num_atoms,)).to(device)
        
        # --- 4. Decode to SMILES ---
        smiles = builder.tensor_to_smiles(coords, atom_types)
        
        if smiles:
            print(f"      🔹 Candidate {i+1}: {smiles}")
            results.append(smiles)
            
            # Save individual file
            save_path = os.path.join(output_dir, f"candidate_{i+1}.smi")
            with open(save_path, "w") as f:
                f.write(smiles)
        else:
            print(f"      🔸 Candidate {i+1}: Failed to construct valid molecule.")

    print(f"✅ Generation Complete. {len(results)} candidates saved to {output_dir}/")
    return results

if __name__ == "__main__":
    # Define default paths relative to the project root
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    # Default inputs
    pdb_path = base_dir / "data/raw/proteins/5R82.pdb"
    ckpt_path = base_dir / "checkpoints/biosyn_epoch_5.pt"
    out_dir = base_dir / "results"
    
    # Run Generation
    generate_candidate(
        protein_pdb_path=str(pdb_path),
        checkpoint_path=str(ckpt_path),
        output_dir=str(out_dir)
    )