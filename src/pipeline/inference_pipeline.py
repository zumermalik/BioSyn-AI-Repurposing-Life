import torch
import os
import sys
from pathlib import Path
import argparse

# Path hacking for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.gnn_encoder import GNNEncoder
from src.models.diffusion import BioDiffusion
from src.ingestion.parser import ProteinParser
from src.chemistry.molecule_gen import MoleculeBuilder
from src.chemistry.feature_extraction import MoleculeFeaturizer 
# Note: We import Featurizer just to get dimensions, though we don't featurize the ligand here (we generate it)

def generate_candidate(protein_pdb_path, checkpoint_path, output_dir, num_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧪 Starting BioSyn Inference on {device}...")
    
    # --- 1. Load Models ---
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
    parser = ProteinParser()
    
    # In a real scenario, we'd need to convert the PDB to a Graph here.
    # For this prototype, we will simulate the protein embedding to match the mock training.
    # Why? Because converting a raw PDB to the exact Graph Tensor required by GNNEncoder 
    # requires a complex topological function (edges based on angstrom distance).
    # To keep this runnable for you immediately, we use a placeholder embedding derived from the file.
    
    # Simulating a "forward pass" of the protein graph for demonstration
    # In full production, you would call: protein_graph = parser.to_graph(protein_pdb_path)
    # context = encoder(protein_graph)
    
    # Mocking the context vector for the demo (Shape: [1, 128])
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
        # sample() expects context to match batch size
        current_context = context.repeat(1, 1) 
        
        # Run Reverse Diffusion
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
    # Test Run
    # Create a dummy checkpoint if one doesn't exist just to test syntax
    if not os.path.exists("checkpoints/biosyn_epoch_5.pt"):
        print("⚠️ No checkpoint found. Please run train_pipeline.py first.")
    else:
        generate_candidate(
            protein_pdb_path="data/raw/proteins/5R82.pdb", # Ensure this file exists or use dummy
            checkpoint_path="checkpoints/biosyn_epoch_5.pt",
            output_dir="results"
        )