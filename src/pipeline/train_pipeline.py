import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.gnn_encoder import GNNEncoder
from src.models.diffusion import BioDiffusion
from src.utils.data_loader import BioSynDataset
from torch_geometric.data import Batch

def collate_fn(batch_list):
    """
    Custom collator to handle complex dictionary structures in batches.
    """
    ligand_pos = []
    protein_graphs = []
    
    # Simple batching logic for the mock data
    max_atoms = max([b['ligand_pos'].shape[0] for b in batch_list])
    
    padded_ligands = []
    
    for item in batch_list:
        # Pad ligand to max size for batching (simplified)
        pos = item['ligand_pos']
        pad = torch.zeros((max_atoms - pos.shape[0], 3))
        padded_ligands.append(torch.cat([pos, pad]))
        
        # Create PyG Data object for protein
        protein_graphs.append(Data(
            x=item['protein_x'], 
            edge_index=item['protein_edge_index']
        ))

    # Batch the protein graphs using PyTorch Geometric's Batch.from_data_list
    protein_batch = Batch.from_data_list(protein_graphs)
    
    return torch.stack(padded_ligands), protein_batch

def train(config_path='configs/model_config.yaml'):
    # --- 1. Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting BioSyn Training on {device}...")

    # Create Checkpoint Dir
    os.makedirs('checkpoints', exist_ok=True)

    # --- 2. Initialize Models ---
    # GNN Encoder: Extracts features from Protein target
    encoder = GNNEncoder(num_node_features=9, embedding_dim=128).to(device)
    
    # Diffusion Model: Generates Ligand
    diffusion = BioDiffusion(hidden_dim=128).to(device)

    # Optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(diffusion.parameters()), lr=1e-3)

    # --- 3. Load Data ---
    # We use mock=True so you can run this immediately without data files
    dataset = BioSynDataset(root='./data', mock=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # --- 4. Training Loop ---
    epochs = 5 # Short run for testing
    
    encoder.train()
    diffusion.train()

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        for batch_ligands, batch_proteins in pbar:
            batch_ligands = batch_ligands.to(device)
            batch_proteins = batch_proteins.to(device)
            
            optimizer.zero_grad()

            # A. Encode Protein Condition (GNN Forward)
            # protein_context shape: [batch_size, 128]
            protein_context = encoder(batch_proteins)

            # B. Diffusion Training Step
            # Sample random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (batch_ligands.shape[0],), device=device).long()
            
            # Calculate Diffusion Loss
            loss = diffusion.p_losses(batch_ligands, t, protein_context)
            
            # C. Backprop
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        print(f"📉 Epoch {epoch+1} Average Loss: {epoch_loss / len(loader):.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'encoder': encoder.state_dict(),
                'diffusion': diffusion.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f"checkpoints/biosyn_epoch_{epoch+1}.pt")
            print("💾 Checkpoint saved.")

if __name__ == "__main__":
    train()