import sys
import torch
import pytest
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.gnn_encoder import GNNEncoder
from src.models.diffusion import BioDiffusion
from src.chemistry.feature_extraction import MoleculeFeaturizer

def test_gnn_dimensions():
    """Ensure GNN outputs the correct embedding shape."""
    num_nodes = 10
    in_features = 9
    out_dim = 128
    
    # Mock PyG Data Batch
    from torch_geometric.data import Data, Batch
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn((num_nodes, in_features))
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data]) # Batch of 1
    
    model = GNNEncoder(num_node_features=in_features, embedding_dim=out_dim)
    output = model(batch)
    
    assert output.shape == (1, out_dim), f"Expected (1, {out_dim}), got {output.shape}"

def test_diffusion_forward():
    """Ensure diffusion model accepts inputs without crashing."""
    batch_size = 2
    atoms = 5
    hidden = 128
    
    model = BioDiffusion(hidden_dim=hidden)
    
    x = torch.randn(batch_size, atoms, 3)
    t = torch.randint(0, 10, (batch_size,))
    context = torch.randn(batch_size, hidden)
    
    # Check loss calculation
    loss = model.p_losses(x, t, context)
    assert not torch.isnan(loss), "Loss is NaN"

def test_chemistry_logic():
    """Ensure featurizer handles SMILES correctly."""
    featurizer = MoleculeFeaturizer()
    smiles = "C" # Methane
    x, edge_index = featurizer.process_smiles(smiles)
    
    # Methane: 1 Carbon atom
    assert x.shape[0] == 1, "Methane should have 1 node"