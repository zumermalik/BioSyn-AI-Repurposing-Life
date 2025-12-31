import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.gnn_encoder import GNNEncoder
# We will import the dataset loader later

def run_training():
    print("🚀 Starting BioSyn Training Pipeline...")
    
    # Hyperparams (In real code, load from configs/model_config.yaml)
    input_dim = 9 # Based on our feature extractor
    hidden_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Model
    model = GNNEncoder(num_node_features=input_dim, embedding_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"✅ Model loaded on {device}")
    print(model)
    
    # TODO: Add Training Loop here in next phase
    
if __name__ == "__main__":
    run_training()