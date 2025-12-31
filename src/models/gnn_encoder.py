import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNEncoder(nn.Module):
    """
    Graph Convolutional Network to encode molecular graphs into latent vectors.
    """
    def __init__(self, num_node_features, embedding_dim=128):
        super(GNNEncoder, self).__init__()
        
        # 3 Graph Convolution Layers
        self.conv1 = GCNConv(num_node_features, embedding_dim)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)
        self.conv3 = GCNConv(embedding_dim, embedding_dim)
        
        # Output processing
        self.lin = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, data):
        """
        Args:
            data: A PyTorch Geometric Batch object containing:
                  - x: Node features [num_atoms, num_features]
                  - edge_index: Graph connectivity [2, num_edges]
                  - batch: Batch vector [num_atoms]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        # 2. Readout layer (Pool node embeddings into a graph embedding)
        x = global_mean_pool(x, batch)  # [batch_size, embedding_dim]

        # 3. Final linear projection
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

if __name__ == "__main__":
    # Quick dimensionality test
    print("🧠 GNN Encoder initialized.")
    model = GNNEncoder(num_node_features=9, embedding_dim=64)
    print(model)