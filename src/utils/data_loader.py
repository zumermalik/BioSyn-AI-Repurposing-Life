import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import os
from tqdm import tqdm

class BioSynDataset(Dataset):
    """
    Custom Dataset for BioSyn AI.
    Pairs Protein Graphs (Conditioning) with Ligand 3D Coordinates (Target).
    """
    def __init__(self, root, mode='train', transform=None, pre_transform=None, mock=False):
        self.mode = mode
        self.mock = mock # Set to True to test pipeline without real files
        super(BioSynDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_train.pt'] if self.mode == 'train' else ['data_val.pt']

    def len(self):
        # In mock mode, pretend we have 100 samples
        return 100 if self.mock else len(self.processed_file_names)

    def get(self, idx):
        if self.mock:
            return self._generate_mock_data()
        
        # Real implementation would load processed .pt files here
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def _generate_mock_data(self):
        """
        Generates valid dummy tensors to test the architecture dimensions.
        """
        # 1. Ligand Data (The thing we generate)
        num_atoms = np.random.randint(10, 30)
        ligand_coords = torch.randn((num_atoms, 3), dtype=torch.float32)
        
        # 2. Protein Data (The condition)
        num_protein_nodes = np.random.randint(50, 100)
        protein_features = torch.randn((num_protein_nodes, 9), dtype=torch.float32) # 9 features from our featurizer
        
        # Create random edges for protein graph
        edge_index = torch.randint(0, num_protein_nodes, (2, num_protein_nodes * 2))
        
        # Batch vector (required for PyG pooling)
        batch = torch.zeros(num_protein_nodes, dtype=torch.long)

        return {
            'ligand_pos': ligand_coords,
            'protein_x': protein_features,
            'protein_edge_index': edge_index,
            'protein_batch': batch
        }

if __name__ == "__main__":
    ds = BioSynDataset(root='./data', mock=True)
    sample = ds[0]
    print(f"📦 Data Loader Test: {sample.keys()}")