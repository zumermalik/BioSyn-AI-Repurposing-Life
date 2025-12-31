import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

class MoleculeFeaturizer:
    """
    Converts SMILES strings into Graph Tensors for Geometric Deep Learning.
    Pipeline Step: 2 (After Ingestion)
    """
    def __init__(self, allowed_atoms=None):
        self.allowed_atoms = allowed_atoms if allowed_atoms else ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P']

    def one_hot_encoding(self, x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def get_atom_features(self, atom):
        """
        Extracts atomic features: Symbol (one-hot), Degree, Charge, Hybridization
        """
        return torch.tensor(
            self.one_hot_encoding(atom.GetSymbol(), self.allowed_atoms) +
            self.one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
            [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +
            self.one_hot_encoding(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, 
                Chem.rdchem.HybridizationType.SP2, 
                Chem.rdchem.HybridizationType.SP3, 
                Chem.rdchem.HybridizationType.SP3D, 
                Chem.rdchem.HybridizationType.SP3D2
            ]),
            dtype=torch.float
        )

    def process_smiles(self, smiles_string):
        """
        Main entry point: SMILES -> (Node Features, Edge Indices)
        """
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return None

        # 1. Node Features (Atoms)
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        x = torch.stack(atom_features)

        # 2. Edge Index (Bonds) -> Adjacency format for PyTorch Geometric
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i]) # Undirected graph
        
        if not edge_indices:
            # Handle single atoms with no bonds
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        return x, edge_index

# --- Testing Block (Runs when called directly) ---
if __name__ == "__main__":
    # Test with Aspirin
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" 
    featurizer = MoleculeFeaturizer()
    x, edge_index = featurizer.process_smiles(test_smiles)
    
    print(f"🧪 Feature Extraction Test: Aspirin")
    print(f"   Node Features Shape: {x.shape} (9 atoms x N features)")
    print(f"   Edge Index Shape: {edge_index.shape} (2 x M edges)")