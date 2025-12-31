import os
from Bio.PDB import PDBParser
import torch
import numpy as np

class ProteinParser:
    """
    Parses raw PDB files into geometric tensors for the AI model.
    """
    def __init__(self):
        self.parser = PDBParser(QUIET=True)

    def parse_pdb(self, pdb_path):
        """
        Reads a PDB file and returns the Alpha-Carbon coordinates (backbone).
        """
        if not os.path.exists(pdb_path):
            print(f"⚠️ Warning: File {pdb_path} not found.")
            return None

        structure = self.parser.get_structure('protein', pdb_path)
        coords = []
        
        # We focus on the first model in the file
        model = structure[0]
        
        for chain in model:
            for residue in chain:
                # Filter for amino acids only (ignore water/heteroatoms)
                if residue.id[0] == ' ':
                    if 'CA' in residue:
                        # Get Carbon Alpha coordinates
                        coords.append(residue['CA'].get_coord())
        
        if len(coords) == 0:
            return None

        # Convert to PyTorch Tensor
        return torch.tensor(np.array(coords), dtype=torch.float32)

    def get_pocket_center(self, coords_tensor):
        """
        Calculates the geometric center of the protein (simplified pocket).
        """
        return torch.mean(coords_tensor, dim=0)

if __name__ == "__main__":
    # Test logic
    # Create a dummy file or point to a real one if available
    print("🧬 Protein Parser ready.")