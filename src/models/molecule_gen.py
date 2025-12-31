import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist, squareform

class MoleculeBuilder:
    """
    Decodes 3D coordinates from the Diffusion Model back into valid RDKit Molecules.
    """
    def __init__(self, atom_decoder_map=None):
        # Default mapping: Index 0 -> Carbon, 1 -> Nitrogen, etc.
        self.atom_map = atom_decoder_map if atom_decoder_map else {0: 6, 1: 7, 2: 8, 3: 16} 

    def points_to_mol(self, coords, atom_types):
        """
        coords: [num_atoms, 3] tensor
        atom_types: [num_atoms] tensor (integers representing element type)
        """
        mol = Chem.RWMol()
        coords = coords.cpu().numpy()
        atom_types = atom_types.cpu().numpy()
        
        # 1. Add Atoms
        node_to_idx = {}
        for i, atom_idx in enumerate(atom_types):
            atomic_num = self.atom_map.get(atom_idx, 6) # Default to Carbon
            a = Chem.Atom(atomic_num)
            mol_idx = mol.AddAtom(a)
            node_to_idx[i] = mol_idx

        # 2. Add Conformer (3D positions)
        conf = Chem.Conformer(len(atom_types))
        for i, coord in enumerate(coords):
            conf.SetAtomPosition(i, (float(coord[0]), float(coord[1]), float(coord[2])))
        mol.AddConformer(conf)

        # 3. Infer Bonds based on Distance
        # Calculate distance matrix
        dist_mat = squareform(pdist(coords))
        
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = dist_mat[i, j]
                
                # Simple bond heuristic: if dist < 1.6 Angstroms, it's a bond
                if d < 1.6: 
                    # Check if bond already exists
                    if mol.GetBondBetweenAtoms(i, j) is None:
                        mol.AddBond(i, j, Chem.BondType.SINGLE)

        # 4. Sanitize
        try:
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            return mol
        except Exception as e:
            # Molecule might be invalid chemically (valency violation)
            # In a real pipeline, we would use a refinement step here
            print(f"⚠️ Validation warning: {e}")
            return mol

    def tensor_to_smiles(self, coords, atom_types):
        mol = self.points_to_mol(coords, atom_types)
        if mol:
            return Chem.MolToSmiles(mol)
        return None