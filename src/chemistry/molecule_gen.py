import torch
import numpy as np
from rdkit import Chem
from scipy.spatial.distance import pdist, squareform

class MoleculeBuilder:
    def __init__(self, atom_decoder_map=None):
        # Index 0: Carbon (Valence 4), 1: Nitrogen (3), 2: Oxygen (2)
        self.atom_map = atom_decoder_map if atom_decoder_map else {0: 6, 1: 7, 2: 8}
        # Max bonds allowed per atom type
        self.valency_map = {6: 4, 7: 3, 8: 2}

    def points_to_mol(self, coords, atom_types):
        mol = Chem.RWMol()
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().detach().numpy()
        if isinstance(atom_types, torch.Tensor):
            atom_types = atom_types.cpu().detach().numpy()
        
        # Track current valence (bond count) for each atom
        current_valence = np.zeros(len(atom_types), dtype=int)
        
        # 1. Add Atoms
        for i, atom_idx in enumerate(atom_types):
            atomic_num = self.atom_map.get(int(atom_idx), 6)
            mol.AddAtom(Chem.Atom(atomic_num))

        # 2. Add Conformer
        conf = Chem.Conformer(len(atom_types))
        for i, coord in enumerate(coords):
            conf.SetAtomPosition(i, (float(coord[0]), float(coord[1]), float(coord[2])))
        mol.AddConformer(conf)

        # 3. Add Bonds (Strict Mode)
        if len(coords) > 1:
            dist_mat = squareform(pdist(coords))
            # Sort bonds by distance (shortest first) to prioritize strong connections
            # We get indices of the upper triangle of the matrix
            indices = np.triu_indices_from(dist_mat, k=1)
            distances = dist_mat[indices]
            sorted_indices = np.argsort(distances)
            
            for idx in sorted_indices:
                i, j = indices[0][idx], indices[1][idx]
                d = distances[idx]
                
                # Heuristic: Bond length < 1.6A
                if d < 1.6:
                    # CHECK VALENCY BEFORE ADDING
                    atom_i_type = self.atom_map.get(int(atom_types[i]), 6)
                    atom_j_type = self.atom_map.get(int(atom_types[j]), 6)
                    
                    max_v_i = self.valency_map.get(atom_i_type, 4)
                    max_v_j = self.valency_map.get(atom_j_type, 4)
                    
                    if current_valence[i] < max_v_i and current_valence[j] < max_v_j:
                        mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
                        current_valence[i] += 1
                        current_valence[j] += 1

        # 4. Final Sanitize (Try/Except to prevent crashing)
        try:
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            # If sanitization fails, return the raw molecule anyway so we can see it
            return mol

    def tensor_to_smiles(self, coords, atom_types):
        mol = self.points_to_mol(coords, atom_types)
        if mol:
            try:
                return Chem.MolToSmiles(mol)
            except:
                return None
        return None