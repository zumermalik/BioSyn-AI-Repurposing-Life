import torch
import numpy as np
from rdkit import Chem
from scipy.spatial.distance import pdist, squareform

class MoleculeBuilder:
    def __init__(self, atom_decoder_map=None):
        self.atom_map = atom_decoder_map if atom_decoder_map else {0: 6, 1: 7, 2: 8}

    def points_to_mol(self, coords, atom_types):
        mol = Chem.RWMol()
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().detach().numpy()
        if isinstance(atom_types, torch.Tensor):
            atom_types = atom_types.cpu().detach().numpy()
        
        # 1. Add Atoms
        for i, atom_idx in enumerate(atom_types):
            atomic_num = self.atom_map.get(int(atom_idx), 6)
            mol.AddAtom(Chem.Atom(atomic_num))

        # 2. Add Conformer
        conf = Chem.Conformer(len(atom_types))
        for i, coord in enumerate(coords):
            conf.SetAtomPosition(i, (float(coord[0]), float(coord[1]), float(coord[2])))
        mol.AddConformer(conf)

        # 3. Add Bonds (KNN Strategy for V0.1 Stability)
        if len(coords) > 1:
            dist_mat = squareform(pdist(coords))
            np.fill_diagonal(dist_mat, np.inf)
            
            for i in range(len(coords)):
                # Connect to closest neighbor only (prevent spiderwebs)
                closest_idx = np.argmin(dist_mat[i])
                if dist_mat[i, closest_idx] < 2.0:
                    if mol.GetBondBetweenAtoms(int(i), int(closest_idx)) is None:
                        mol.AddBond(int(i), int(closest_idx), Chem.BondType.SINGLE)

        try:
            mol.UpdatePropertyCache(strict=False)
            return mol
        except:
            return mol

    def tensor_to_smiles(self, coords, atom_types):
        mol = self.points_to_mol(coords, atom_types)
        if mol:
            try:
                return Chem.MolToSmiles(mol)
            except:
                return None
        return None