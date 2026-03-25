from rdkit import Chem
from rdkit.Chem import Descriptors


def generate_descriptors(smiles:str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    descriptores = {}
    for name, func in Descriptors._descList:
        try:
            descriptores[name] = func(mol)
        except:
            descriptores[name] = None
    return descriptores