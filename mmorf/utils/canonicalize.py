from rdkit import Chem

def canonicalize(smiles, isomeric=True):
    """Returns the canonical SMILES representation of the provided SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES."
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric, canonical=True)