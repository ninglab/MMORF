from rdkit import Chem
from rdkit.Chem import rdChemReactions

def reaction_from_smiles(reactant_smiles, product_smiles):
    rxn = rdChemReactions.ChemicalReaction()
    for smi in reactant_smiles.split('.'):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid reactant SMILES: {smi}")
        rxn.AddReactantTemplate(mol)
    for smi in product_smiles.split('.'):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid product SMILES: {smi}")
        rxn.AddProductTemplate(mol)
    return rxn