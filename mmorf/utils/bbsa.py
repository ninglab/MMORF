from rdkit import Chem

BBSA_ALERTS = {
    "aromatic_nitro": "[c][N+](=O)[O-]",
    "aromatic_amine": "[N;H1,H2][c]",
    "alkyl_halide": "[C;!a][F,Cl,Br,I]",
    "aryl_halide": "c[F,Cl,Br,I]",
    "acyl_halide": "C(=O)[F,Cl,Br,I]",
    "epoxide": "[OX2r3][#6]@[#6]",
    "alpha_beta_unsaturated_aldehyde": "C=CC=O",
    "alpha_beta_unsaturated_ketone": "C=CC(=O)C",
    "peroxide": "OO",
    "azo": "N=N",
    "diazo": "[N+]=[N-]",
    "aromatic_nitroso": "[NX2]=O",
    "aliphatic_nitroso": "[NX2]=O",
    "epichlorohydrin": "C1C(O)C1Cl",
    "oxime": "C=NO",
    "isocyanate": "N=C=O",
    "urethane": "NC(=O)O",
    "sulfonamide": "S(=O)(=O)N",
    "sulfide": "C-S-C",
    "thiirane": "C1SC1",
    "nitrosamine": "[NX2]N=O",
    "halo_methyl_ketone": "C(=O)CCl",
    "haloform": "C(Cl)(Cl)Cl",
    "aldehyde_aromatic": "cC=O",
    "alkyl_halide_secondary": "[CX4]([F,Cl,Br,I])[F,Cl,Br,I]",
    "aryl_sulfonate": "cS(=O)(=O)O",
    "aromatic_epoxide": "c1cc2OCOc2c1",
    "furan": "c1ccco1",
    "pyrrole": "c1cc[nH]c1",
    "imidazole": "c1cnc[nH]1",
    "peroxide2": "COOC",
    "methyloxirane": "C1CO1"
}

def bbsa_alerts(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    hits = []
    for name, smarts in BBSA_ALERTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            hits.append(name)
    return hits