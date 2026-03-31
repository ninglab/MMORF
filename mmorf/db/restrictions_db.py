# restriction_database.py
# Requires RDKit: https://www.rdkit.org/
import json
from typing import List, Dict, Any, Optional
from rdkit import Chem
from ..utils.tanimoto import tanimoto_similarity


class RestrictionDatabase:
    """
    Load and query a collection of restriction objects.
    Each entry is expected to be a dict with at least:
      - "apply_when": list of SMARTS strings
      - "product": SMILES string
    Other keys (e.g. "molecules", "specific_reactions", "reaction_templates", "max_depth")
    are preserved and returned with matches.
    """

    def __init__(self):
        # raw entries in the order loaded
        self._entries: List[Dict[str, Any]] = []
        # compiled patterns for each entry (list of RDKit Mol for SMARTS)
        self._patterns: List[List[Chem.Mol]] = []
        # canonical product SMILES for each entry (or None if unparsable)
        self._product_smiles: List[Optional[str]] = []
        # index from canonical product smiles -> list of entry indices
        self._product_index: Dict[str, List[int]] = {}

    def load_from_jsonl(self, path: str, skip_invalid: bool = True, skip_partial: bool = True, include_rationale=False) -> None:
        """
        Load entries from a JSONL file (one JSON object per line).
        If skip_invalid is True, entries with unparsable SMARTS/SMILES are skipped.
        """
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON at line {line_no}: {line}")
                    if skip_invalid:
                        continue
                    raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e
                try:
                    self._add_entry(obj, raise_invalid=not skip_partial, include_rationale=include_rationale)
                except Exception:
                    if skip_invalid:
                        continue
                    raise

    def _add_entry(self, entry: Dict[str, Any], raise_invalid: bool = True, include_rationale=False) -> None:
        # validate presence of keys
        apply_when = entry.get("apply_when")
        product = entry.get("product")
        if not isinstance(apply_when, list):
            raise ValueError("entry['apply_when'] must be a list of SMARTS strings")
        # compile SMARTS patterns
        compiled = []
        for smarts in apply_when:
            if not isinstance(smarts, str) and raise_invalid:
                raise ValueError("SMARTS must be strings")
            elif not isinstance(smarts, str):
                continue
            patt = Chem.MolFromSmarts(smarts)
            if patt is None and raise_invalid:
                raise ValueError(f"Invalid SMARTS: {smarts}")
            elif patt is None:
                continue
            compiled.append(patt)
        # canonicalize product smiles if present
        product_can = None
        if product is not None:
            if not isinstance(product, str):
                raise ValueError("product must be a SMILES string")
            mol = Chem.MolFromSmiles(product)
            if mol is None:
                raise ValueError(f"Invalid product SMILES: {product}")
            product_can = Chem.MolToSmiles(mol, isomericSmiles=False)
        # check all restriction strings:
        keys = ["molecules", "specific_reactions", "reaction_templates"]
        if include_rationale:
            keys.append("rationale")
        for key in keys:
            items = [z for z in entry.get(key, [])]
            for item in items:
                if not isinstance(item, str) and raise_invalid:
                    raise ValueError(f"entry['{key}'] must be a list of strings")
                elif not isinstance(item, str):
                    continue
                if key == "molecules":
                    mol = Chem.MolFromSmiles(item)
                    if mol is None and raise_invalid:
                        raise ValueError(f"Invalid molecule SMILES: {item}")
                    elif mol is None:
                        entry[key].remove(item)
                    entry[key][entry[key].index(item)] = Chem.MolToSmiles(mol, isomericSmiles=False) # canonicalize without isomerism
                if key == "specific_reactions":
                    try:
                        reactants, product = item.split(">>")
                        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
                        product_mol = Chem.MolFromSmiles(product)
                        if any(m is None for m in reactant_mols) or product_mol is None:
                            if raise_invalid:
                                raise ValueError(f"Invalid reaction SMILES: {item}")
                            else:
                                entry[key].remove(item)
                                continue
                    except Exception:
                        if raise_invalid:
                            raise ValueError(f"Invalid reaction SMILES: {item}")
                        else:
                            entry[key].remove(item)
                            continue    
        # store entry
        index = len(self._entries)
        self._entries.append(entry)
        self._patterns.append(compiled)
        self._product_smiles.append(product_can)
        if product_can is not None:
            self._product_index.setdefault(product_can, []).append(index)

    def find_applicable_filters(
        self,
        target_smiles: str,
        similarity_threshold: Optional[float] = None,
        similarity_logic: str = "or",
    ) -> List[Dict[str, Any]]:
        """
        Return a list of restriction entries that apply to the target molecule (and optionally to the product).
        - target_smiles: SMILES of the substrate/reactant to test SMARTS against.
        - product_smiles: if provided, only entries whose product matches (exact canonical SMILES) are considered.
        - require_all_apply_when: if True, require that ALL SMARTS in an entry match; otherwise any SMARTS match suffices.
        """
        similarity_logic = similarity_logic.lower()
        if similarity_logic not in {"or", "and", "max"}:
            raise ValueError(f"Invalid similarity_logic: {similarity_logic}")
        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is None:
            raise ValueError(f"Invalid target SMILES: {target_smiles}")
        candidate_indices = range(len(self._entries))
        matches: List[Dict[str, Any]] = []
        for idx in candidate_indices:
            patterns = self._patterns[idx]
            if not patterns:
                continue
            else:
                ok = any(target_mol.HasSubstructMatch(p) for p in patterns)
                ok = ok or (Chem.MolToSmiles(target_mol, isomericSmiles=False) == self._product_smiles[idx])
                if similarity_threshold is not None and self._product_smiles[idx] is not None:
                    prod_mol = Chem.MolFromSmiles(self._product_smiles[idx])
                    if prod_mol is not None:
                        sim = tanimoto_similarity(Chem.MolToSmiles(target_mol, isomericSmiles=False), self._product_smiles[idx], sanitize=True)
                        if similarity_logic == "or":
                            ok = ok or (sim >= similarity_threshold)
                        else:  # similarity_logic == "and"
                            ok = ok and (sim >= similarity_threshold)
            if ok:
                matches.append(self._entries[idx])
        if matches and similarity_logic == "max":
            # sort by descending similarity
            matches.sort(key=lambda entry: tanimoto_similarity(
                Chem.MolToSmiles(target_mol, isomericSmiles=False),
                entry.get("product", ""),
                sanitize=True
            ), reverse=True)
        return matches

    def all_entries(self) -> List[Dict[str, Any]]:
        "Return a copy of all loaded entries."
        return list(self._entries)
    
    def merge_random_filters(self, n: int, max_depth="max") -> Dict[str, Any]:
        """
        Merge n random restriction entries into a single restriction dict.
        Lists are combined and deduplicated; max_depth takes the minimum (if any).
        """
        import random
        if n > len(self._entries):
            raise ValueError(f"Requested {n} entries, but only {len(self._entries)} are available.")
        sampled = random.sample(self._entries, n)
        merged: Dict[str, Any] = {
            "molecules": [],
            "specific_reactions": [],
            "reaction_templates": [],
            "max_depth": None,
        }
        for entry in sampled:
            for key in ["molecules", "specific_reactions", "reaction_templates"]:
                items = entry.get(key, [])
                if isinstance(items, list):
                    merged[key].extend(items)
            depth = entry.get("max_depth")
            if isinstance(depth, int):
                if merged["max_depth"] is None:
                    merged["max_depth"] = depth
                if max_depth == "min":
                    if depth == -1:
                        continue
                    merged["max_depth"] = min(merged["max_depth"], depth)
                elif max_depth == "max":
                    if depth == -1 or merged["max_depth"] == -1:
                        merged["max_depth"] = -1
                    else:
                        merged["max_depth"] = max(merged["max_depth"], depth)
                elif max_depth == "none":
                    merged["max_depth"] = -1
                elif isinstance(max_depth, int):
                    merged["max_depth"] = max_depth
                else:
                    raise ValueError(f"Invalid max_depth option: {max_depth}")
        # deduplicate lists
        for key in ["molecules", "specific_reactions", "reaction_templates"]:
            merged[key] = list(set(merged[key]))
        return merged
    
    def _merge_filters(self, filters, max_depth="max") -> Dict[str, Any]:
        merged: Dict[str, Any] = {
            "molecules": [],
            "specific_reactions": [],
            "reaction_templates": [],
            "max_depth": None,
        }
        for entry in filters:
            for key in ["molecules", "specific_reactions", "reaction_templates"]:
                items = entry.get(key, [])
                if isinstance(items, list):
                    merged[key].extend(items)
            depth = entry.get("max_depth")
            if isinstance(depth, int):
                if merged["max_depth"] is None:
                    merged["max_depth"] = depth
                if max_depth == "min":
                    if depth == -1:
                        continue
                    merged["max_depth"] = min(merged["max_depth"], depth)
                elif max_depth == "max":
                    if depth == -1 or merged["max_depth"] == -1:
                        merged["max_depth"] = -1
                    else:
                        merged["max_depth"] = max(merged["max_depth"], depth)
                elif max_depth == "none":
                    merged["max_depth"] = -1
                elif isinstance(max_depth, int):
                    merged["max_depth"] = max_depth
                else:
                    raise ValueError(f"Invalid max_depth option: {max_depth}")
        # deduplicate lists
        for key in ["molecules", "specific_reactions", "reaction_templates"]:
            merged[key] = list(set(merged[key]))
        return merged
    
    def merge_filters_matching(self, target_smiles: str, max_depth="max", similarity_threshold=None, similarity_logic="or") -> Dict[str, Any]:
        """
        Merge all applicable restriction entries for the target molecule into a single restriction dict.
        Lists are combined and deduplicated; max_depth takes the minimum (if any).
        """
        applicable = self.find_applicable_filters(target_smiles, similarity_threshold=similarity_threshold, similarity_logic=similarity_logic)
        merged: Dict[str, Any] = {
            "molecules": [],
            "specific_reactions": [],
            "reaction_templates": [],
            "max_depth": None,
        }
        for entry in applicable:
            for key in ["molecules", "specific_reactions", "reaction_templates"]:
                items = entry.get(key, [])
                if isinstance(items, list):
                    merged[key].extend(items)
            depth = entry.get("max_depth")
            if isinstance(depth, int):
                if merged["max_depth"] is None:
                    merged["max_depth"] = depth
                if max_depth == "min":
                    if depth == -1:
                        continue
                    merged["max_depth"] = min(merged["max_depth"], depth)
                elif max_depth == "max":
                    if depth == -1 or merged["max_depth"] == -1:
                        merged["max_depth"] = -1
                    else:
                        merged["max_depth"] = max(merged["max_depth"], depth)
                elif max_depth == "none":
                    merged["max_depth"] = -1
                elif isinstance(max_depth, int):
                    merged["max_depth"] = max_depth
                else:
                    raise ValueError(f"Invalid max_depth option: {max_depth}")
        # deduplicate lists
        for key in ["molecules", "specific_reactions", "reaction_templates"]:
            merged[key] = list(set(merged[key]))
        return merged


# Example usage (remove or adapt when integrating into your project):
if __name__ == "__main__":
    db = RestrictionDatabase()
    # db.load_from_jsonl("restrictions.jsonl")
    # matches = db.find_applicable_filters("CCO", product_smiles="CC(=O)O")
    # print(matches)