import os, requests, pickle

CACHE_DIR = os.environ.get("mmorf_CACHE_DIR", os.path.dirname(__file__))
ghs_cache = {}
if os.path.exists(CACHE_DIR + "/ghs_cache.pkl"):
    with open(CACHE_DIR + "/ghs_cache.pkl", "rb") as f:
        ghs_cache = pickle.load(f)

def get_ghs_from_pubchem(cas_or_smiles : str) -> set:
    # Step 1: Convert CAS to CID
    if cas_or_smiles in ghs_cache:
        return ghs_cache[cas_or_smiles] - {"Not"}
    
    url_cid = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{cas_or_smiles}/cids/JSON"
    try:
        cid_response = requests.get(url_cid, timeout=180)
        if cid_response.status_code != 200:
            url_cid = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas_or_smiles}/cids/JSON"
            cid_response = requests.get(url_cid, timeout=180)
            if cid_response.status_code != 200:
                return set()
        cid = cid_response.json()['IdentifierList']['CID'][0]
    except Exception as e:
        return set()

    # Step 2: Get GHS classification via classification endpoint
    url_classification = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    try:
        data_response = requests.get(url_classification, timeout=180)
        if data_response.status_code != 200:
            return set()
        data = data_response.json()
    except Exception as e:
        return set()

    # Step 3: Search for GHS hazard sections
    ghs_info = []
    for section in data['Record']['Section']:
        if section['TOCHeading'] == "Safety and Hazards":
            for sec in section['Section']:
                if "Hazards Identification" in sec['TOCHeading']:
                    for subsec in sec["Section"]:
                        if "GHS Classification" in subsec['TOCHeading']:
                            # Extract GHS classification information
                            for info in subsec["Information"]:
                                if info["Name"] == "GHS Hazard Statements":
                                    stmt = info.get('Value', {}).get('StringWithMarkup', [{}])[0].get('String', '')
                                    ghs_info.append(stmt.replace(":", " ").split()[0])
    ghs_cache[cas_or_smiles] = set(ghs_info)
    if not os.environ.get("mmorf_NO_CACHE", False):
        with open(CACHE_DIR + "/ghs_cache.pkl", "wb") as f:
            pickle.dump(ghs_cache, f)
    return set(ghs_info) - {"Not"}