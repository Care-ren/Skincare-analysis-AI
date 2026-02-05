import os, json, re
import pandas as pd
from typing import List, Dict, Tuple

CONCERN_ALIASES = {
    "hyperpignemntation": "hyperpigmentation",
    "pigment": "hyperpigmentation",
    "dark spots": "hyperpigmentation",
    "pores": "clogged_pores",
    "blackheads": "clogged_pores",
    "redness": "redness_sensitivity",
    "sensitivity": "redness_sensitivity",
    "dry": "dryness",
    "dry skin": "dryness"
}

def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_concerns(concerns: List[str]) -> List[str]:
    out = []
    for c in concerns:
        c = normalize(c)
        out.append(CONCERN_ALIASES.get(c, c))
    # unike
    seen=set(); final=[]
    for x in out:
        if x and x not in seen:
            seen.add(x); final.append(x)
    return final

def load_rules(json_path: str) -> Dict:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Fant ikke rules: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_products(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fant ikke CSV: {csv_path}")
    df = pd.read_csv(csv_path, sep=";")
    required = {"id","name","brand","description","ingredients","category","skin_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV mangler kolonner: {missing}")

    for col in required:
        df[col] = df[col].fillna("").astype(str)
    return df

def parse_skin_types(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    # kan være "dry;normal" eller '"dry;normal"'
    s = s.strip('"')
    return [x.strip().lower() for x in s.split(";") if x.strip()]

def contains_keyword(haystack: str, needle: str) -> bool:
    needle_n = normalize(needle)
    if len(needle_n) <= 3:
        return False
    return needle_n in haystack

def explain_matches(haystack: str, keywords: List[str]) -> List[str]:
    found=[]
    for kw in keywords:
        if contains_keyword(haystack, kw):
            found.append(kw)
    seen=set(); out=[]
    for x in found:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def build_reco_profile(rules: Dict, concerns: List[str], avoid_toggles: List[str], skin_type: str):
    concerns_rules = rules.get("concerns", {})
    avoid_rules = rules.get("avoid_toggles", {})
    skin_boost = rules.get("skin_type_boost", {})

    rec=[]; avoid=[]; notes=[]
    for c in concerns:
        if c not in concerns_rules:
            notes.append(f"(Ukjent concern '{c}' – hoppet over)")
            continue
        rec += concerns_rules[c].get("recommended", [])
        avoid += concerns_rules[c].get("avoid", [])
        note = concerns_rules[c].get("notes","")
        if note: notes.append(note)

    for t in avoid_toggles:
        avoid += avoid_rules.get(t, [])

    st = normalize(skin_type)
    if st in skin_boost:
        rec += skin_boost[st]
        notes.append(f"Hudtype-boost aktiv: {st}")

    def uniq(seq):
        seen=set(); out=[]
        for x in seq:
            x = normalize(x)
            if x and x not in seen:
                seen.add(x); out.append(x)
        return out

    return uniq(rec), uniq(avoid), notes

def score_product(row, rec_keywords, avoid_keywords, user_skin_type: str) -> Tuple[int, Dict]:
    text = normalize(row["ingredients"] + " " + row["description"] + " " + row["name"])
    product_skin = set(parse_skin_types(row["skin_type"]))
    matched_rec = explain_matches(text, rec_keywords)
    matched_avoid = explain_matches(text, avoid_keywords)

    score = 0
    score += 2 * len(matched_rec)
    score -= 3 * len(matched_avoid)

    ust = normalize(user_skin_type)
    skin_match = (ust in product_skin) if ust else False
    if skin_match:
        score += 2

    return score, {
        "matched_recommended": matched_rec,
        "matched_avoid": matched_avoid,
        "skin_match": skin_match
    }

def ranked_products(df, rules, concerns, avoid_toggles, skin_type):
    concerns = normalize_concerns(concerns)
    rec, avoid, notes = build_reco_profile(rules, concerns, avoid_toggles, skin_type)

    results=[]
    for _, row in df.iterrows():
        score, details = score_product(row, rec, avoid, skin_type)
        results.append((score, row, details))
    results.sort(key=lambda x: x[0], reverse=True)

    return rec, avoid, notes, results

# --- Rutine steg ---
ROUTINE_STEPS = [
    ("Oil cleanser", "oil_cleanser"),
    ("Foam cleanser", "foam_cleanser"),
    ("Toner", "toner"),
    ("Essence", "essence"),
    ("Serum", "serum"),
    ("Moisturizer", "moisturizer"),
    ("Sunscreen", "sunscreen"),
    ("Toner pads", "toner_pads"),
]

def routine_slot_match(slot_key: str, row) -> bool:
    cat = normalize(row["category"])
    text = normalize(row["name"] + " " + row["description"])

    # Oil cleanser / foam cleanser: ofte ligger alt i category=cleanser hos deg.
    if slot_key == "oil_cleanser":
        return (cat == "cleanser") and ("oil" in text or "cleansing oil" in text)
    if slot_key == "foam_cleanser":
        return (cat == "cleanser") and ("foam" in text or "cleansing foam" in text)
    if slot_key == "toner":
        return cat == "toner"
    if slot_key == "essence":
        # du har mange essences som category=serum → vi gjetter via ordet "essence"
        return ("essence" in text) and (cat in ["serum","toner","moisturizer","cleanser","mask","exfoliant"])
    if slot_key == "serum":
        # Serum skal ikke “stjele” essence
        return (cat == "serum") and ("essence" not in text)
    if slot_key == "moisturizer":
        return cat == "moisturizer"
    if slot_key == "sunscreen":
        return cat == "sunscreen"
    if slot_key == "toner_pads":
        return ("pad" in text or "pads" in text)
    return False

def build_routine(df, rules, concerns, avoid_toggles, skin_type) -> Dict:
    rec, avoid, notes, ranked = ranked_products(df, rules, concerns, avoid_toggles, skin_type)

    picked = {}
    used_ids=set()

    for label, slot in ROUTINE_STEPS:
        chosen = None
        for score, row, details in ranked:
            pid = str(row["id"])
            if pid in used_ids:
                continue
            if routine_slot_match(slot, row):
                chosen = (score, row.to_dict(), details)
                used_ids.add(pid)
                break
        picked[slot] = chosen

    return {
        "recommended_ingredients": rec,
        "avoid_ingredients": avoid,
        "notes": notes,
        "routine": picked
    }
