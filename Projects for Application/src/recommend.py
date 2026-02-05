import os
import json
import re
import pandas as pd
from typing import List, Dict, Tuple

CONCERN_ALIASES = {
    "hyperpignemntation": "hyperpigmentation",
    "pigment": "hyperpigmentation",
    "dark spots": "hyperpigmentation",
    "spots": "hyperpigmentation",
    "pores": "clogged_pores",
    "blackheads": "clogged_pores",
    "redness": "redness_sensitivity",
    "sensitivity": "redness_sensitivity",
    "dry": "dryness",
    "dry skin": "dryness",
    "acne breakouts": "acne"
}

def normalize_concerns(concerns: List[str]) -> List[str]:
    out = []
    for c in concerns:
        c = c.strip().lower()
        if c in CONCERN_ALIASES:
            out.append(CONCERN_ALIASES[c])
        else:
            out.append(c)
    # unike
    seen = set()
    final = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            final.append(x)
    return final


def load_products(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fant ikke CSV: {csv_path}")

    # Du bruker semikolon-separert CSV
    df = pd.read_csv(csv_path, sep=";")

    required = {"name", "brand", "description", "ingredients", "category", "skin_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV mangler kolonner: {missing}")

    # Fyll NaN
    for col in ["name", "brand", "description", "ingredients", "category", "skin_type"]:
        df[col] = df[col].fillna("")

    return df


def load_rules(json_path: str) -> Dict:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Fant ikke rules-fil: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text: str) -> str:
    # lowercase + litt rydding (vi vil fortsatt beholde commas osv)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_keyword(haystack: str, needle: str) -> bool:
    """
    En enkel, robust match:
    - vi sjekker om "needle" finnes som substring
    - for veldig korte needle (<=3) hopper vi over for å unngå støy
    """
    needle_n = normalize(needle)
    if len(needle_n) <= 3:
        return False
    return needle_n in haystack


def explain_matches(haystack: str, keywords: List[str]) -> List[str]:
    found = []
    for kw in keywords:
        if contains_keyword(haystack, kw):
            found.append(kw)
    # fjern duplikater, behold rekkefølge
    seen = set()
    out = []
    for x in found:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_skin_types(s: str) -> List[str]:
    # "dry;normal;sensitive" -> ["dry","normal","sensitive"]
    s = s.strip()
    if not s:
        return []
    return [x.strip().lower() for x in s.split(";") if x.strip()]


def build_reco_profile(
    rules: Dict,
    concerns: List[str],
    avoid_toggles: List[str],
    skin_type: str
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returnerer:
    - recommended_keywords
    - avoid_keywords
    - notes (forklaringstekster)
    """
    concerns_rules = rules.get("concerns", {})
    avoid_rules = rules.get("avoid_toggles", {})
    skin_boost = rules.get("skin_type_boost", {})

    rec = []
    avoid = []
    notes = []

    for c in concerns:
        c = c.strip()
        if not c:
            continue
        if c not in concerns_rules:
            notes.append(f"(Ukjent concern '{c}' – hoppet over)")
            continue
        rec += concerns_rules[c].get("recommended", [])
        avoid += concerns_rules[c].get("avoid", [])
        note = concerns_rules[c].get("notes", "")
        if note:
            notes.append(note)

    for t in avoid_toggles:
        t = t.strip()
        if not t:
            continue
        avoid += avoid_rules.get(t, [])

    st = (skin_type or "").strip().lower()
    if st in skin_boost:
        rec += skin_boost[st]
        notes.append(f"Hudtype-boost aktiv: {st}")

    # rydd: unike
    def uniq(seq):
        seen = set()
        out = []
        for x in seq:
            x = normalize(x)
            if not x:
                continue
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return uniq(rec), uniq(avoid), notes


def score_product(
    row: pd.Series,
    rec_keywords: List[str],
    avoid_keywords: List[str],
    user_skin_type: str
) -> Tuple[int, Dict]:
    """
    Score logikk:
    +2 for hver recommended-ingredient funnet
    -3 for hver avoid-ingredient funnet
    +2 hvis produktet er merket med brukerens skin_type (fra din skin_type-kolonne)
    """
    ingredient_text = normalize(row["ingredients"] + " " + row["description"])
    product_skin_types = set(parse_skin_types(row["skin_type"]))

    matched_rec = explain_matches(ingredient_text, rec_keywords)
    matched_avoid = explain_matches(ingredient_text, avoid_keywords)

    score = 0
    score += 2 * len(matched_rec)
    score -= 3 * len(matched_avoid)

    ust = (user_skin_type or "").strip().lower()
    skin_match = (ust in product_skin_types) if ust else False
    if skin_match:
        score += 2

    details = {
        "matched_recommended": matched_rec,
        "matched_avoid": matched_avoid,
        "skin_match": skin_match
    }
    return score, details


def recommend_products(
    df: pd.DataFrame,
    rules: Dict,
    concerns: List[str],
    avoid_toggles: List[str],
    skin_type: str,
    top_k: int = 5
):
    rec_keywords, avoid_keywords, notes = build_reco_profile(
        rules=rules,
        concerns=concerns,
        avoid_toggles=avoid_toggles,
        skin_type=skin_type
    )

    results = []
    for _, row in df.iterrows():
        score, details = score_product(row, rec_keywords, avoid_keywords, skin_type)
        results.append((score, row, details))

    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:top_k]

    return rec_keywords, avoid_keywords, notes, top


def main():
    # ---- DISCLAIMER (for trygghet) ----
    print("⚠️ Disclaimer: Dette er ikke medisinsk rådgivning. Patch-test, og kontakt lege/hudlege ved sterke plager.\n")

    # ---- Paths ----
    csv_path = os.path.join("Data", "skincare_products.csv")
    rules_path = "ingredients_rules.json"

    df = load_products(csv_path)
    rules = load_rules(rules_path)

    # ---- “Skjema” i terminalen (enkelt å starte med) ----
    print("Velg hudtype: dry / oily / combination / normal / sensitive")
    skin_type = input("Hudtype: ").strip().lower()

    print("\nVelg concerns (skriv kommaseparert). Mulige:")
    print("acne, clogged_pores, dryness, redness_sensitivity, hyperpigmentation, texture")
    concerns_in = input("Concerns: ").strip().lower()
    concerns = normalize_concerns([c.strip() for c in concerns_in.split(",") if c.strip()])


    print("\nVelg 'avoid toggles' (skriv kommaseparert). Mulige:")
    print("fragrance, alcohol_denat, essential_oils")
    avoid_in = input("Avoid: ").strip().lower()
    avoid_toggles = [a.strip() for a in avoid_in.split(",") if a.strip()]

    rec_keywords, avoid_keywords, notes, top = recommend_products(
        df=df,
        rules=rules,
        concerns=concerns,
        avoid_toggles=avoid_toggles,
        skin_type=skin_type,
        top_k=5
    )

    print("\n✅ Ingredienser å se etter (basert på valgene dine):")
    print(", ".join(rec_keywords) if rec_keywords else "(ingen)")

    print("\n⛔ Ingredienser å være obs på / unngå (basert på valgene dine):")
    print(", ".join(avoid_keywords) if avoid_keywords else "(ingen)")

    if notes:
        print("\n📝 Notater:")
        for n in notes:
            print(f"- {n}")

    print("\n🏆 Topp anbefalte produkter (fra databasen din):")
    for rank, (score, row, details) in enumerate(top, start=1):
        title = f'{row["brand"]} — {row["name"]}'.strip(" —")
        cat = row["category"]
        st = row["skin_type"]

        print(f"\n#{rank}  SCORE={score}  [{cat}]  skin_type_in_db={st}")
        print(title)

        if details["skin_match"]:
            print("  + Hudtype-match i databasen")

        if details["matched_recommended"]:
            print("  + Treffer 'bra' ingredienser:", ", ".join(details["matched_recommended"][:8]))

        if details["matched_avoid"]:
            print("  - Treffer 'unngå' ingredienser:", ", ".join(details["matched_avoid"][:8]))

        # Kort visning av ingredienser
        ingr_preview = str(row["ingredients"])[:160]
        print(f"  INCI preview: {ingr_preview}...")

    print("\nFerdig ✅")


if __name__ == "__main__":
    main()
