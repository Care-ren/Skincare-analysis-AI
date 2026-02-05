import os
import sys

import pandas as pd

TESTS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(TESTS_DIR, "..", "src"))
sys.path.append(SRC_DIR)

from recommend_core import (
    normalize_concerns,
    build_reco_profile,
    score_product,
    build_routine,
)


def test_normalize_concerns_alias_and_dedupe():
    concerns = ["Dry", "dry skin", "acne", "Acne"]
    assert normalize_concerns(concerns) == ["dryness", "acne"]


def test_build_reco_profile_rules_and_skin_boost():
    rules = {
        "concerns": {
            "acne": {"recommended": ["salicylic acid"], "avoid": ["coconut oil"], "notes": "Hold porene åpne"},
        },
        "avoid_toggles": {"fragrance": ["fragrance"]},
        "skin_type_boost": {"dry": ["ceramide"]},
    }

    rec, avoid, notes = build_reco_profile(
        rules=rules,
        concerns=["acne"],
        avoid_toggles=["fragrance"],
        skin_type="dry",
    )

    assert "salicylic acid" in rec
    assert "ceramide" in rec
    assert "fragrance" in avoid
    assert "Hold porene åpne" in notes


def test_score_product_matches_and_skin_type():
    row = pd.Series(
        {
            "ingredients": "Water, Niacinamide, Fragrance",
            "description": "Soothing serum",
            "name": "Bright Serum",
            "skin_type": "dry;normal",
        }
    )
    score, details = score_product(row, ["niacinamide"], ["fragrance"], "dry")

    assert score == 1
    assert details["skin_match"] is True
    assert details["matched_recommended"] == ["niacinamide"]
    assert details["matched_avoid"] == ["fragrance"]


def test_build_routine_picks_expected_slots():
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "name": "Gentle Cleansing Oil",
                "brand": "TestBrand",
                "description": "Oil cleanser for makeup",
                "ingredients": "oil",
                "category": "cleanser",
                "skin_type": "dry;normal",
            },
            {
                "id": "2",
                "name": "Vitamin C Serum",
                "brand": "TestBrand",
                "description": "Brightening serum",
                "ingredients": "ascorbic acid",
                "category": "serum",
                "skin_type": "dry;normal",
            },
        ]
    )

    rules = {"concerns": {}, "avoid_toggles": {}, "skin_type_boost": {}}
    data = build_routine(df, rules, concerns=[], avoid_toggles=[], skin_type="dry")

    assert data["routine"]["oil_cleanser"] is not None
    assert data["routine"]["serum"] is not None
