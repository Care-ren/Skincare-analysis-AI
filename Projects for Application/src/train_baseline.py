import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fant ikke filen: {csv_path}")
    
    # Viktig: semikolon-separert CSV
    df = pd.read_csv(csv_path, sep=";")
    
    required_cols = {"description", "ingredients", "category", "skin_type"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Mangler kolonner i CSV: {missing}")
    return df


def build_text_column(df: pd.DataFrame) -> pd.Series:
    """Slår sammen description + ingredients til én tekststreng per rad."""
    desc = df["description"].fillna("")
    ingr = df["ingredients"].fillna("")
    text = desc + " " + ingr
    return text


def train_baseline_model(df: pd.DataFrame):
    """Modell for å klassifisere produktkategori (cleanser, toner, osv.)."""
    X = build_text_column(df)
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ("clf", LogisticRegression(
            max_iter=200,
            n_jobs=-1
        ))
    ])

    print("🔹 Trener baseline-modell på kategori...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n📊 Klassifikasjonsrapport (kategori):")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("📉 Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model


def train_skin_type_model(df: pd.DataFrame):
    """Multi-label modell for å forutsi hudtype(r)."""
    X = build_text_column(df)

    # Gjør "dry;normal;sensitive" om til ["dry", "normal", "sensitive"]
    skin_lists = df["skin_type"].fillna("").apply(
        lambda s: [x.strip() for x in s.split(";")] if s else []
    )

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(skin_lists)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=42
        # ingen stratify her, multilabel er mer komplisert å stratifisere
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                max_iter=300,
                n_jobs=-1
            ))
        )
    ])

    print("\n🔹 Trener multi-label hudtype-modell...")
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    print("\n📊 Klassifikasjonsrapport (hudtype, multi-label):")
    print(
        classification_report(
            Y_test,
            Y_pred,
            target_names=mlb.classes_,
            zero_division=0
        )
    )

    # Vis noen eksempel-prediksjoner
    print("🔍 Eksempler på prediksjoner:")
    for i in range(min(5, X_test.shape[0])):
        text = X_test.iloc[i]
        true_labels = [mlb.classes_[j] for j in range(len(mlb.classes_)) if Y_test[i, j] == 1]
        pred_labels = [mlb.classes_[j] for j in range(len(mlb.classes_)) if Y_pred[i, j] == 1]
        print(f"\nProdukt-tekst: {text[:120]}...")
        print(f"Sann hudtype: {true_labels}")
        print(f"Predikert hudtype: {pred_labels}")

    return model, mlb


def main():
    csv_path = os.path.join("Data", "skincare_products.csv")
    print(f"📂 Leser data fra: {csv_path}")
    df = load_data(csv_path)

    print("\n🔎 Første titt på datasettet:")
    print(df.head())

    print("\n📊 Kategorifordeling:")
    print(df["category"].value_counts())

    print("\n📊 Hudtype-fordeling (rå streng):")
    print(df["skin_type"].value_counts().head())

    print("\n🔍 Kategorier med færre enn 2 produkter:")
    print(df['category'].value_counts()[df['category'].value_counts() < 2])

    # Kjør gjerne kategori-modellen fortsatt
    _ = train_baseline_model(df)

    # Ny: hudtype-modellen
    _, _ = train_skin_type_model(df)


if __name__ == "__main__":
    main()
