import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

from transformers import AutoTokenizer, AutoModel


# ---------- DATA ----------

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fant ikke filen: {csv_path}")
    
    # semikolon-separert CSV
    df = pd.read_csv(csv_path, sep=";")
    
    required_cols = {"description", "ingredients", "skin_type"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Mangler kolonner i CSV: {missing}")
    return df


def build_text_column(df: pd.DataFrame) -> pd.Series:
    desc = df["description"].fillna("")
    ingr = df["ingredients"].fillna("")
    text = desc + " " + ingr
    return text


def build_skin_labels(df: pd.DataFrame):
    # "dry;normal;sensitive" -> ["dry", "normal", "sensitive"]
    skin_lists = df["skin_type"].fillna("").apply(
        lambda s: [x.strip() for x in s.split(";")] if s else []
    )
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(skin_lists)
    return Y, mlb


# ---------- TRANSFORMER-EMBEDDINGS ----------

def load_transformer(model_name: str = "distilbert-base-uncased"):
    print(f"📥 Laster transformer-modell: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def encode_texts(texts, tokenizer, model, batch_size: int = 8, max_length: int = 256):
    """
    Gjør tekst -> vektorer ved å bruke DistilBERT.
    Vi bruker [CLS]-token (første token) som setningsrepresentasjon.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = list(texts[i:i + batch_size])
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # flytt til CPU (det er der vi er uansett)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # shape: (batch_size, seq_len, hidden_dim)
        last_hidden = outputs.last_hidden_state
        # bruk CLS-token (posisjon 0)
        cls_embeddings = last_hidden[:, 0, :]  # (batch_size, hidden_dim)
        all_embeddings.append(cls_embeddings.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    return embeddings  # shape: (n_samples, hidden_dim)


# ---------- MODELL ----------

def train_skin_type_transformer(df: pd.DataFrame):
    print("🔹 Forbereder tekst og labels...")
    X_text = build_text_column(df)
    Y, mlb = build_skin_labels(df)

    print("🔹 Laster transformer...")
    tokenizer, model = load_transformer()

    print("🔹 Lager embeddings (dette kan ta litt tid)...")
    X_embeddings = encode_texts(X_text, tokenizer, model, batch_size=8)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_embeddings,
        Y,
        test_size=0.2,
        random_state=42
    )

    print("🔹 Trener logistisk regresjon oppå transformer-embeddings...")
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, n_jobs=-1)
    )
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)

    print("\n📊 Klassifikasjonsrapport (hudtype, transformer-basert):")
    print(
        classification_report(
            Y_test,
            Y_pred,
            target_names=mlb.classes_,
            zero_division=0
        )
    )

    # Noen eksempel-prediksjoner
    print("🔍 Eksempler på prediksjoner:")
    for i in range(min(5, X_test.shape[0])):
        true_labels = [mlb.classes_[j] for j in range(len(mlb.classes_)) if Y_test[i, j] == 1]
        pred_labels = [mlb.classes_[j] for j in range(len(mlb.classes_)) if Y_pred[i, j] == 1]
        print(f"\nSann hudtype: {true_labels}")
        print(f"Predikert hudtype: {pred_labels}")

    return clf, mlb


def main():
    csv_path = os.path.join("Data", "skincare_products.csv")
    print(f"📂 Leser data fra: {csv_path}")
    df = load_data(csv_path)

    print("\n📊 Hudtype-fordeling (rå streng):")
    print(df["skin_type"].value_counts().head())

    _ = train_skin_type_transformer(df)


if __name__ == "__main__":
    main()
