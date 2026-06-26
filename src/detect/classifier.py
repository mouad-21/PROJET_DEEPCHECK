"""
Detection de fake news.

Choix de modele pour le MVP : TF-IDF + Regression Logistique.
Pourquoi ce choix (a justifier dans le dossier) :
 - rapide a entrainer, peu gourmand (coherent avec l'axe Green IT),
 - INTERPRETABLE : on peut sortir le poids de chaque mot -> explicabilite native,
 - baseline solide pour comparer ensuite a un modele BERT/RoBERTa.

Le passage a un transformer (CamemBERT en FR, RoBERTa en EN) est documente
dans docs/technical_doc.md : memes interfaces train()/predict() a respecter.

Le modele renvoie une PROBABILITE de fake [0,1], convertie ensuite en
SCORE DE CREDIBILITE [0,100] par src/detect/credibility.py.
"""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score, roc_auc_score,
)

from config import settings
from src.preprocess.cleaner import clean_text


def build_pipeline() -> Pipeline:
    """Construit le pipeline TF-IDF -> Regression Logistique."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),       # uni + bigrammes
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            strip_accents=None,        # on garde les accents (FR)
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=4.0,
            class_weight="balanced",   # robuste si classes desequilibrees
            random_state=settings.RANDOM_STATE,
        )),
    ])


def _eval_block(pipe, X, y):
    """Calcule un bloc de metriques pour un jeu (X, y)."""
    if len(y) == 0:
        return None
    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)[:, 1]
    out = {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "n": int(len(y)),
    }
    # AUC seulement si les 2 classes sont presentes
    if len(set(y)) == 2:
        out["roc_auc"] = round(roc_auc_score(y, y_proba), 4)
    return out


def train(df: pd.DataFrame, text_col: str = "text", label_col: str = "label",
          lang_col: str = "lang", save: bool = True) -> dict:
    """Entraine le modele et renvoie les metriques (globales + par langue).

    df doit contenir text_col et label_col (label 1 = fake, 0 = fiable).
    - Si df contient une colonne 'split' (train/valid/test) : on la respecte.
    - Si df contient lang_col : metriques calculees aussi PAR LANGUE.
    """
    df = df.copy()
    df["_X"] = df[text_col].apply(lambda t: clean_text(t).text_norm)

    if "split" in df.columns and (df["split"] == "test").any():
        tr = df[df["split"].isin(["train", "valid"])]
        te = df[df["split"] == "test"]
        split_mode = "officiel / fourni (split column)"
    else:
        tr, te = train_test_split(
            df, test_size=settings.TEST_SIZE,
            random_state=settings.RANDOM_STATE, stratify=df[label_col],
        )
        split_mode = f"aleatoire stratifie ({int(settings.TEST_SIZE*100)}% test)"

    pipe = build_pipeline()
    pipe.fit(tr["_X"].tolist(), tr[label_col].astype(int).values)

    # metriques globales sur le test
    glob = _eval_block(pipe, te["_X"].tolist(), te[label_col].astype(int).values)

    metrics = {"split_mode": split_mode, "n_train": int(len(tr)), **glob}

    # metriques PAR LANGUE
    if lang_col in te.columns:
        per_lang = {}
        for lang, sub in te.groupby(lang_col):
            blk = _eval_block(pipe, sub["_X"].tolist(),
                              sub[label_col].astype(int).values)
            if blk:
                per_lang[lang] = blk
        metrics["par_langue"] = per_lang

    metrics["report"] = classification_report(
        te[label_col].astype(int).values, pipe.predict(te["_X"].tolist()),
        target_names=["fiable", "fake"], output_dict=True, zero_division=0,
    )

    if save:
        joblib.dump(pipe, settings.MODEL_PATH)
        metrics["model_path"] = str(settings.MODEL_PATH)

    return metrics


def load_model() -> Pipeline:
    """Charge le modele entraine ; leve une erreur explicite s'il manque."""
    if not settings.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modele introuvable : {settings.MODEL_PATH}. "
            f"Lance d'abord 'python -m scripts.train_model'."
        )
    return joblib.load(settings.MODEL_PATH)


_MODEL_CACHE = None


def _get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = load_model()
    return _MODEL_CACHE


def predict_proba_fake(text: str) -> float:
    """Probabilite que le message soit une fake news [0,1]."""
    model = _get_model()
    norm = clean_text(text).text_norm
    return float(model.predict_proba([norm])[0, 1])


if __name__ == "__main__":
    # mini auto-test si le dataset d'exemple existe
    csv = settings.SAMPLE_DIR / "sample_posts.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        m = train(df)
        print(f"F1={m['f1']} | AUC={m['roc_auc']} | acc={m['accuracy']}")
    else:
        print("Genere d'abord le dataset : python -m scripts.make_sample_data")
