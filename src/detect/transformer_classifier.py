"""
Detection de fake news -- VERSION TRANSFORMER (amelioration du TF-IDF).

Pourquoi un transformer plutot que TF-IDF :
 - TF-IDF ne regarde que le VOCABULAIRE (quels mots sont presents), pas le SENS
   ni l'ordre des mots. Un transformer pre-entraine (CamemBERT, DistilBERT
   multilingue...) comprend le contexte -> meilleur sur les cas ambigus et,
   surtout, exploite un pre-entrainement massif : il a besoin de BEAUCOUP MOINS
   de donnees labellisees, ce qui est exactement notre probleme en francais.
 - Modeles 100 % open source et gratuits (Hugging Face), telecharges une fois.

Choix du modele (configurable dans config/settings.py -> TRANSFORMER_MODEL_NAME) :
 - defaut : "distilbert-base-multilingual-cased" -> LEGER, gere FR + EN, bon
   compromis pour tourner sur un laptop (axe Green IT).
 - alternative FR : "camembert-base" (meilleur en francais pur, plus lourd).
 - alternative forte : "xlm-roberta-base" (multilingue, plus precis, plus lourd).

IMPORTANT (Green IT) : un transformer consomme nettement plus qu'un TF-IDF.
Le but du projet est justement de CHIFFRER cet arbitrage (gain de F1 vs cout CO2)
grace au tracker energetique deja en place dans scripts/train_model.py.

Ce module respecte la MEME interface que src/detect/classifier.py :
    train(df, ...)            -> dict de metriques (memes cles)
    predict_proba_fake(text)  -> float [0,1]
    load_model() / _get_model()
Plus une explicabilite par OCCLUSION (word_importances) qui remplace le
"poids des mots" du modele lineaire, sans dependance lourde (ni SHAP ni LIME).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score, roc_auc_score,
)

from config import settings
from src.preprocess.cleaner import clean_text

# torch / transformers sont importes PARESSEUSEMENT (dans les fonctions) pour
# que le reste du projet (TF-IDF) fonctionne meme si torch n'est pas installe.


# --------------------------------------------------------------------------- #
#  Utilitaires                                                                #
# --------------------------------------------------------------------------- #
def _pick_device():
    """Choisit le meilleur peripherique disponible : CUDA > Apple MPS > CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _prep_text(t: str) -> str:
    """Texte d'entree du transformer : on garde la casse (modele 'cased')."""
    return clean_text(t).text_clean


def _metrics_from_preds(y_true, y_pred, y_proba):
    """Memes cles que classifier._eval_block, pour rester compatible."""
    if len(y_true) == 0:
        return None
    out = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": int(len(y_true)),
    }
    if len(set(y_true)) == 2:
        out["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
    return out


# --------------------------------------------------------------------------- #
#  Dataset torch                                                              #
# --------------------------------------------------------------------------- #
def _make_dataset(texts, labels, tokenizer, max_len):
    import torch

    class _DS(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(
                list(texts), truncation=True, padding="max_length",
                max_length=max_len, return_tensors="pt",
            )
            self.labels = torch.tensor(list(labels), dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.labels[i]
            return item

    return _DS(texts, labels)


# --------------------------------------------------------------------------- #
#  Entrainement                                                               #
# --------------------------------------------------------------------------- #
def train(df: pd.DataFrame, text_col: str = "text", label_col: str = "label",
          lang_col: str = "lang", save: bool = True) -> dict:
    """Fine-tune le transformer et renvoie les memes metriques que le TF-IDF.

    Respecte la colonne 'split' (train/valid/test) si presente, sinon split
    stratifie 80/20. Metriques globales + PAR LANGUE + classification_report.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup,
    )

    df = df.copy()
    df["_X"] = df[text_col].astype(str).apply(_prep_text)
    df["_y"] = df[label_col].astype(int)

    # --- split (meme logique que classifier.py) ---
    if "split" in df.columns and (df["split"] == "test").any():
        tr = df[df["split"].isin(["train", "valid"])]
        te = df[df["split"] == "test"]
        split_mode = "officiel / fourni (split column)"
    else:
        tr, te = train_test_split(
            df, test_size=settings.TEST_SIZE,
            random_state=settings.RANDOM_STATE, stratify=df["_y"],
        )
        split_mode = f"aleatoire stratifie ({int(settings.TEST_SIZE*100)}% test)"

    # garde-fou Green IT : on peut plafonner le nb d'exemples d'entrainement
    cap = getattr(settings, "TRANSFORMER_MAX_TRAIN_SAMPLES", None)
    if cap and len(tr) > cap:
        tr = tr.sample(n=cap, random_state=settings.RANDOM_STATE)

    model_name = settings.TRANSFORMER_MODEL_NAME
    max_len = settings.TRANSFORMER_MAX_LEN
    device = _pick_device()
    print(f"[transformer] modele={model_name} | device={device} | "
          f"train={len(tr)} test={len(te)}")

    torch.manual_seed(settings.RANDOM_STATE)
    np.random.seed(settings.RANDOM_STATE)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_ds = _make_dataset(tr["_X"].tolist(), tr["_y"].tolist(), tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=settings.TRANSFORMER_BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.TRANSFORMER_LR)
    total_steps = len(train_loader) * settings.TRANSFORMER_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    # --- boucle d'entrainement (volontairement explicite, pedagogique) ---
    model.train()
    for epoch in range(settings.TRANSFORMER_EPOCHS):
        running = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += out.loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch+1}/{settings.TRANSFORMER_EPOCHS} "
                      f"step {step}/{len(train_loader)} loss={out.loss.item():.4f}")
        print(f"  -> epoch {epoch+1} loss moyen = {running/max(len(train_loader),1):.4f}")

    # --- evaluation ---
    def _predict(texts):
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), settings.TRANSFORMER_BATCH_SIZE):
                chunk = texts[i:i + settings.TRANSFORMER_BATCH_SIZE]
                enc = tokenizer(chunk, truncation=True, padding=True,
                                max_length=max_len, return_tensors="pt").to(device)
                logits = model(**enc).logits
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs.extend(p.tolist())
        return np.array(probs)

    y_true = te["_y"].values
    y_proba = _predict(te["_X"].tolist())
    y_pred = (y_proba >= 0.5).astype(int)

    glob = _metrics_from_preds(y_true, y_pred, y_proba)
    metrics = {"split_mode": split_mode, "backend": "transformer",
               "model_name": model_name, "n_train": int(len(tr)), **glob}

    # metriques PAR LANGUE
    if lang_col in te.columns:
        per_lang = {}
        for lang, sub in te.groupby(lang_col):
            yp = _predict(sub["_X"].tolist())
            blk = _metrics_from_preds(sub["_y"].values, (yp >= 0.5).astype(int), yp)
            if blk:
                per_lang[lang] = blk
        metrics["par_langue"] = per_lang

    metrics["report"] = classification_report(
        y_true, y_pred, target_names=["fiable", "fake"],
        output_dict=True, zero_division=0,
    )

    if save:
        out_dir = Path(settings.TRANSFORMER_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        (out_dir / "backend.json").write_text(
            json.dumps({"backend": "transformer", "model_name": model_name}, indent=2)
        )
        metrics["model_path"] = str(out_dir)
        print(f"[transformer] modele sauvegarde dans {out_dir}")

    return metrics


# --------------------------------------------------------------------------- #
#  Chargement + inference                                                     #
# --------------------------------------------------------------------------- #
_MODEL_CACHE = None  # (tokenizer, model, device)


def load_model():
    """Charge le transformer fine-tune ; erreur explicite s'il manque."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    out_dir = Path(settings.TRANSFORMER_DIR)
    if not out_dir.exists() or not any(out_dir.iterdir()):
        raise FileNotFoundError(
            f"Modele transformer introuvable dans {out_dir}. "
            f"Entraine-le d'abord : MODEL_BACKEND=transformer "
            f"python -m scripts.train_model"
        )
    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    model = AutoModelForSequenceClassification.from_pretrained(out_dir)
    device = _pick_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = load_model()
    return _MODEL_CACHE


def predict_proba_fake(text: str) -> float:
    """Probabilite que le message soit une fake news [0,1]."""
    import torch
    tokenizer, model, device = _get_model()
    enc = tokenizer(_prep_text(text), truncation=True, padding=True,
                    max_length=settings.TRANSFORMER_MAX_LEN, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        proba = torch.softmax(logits, dim=-1)[0, 1].item()
    return float(proba)


# --------------------------------------------------------------------------- #
#  Explicabilite par OCCLUSION (remplace le "poids des mots" du lineaire)     #
# --------------------------------------------------------------------------- #
def word_importances(text: str, top_k: int = 8):
    """Estime l'importance de chaque mot par OCCLUSION.

    Principe : on calcule la proba de fake sur le texte complet (baseline),
    puis on RETIRE chaque mot un par un et on regarde de combien la proba
    bouge. Si retirer le mot FAIT BAISSER la proba de fake, ce mot POUSSAIT
    vers "fake" (contribution positive), et inversement.

    Avantage : aucune dependance supplementaire (pas de SHAP/LIME), et c'est
    valable pour n'importe quel modele "boite noire".

    Renvoie (mots_vers_fake, mots_vers_fiable), memes formats que l'explainer
    TF-IDF : listes de (mot, contribution) triees.
    """
    base = predict_proba_fake(text)
    words = _prep_text(text).split()
    seen = set()
    contribs = []
    for i, w in enumerate(words):
        key = w.lower()
        if key in seen or len(key) <= 1:
            continue
        seen.add(key)
        reduced = " ".join(words[:i] + words[i + 1:])
        if not reduced.strip():
            continue
        p = predict_proba_fake(reduced)
        contrib = round(float(base - p), 4)   # >0 : le mot poussait vers fake
        contribs.append((w, contrib))

    contribs.sort(key=lambda x: x[1], reverse=True)
    vers_fake = [c for c in contribs if c[1] > 0][:top_k]
    vers_fiable = [c for c in reversed(contribs) if c[1] < 0][:top_k]
    return vers_fake, vers_fiable


if __name__ == "__main__":
    # mini demo (suppose le modele deja entraine)
    txt = "URGENT remede miracle soigne tout, les medecins sont furieux !!"
    print("proba_fake :", round(predict_proba_fake(txt), 3))
    vf, vfi = word_importances(txt)
    print("vers FAKE   :", vf)
    print("vers FIABLE :", vfi)
