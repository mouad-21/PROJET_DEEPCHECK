"""
Entrainement du modele de detection de fake news.

Usage :
    python -m scripts.train_model            # entraine sur le dataset d'exemple
    python -m scripts.train_model --data path/to/dataset.csv  --text text --label label

Le dataset doit etre un CSV avec une colonne texte et une colonne label
(1 = fake, 0 = fiable). Affiche les metriques (F1, AUC...) et sauvegarde
le modele dans models/. La consommation energetique est mesuree (Green IT).
"""
import argparse
import json
import pandas as pd

from config import settings
from src.detect.model import train
from src.energy.carbon_tracker import EnergyTracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(settings.RAW_DIR / "combined_dataset.csv"),
                    help="CSV d'entrainement (defaut: vrai dataset combine EN+FR)")
    ap.add_argument("--text", default="text")
    ap.add_argument("--label", default="label")
    args = ap.parse_args()

    from pathlib import Path
    if not Path(args.data).exists():
        print(f"Dataset introuvable : {args.data}")
        print("-> Telecharge d'abord les vraies donnees : python -m scripts.download_dataset")
        print("   (ou entraine sur la demo synthetique : --data data/sample/sample_posts.csv)")
        return

    df = pd.read_csv(args.data)
    print(f"Dataset : {args.data}  ({len(df)} lignes)")

    with EnergyTracker("entrainement") as tracker:
        metrics = train(df, text_col=args.text, label_col=args.label)

    print("\n=== METRIQUES GLOBALES (jeu de test) ===")
    print(f"  split : {metrics['split_mode']}  | train={metrics['n_train']} test={metrics['n']}")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if k in metrics:
            print(f"  {k:10s}: {metrics[k]}")
    print(f"  matrice de confusion (lignes=reel, cols=predit) : {metrics['confusion_matrix']}")

    if "par_langue" in metrics:
        print("\n=== METRIQUES PAR LANGUE ===")
        for lang, m in metrics["par_langue"].items():
            auc = m.get("roc_auc", "n/a")
            print(f"  {lang.upper()} (n={m['n']:4d}) : F1={m['f1']}  acc={m['accuracy']}  AUC={auc}")

    print(f"\nModele sauvegarde : {metrics.get('model_path')}")
    print(f"Energie : {tracker.result}")

    # sauvegarde des metriques pour le dossier technique
    out = settings.MODELS_DIR / "metrics.json"
    metrics["energie"] = tracker.result
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Metriques : {out}")


if __name__ == "__main__":
    main()
