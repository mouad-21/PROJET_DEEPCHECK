"""
Orchestrateur du pipeline Thumalien.

Enchaine : nettoyage -> classification fake news -> score credibilite
        -> analyse emotionnelle -> explicabilite.

Deux entrees :
 - analyze_post(text)   : analyse un message unique -> dict complet
 - run_batch(posts)     : analyse une liste de posts + stockage en base,
                          le tout mesure par CodeCarbon (Green IT).
"""
from src.preprocess.cleaner import clean_text
from src.detect.model import predict_proba_fake
from src.detect.credibility import compute_credibility
from src.emotion.emotion_analyzer import analyze_emotions
from src.explain.explainer import explain
from src.energy.carbon_tracker import EnergyTracker
from src.storage import database as db


def analyze_post(text: str, with_explanation: bool = True) -> dict:
    """Analyse complete d'un message. Renvoie un dict serialisable."""
    clean = clean_text(text)
    proba = predict_proba_fake(text)
    cred = compute_credibility(text, proba, clean=clean)
    emo = analyze_emotions(text)
    expl = explain(text).as_dict() if with_explanation else {}

    return {
        "text": clean.text_clean,
        "lang": clean.lang,
        "proba_fake": round(proba, 4),
        "credibility": cred.as_dict(),
        "emotion": emo.as_dict(),
        "explanation": expl,
    }


def run_batch(posts: list[dict], store: bool = True) -> dict:
    """Analyse une liste de posts normalises (sortie du collecteur).

    Chaque post doit avoir au minimum : id, text_raw.
    Mesure l'energie consommee et stocke posts + analyses en base.
    """
    if store:
        db.init_db()

    results = []
    with EnergyTracker("inference_batch") as tracker:
        for p in posts:
            analysis = analyze_post(p["text_raw"])
            results.append({"id": p["id"], **analysis})
            if store:
                db.save_post(p)
                db.save_analysis(p["id"], analysis)

    return {
        "n_analyses": len(results),
        "energie": tracker.result,
        "resultats": results,
    }


if __name__ == "__main__":
    r = analyze_post("URGENT !! ce remede miracle soigne TOUT, les medecins furieux !!")
    print(f"Credibilite : {r['credibility']['score']}/100 ({r['credibility']['niveau']})")
    print(f"Emotion     : {r['emotion']['dominante']}")
    print(f"Top mots fake: {r['explanation']['mots_vers_fake'][:3]}")
