"""
Tests unitaires Thumalien.

Lancement :
    pytest -q
ou sans pytest :
    python -m tests.test_pipeline

Couvre : pretraitement, emotion, credibilite, classifieur, pipeline complet.
Le classifieur exige un modele entraine (lance scripts.train_model avant,
ou les tests qui en dependent seront ignores).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from src.preprocess.cleaner import clean_text
from src.emotion.emotion_analyzer import analyze_emotions
from src.detect.credibility import compute_credibility


# ---------- Pretraitement ----------
def test_clean_supprime_url_et_mention():
    r = clean_text("Coucou @bob regarde https://x.com/abc !!")
    assert "http" not in r.text_clean
    assert "@bob" not in r.text_clean
    assert r.n_excl == 2

def test_detection_majuscules():
    r = clean_text("CECI EST UN MESSAGE EN MAJUSCULES")
    assert r.upper_ratio > 0.9

def test_langue_francais():
    r = clean_text("Bonjour, ceci est une phrase en francais correcte.")
    assert r.lang in ("fr", "unknown")


# ---------- Emotion ----------
def test_emotion_peur():
    r = analyze_emotions("danger imminent, alerte, panique generale")
    assert r.dominante == "peur"

def test_emotion_neutre():
    r = analyze_emotions("Reunion prevue mardi a 14h en salle 3.")
    assert r.dominante == "neutre"

def test_humour_detecte():
    r = analyze_emotions("mdr c'est une blague serieux ?")
    assert r.humour is True


# ---------- Credibilite ----------
def test_credibilite_penalise_sensationnalisme():
    # proba fake faible mais texte tres sensationnaliste -> penalites
    r = compute_credibility("ATTENTION DANGER URGENT PARTAGEZ VITE !!!", proba_fake=0.2)
    assert r.penalites  # au moins une penalite
    assert 0 <= r.score <= 100

def test_credibilite_bornes():
    r_haut = compute_credibility("texte neutre et mesure", proba_fake=0.0)
    r_bas = compute_credibility("texte", proba_fake=1.0)
    assert r_haut.score <= 100 and r_bas.score >= 0
    assert r_haut.niveau == "fiable"


# ---------- Classifieur + pipeline (si modele dispo) ----------
def test_pipeline_si_modele():
    if not settings.MODEL_PATH.exists():
        print("[skip] modele absent : lance scripts.train_model")
        return
    from src.pipeline import analyze_post
    fake = analyze_post("URGENT remede miracle soigne tout, partagez !!")
    real = analyze_post("Selon l'INSEE, l'inflation s'etablit a 2,1 %.")
    # le message douteux doit avoir un score plus bas que le fiable
    assert fake["credibility"]["score"] < real["credibility"]["score"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
            ok += 1
        except AssertionError as e:
            print(f"  FAIL {fn.__name__} : {e}")
        except Exception as e:
            print(f"  ERR  {fn.__name__} : {e}")
    print(f"\n{ok}/{len(fns)} tests passes.")
