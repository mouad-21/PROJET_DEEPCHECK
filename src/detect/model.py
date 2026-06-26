"""
Facade de selection du modele de detection.

Un SEUL interrupteur dans config/settings.py -> MODEL_BACKEND decide quel
moteur est utilise, sans rien changer ailleurs :
    "tfidf"        -> src/detect/classifier.py            (baseline legere)
    "transformer"  -> src/detect/transformer_classifier.py (CamemBERT/DistilBERT)

Le reste du code (pipeline, dashboard, scripts) importe TOUJOURS depuis ici :
    from src.detect.model import train, predict_proba_fake, load_model

=> on peut comparer baseline vs transformer (F1 ET cout energetique Green IT)
   en changeant juste MODEL_BACKEND, sans toucher au pipeline.
"""
from config import settings

_BACKEND = getattr(settings, "MODEL_BACKEND", "tfidf").lower()

if _BACKEND == "transformer":
    from src.detect.transformer_classifier import (  # noqa: F401
        train, predict_proba_fake, load_model, _get_model,
    )
else:
    from src.detect.classifier import (  # noqa: F401
        train, predict_proba_fake, load_model, _get_model,
    )


def current_backend() -> str:
    """Renvoie le backend actif ('tfidf' ou 'transformer')."""
    return _BACKEND
