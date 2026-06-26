"""
Explicabilite : pourquoi ce message est-il juge douteux (ou fiable) ?

Le modele etant lineaire (Regression Logistique sur features TF-IDF),
la contribution de chaque terme present dans le message =
    valeur_tfidf(terme) * coefficient(terme)

Une contribution positive pousse vers "fake", negative vers "fiable".
On restitue le top des mots dans chaque sens -> "poids des mots" demande
dans le cahier des charges (innovation : explicabilite IA).

NB : si on passe a un transformer, remplacer cette logique par SHAP ou
LIME (documente dans docs/technical_doc.md), en gardant l'interface explain().
"""
from dataclasses import dataclass, asdict
import numpy as np

from config import settings
from src.preprocess.cleaner import clean_text

# Mots vides FR+EN filtres de l'explication (bruit peu informatif).
# On les retire seulement de l'AFFICHAGE : ils restent dans le modele.
STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "a", "au", "aux",
    "en", "ce", "cette", "ces", "se", "sa", "son", "ses", "que", "qui", "ne",
    "pas", "plus", "pour", "par", "sur", "dans", "avec", "est", "sont", "ont",
    "il", "elle", "ils", "elles", "on", "vous", "nous", "je", "tu", "sera",
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "are", "for",
    "it", "they", "this", "that", "with", "as", "at", "by", "be", "will",
}


@dataclass
class ExplainResult:
    mots_vers_fake: list      # [(mot, contribution), ...] tries desc
    mots_vers_fiable: list    # [(mot, contribution), ...] tries asc
    intercept: float

    def as_dict(self):
        return asdict(self)


def explain(text: str, top_k: int = 8) -> ExplainResult:
    """Renvoie les mots qui poussent le plus vers 'fake' et vers 'fiable'.

    - Backend TF-IDF (lineaire)  : contribution = tfidf(mot) x coef(mot).
    - Backend transformer        : contribution estimee par OCCLUSION
      (on retire chaque mot et on mesure l'impact sur la proba de fake).
    L'interface et le format de sortie sont IDENTIQUES dans les deux cas.
    """
    if getattr(settings, "MODEL_BACKEND", "tfidf").lower() == "transformer":
        from src.detect.transformer_classifier import word_importances
        vers_fake, vers_fiable = word_importances(text, top_k=top_k)
        return ExplainResult(
            mots_vers_fake=vers_fake,
            mots_vers_fiable=vers_fiable,
            intercept=0.0,   # pas d'intercept lineaire pour un transformer
        )

    # --- backend TF-IDF (modele lineaire) ---
    from src.detect.classifier import _get_model
    model = _get_model()
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    norm = clean_text(text).text_norm
    X = vectorizer.transform([norm])          # 1 x vocab (sparse)
    coefs = clf.coef_[0]                       # vocab
    feature_names = vectorizer.get_feature_names_out()

    X_coo = X.tocoo()
    contributions = []
    for idx, val in zip(X_coo.col, X_coo.data):
        term = feature_names[idx]
        # on ignore les mots vides et les bigrammes 100% mots vides
        mots = term.split()
        if all(m in STOPWORDS for m in mots):
            continue
        contrib = float(val * coefs[idx])
        contributions.append((term, round(contrib, 4)))

    # tri par contribution
    contributions.sort(key=lambda x: x[1], reverse=True)
    vers_fake = [c for c in contributions if c[1] > 0][:top_k]
    vers_fiable = [c for c in reversed(contributions) if c[1] < 0][:top_k]

    return ExplainResult(
        mots_vers_fake=vers_fake,
        mots_vers_fiable=vers_fiable,
        intercept=round(float(clf.intercept_[0]), 4),
    )


if __name__ == "__main__":
    r = explain("URGENT remede miracle soigne tout, les medecins sont furieux !!")
    print("Vers FAKE   :", r.mots_vers_fake)
    print("Vers FIABLE :", r.mots_vers_fiable)
