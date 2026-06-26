"""
Score de credibilite [0,100].

On part de la probabilite de fake donnee par le modele, puis on applique
des ajustements TRANSPARENTS bases sur des signaux de surface connus pour
correler avec la desinformation. Chaque ajustement est trace et restitue
a l'utilisateur (exigence d'explicabilite du cahier des charges).

score_base   = (1 - proba_fake) * 100
penalites    = sensationnalisme (MAJUSCULES, '!' en rafale), message tres court...
score_final  = clamp(score_base - penalites, 0, 100)

Le detail (proba modele + chaque penalite) est renvoye pour affichage.
"""
from dataclasses import dataclass, field, asdict
from config import settings
from src.preprocess.cleaner import clean_text, CleanResult


@dataclass
class CredibilityResult:
    score: int                       # 0-100 (100 = tres fiable)
    niveau: str                      # "fiable" / "a verifier" / "douteux"
    couleur: str                     # code couleur dashboard
    proba_fake: float                # sortie brute du modele [0,1]
    score_base: float                # avant penalites
    penalites: dict = field(default_factory=dict)
    signaux: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


def _niveau_et_couleur(score: int):
    if score < settings.CRED_SEUIL_DOUTEUX:
        return "douteux", settings.COULEUR_ALERTE
    if score < settings.CRED_SEUIL_VIGILANCE:
        return "a verifier", settings.COULEUR_VIGILANCE
    return "fiable", settings.COULEUR_OK


def compute_credibility(text: str, proba_fake: float,
                        clean: CleanResult | None = None) -> CredibilityResult:
    """Calcule le score de credibilite a partir de la proba du modele
    et des signaux de surface du message."""
    if clean is None:
        clean = clean_text(text)

    score_base = (1.0 - proba_fake) * 100.0
    penalites = {}

    # 1. Sensationnalisme : beaucoup de MAJUSCULES sur un texte assez long
    if clean.upper_ratio > 0.30 and clean.n_chars > 15:
        penalites["majuscules_excessives"] = round(min(15, clean.upper_ratio * 25), 1)

    # 2. Ponctuation en rafale (!!! ou ???)
    if clean.n_excl >= 3:
        penalites["exclamations_en_rafale"] = min(10, (clean.n_excl - 2) * 3)
    if clean.n_quest >= 3:
        penalites["interrogations_en_rafale"] = min(6, (clean.n_quest - 2) * 2)

    # 3. Message tres court = peu de contexte verifiable
    if clean.n_chars < 25:
        penalites["message_trop_court"] = 5

    total_penalite = sum(penalites.values())
    score_final = max(0, min(100, round(score_base - total_penalite)))
    niveau, couleur = _niveau_et_couleur(score_final)

    return CredibilityResult(
        score=score_final,
        niveau=niveau,
        couleur=couleur,
        proba_fake=round(proba_fake, 4),
        score_base=round(score_base, 1),
        penalites=penalites,
        signaux={
            "upper_ratio": clean.upper_ratio,
            "n_excl": clean.n_excl,
            "n_quest": clean.n_quest,
            "n_chars": clean.n_chars,
            "lang": clean.lang,
        },
    )
