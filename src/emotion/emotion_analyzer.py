"""
Analyse emotionnelle des messages.

Approche MVP : lexique d'emotions (FR + EN) -> robuste, rapide, sans
telechargement de modele (coherent Green IT). Categories couvertes :
colere, peur, joie, tristesse, surprise, degout, + heuristique humour/ironie.

Le passage a un modele type EmotionBERT / j-hartmann/emotion-english... est
documente dans docs/technical_doc.md (meme interface analyze_emotions()).

Le lexique ci-dessous est volontairement compact. Pour la version finale,
l'enrichir avec une ressource ouverte (ex. NRC Emotion Lexicon, FEEL pour le FR).
[A COMPLETER : brancher un lexique complet ou un modele EmotionBERT]
"""
import re
from collections import defaultdict
from dataclasses import dataclass, asdict

# Lexique compact {emotion: [mots-cles FR/EN]}
EMOTION_LEXICON = {
    "colere": [
        "colere", "rage", "scandale", "honte", "inadmissible", "furieux",
        "revolte", "indigne", "injuste", "trahison", "menteur", "corrompu",
        "angry", "rage", "furious", "outrage", "scandal", "disgrace", "liar",
    ],
    "peur": [
        "peur", "danger", "menace", "alerte", "urgent", "catastrophe", "crise",
        "panique", "terrible", "effrayant", "risque", "mortel", "attention",
        "fear", "danger", "threat", "alert", "urgent", "warning", "deadly",
        "catastrophe", "crisis", "panic", "scary",
    ],
    "joie": [
        "joie", "super", "genial", "incroyable", "fantastique", "merci",
        "heureux", "victoire", "bravo", "magnifique", "formidable", "excellent",
        "joy", "great", "amazing", "awesome", "fantastic", "happy", "win",
        "love", "wonderful", "excellent",
    ],
    "tristesse": [
        "triste", "tristesse", "pleure", "deuil", "perte", "douleur", "malheur",
        "desespoir", "souffrance", "drame", "tragique",
        "sad", "grief", "loss", "pain", "cry", "mourning", "tragic", "despair",
    ],
    "surprise": [
        "incroyable", "inattendu", "surprise", "choc", "stupefiant", "etonnant",
        "jamais vu", "revelation",
        "shocking", "unexpected", "surprise", "stunning", "revealed", "unbelievable",
    ],
    "degout": [
        "degout", "ecoeurant", "repugnant", "sale", "pourri", "infect",
        "disgust", "disgusting", "gross", "rotten", "nasty",
    ],
}

# Heuristique humour / ironie (signaux faibles)
HUMOUR_MARQUEURS = ["mdr", "lol", "ptdr", "haha", "ironie", "blague", "humour",
                    "😂", "🤣", "lmao", "rofl", "joke", "funny", "kidding"]

WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class EmotionResult:
    dominante: str               # emotion principale (ou "neutre")
    scores: dict                 # {emotion: score normalise 0-1}
    humour: bool                 # presence de marqueurs d'humour/ironie
    n_mots_emotionnels: int

    def as_dict(self):
        return asdict(self)


def analyze_emotions(text: str) -> EmotionResult:
    """Detecte les emotions presentes dans un message (approche lexicale)."""
    low = text.lower()
    tokens = set(WORD_RE.findall(low))

    raw = defaultdict(int)
    for emo, mots in EMOTION_LEXICON.items():
        for m in mots:
            if " " in m:                      # expression -> recherche directe
                if m in low:
                    raw[emo] += 1
            elif m in tokens:                 # mot simple -> match exact
                raw[emo] += 1

    total = sum(raw.values())
    if total == 0:
        scores = {e: 0.0 for e in EMOTION_LEXICON}
        dominante = "neutre"
    else:
        scores = {e: round(raw.get(e, 0) / total, 3) for e in EMOTION_LEXICON}
        dominante = max(scores, key=scores.get)

    humour = any(h in low for h in HUMOUR_MARQUEURS)

    return EmotionResult(
        dominante=dominante,
        scores=scores,
        humour=humour,
        n_mots_emotionnels=total,
    )


if __name__ == "__main__":
    for t in ["URGENT danger imminent, panique generale !!",
              "Bravo super victoire, magnifique journee !",
              "mdr c'est une blague ?",
              "Reunion prevue mardi a 14h."]:
        r = analyze_emotions(t)
        print(f"{r.dominante:9s} humour={r.humour} -> {t}")
