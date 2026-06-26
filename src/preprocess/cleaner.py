"""
Pretraitement NLP des messages.

Pipeline :
 1. nettoyage (URLs, mentions @, hashtags, espaces, caracteres de controle)
 2. detection de langue (langdetect)
 3. normalisation legere pour le modele (minuscule, ponctuation reduite)

On garde DEUX versions du texte :
 - text_clean : lisible par un humain (pour le dashboard / l'explicabilite)
 - text_norm  : normalise pour le modele (features TF-IDF)

Remarque : on ne supprime PAS les majuscules ni la ponctuation excessive trop tot,
car ce sont des signaux utiles de sensationnalisme. On calcule des features
dediees (ratio de majuscules, nombre de "!") avant de normaliser.
"""
import re
import unicodedata
from dataclasses import dataclass, asdict

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0          # rend la detection deterministe
    _HAS_LANGDETECT = True
except Exception:                      # pragma: no cover
    _HAS_LANGDETECT = False

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w.\-]+")
HASHTAG_RE = re.compile(r"#(\w+)")
MULTISPACE_RE = re.compile(r"\s+")
NON_PRINT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass
class CleanResult:
    """Resultat structure du pretraitement d'un message."""
    text_raw: str
    text_clean: str
    text_norm: str
    lang: str
    n_chars: int
    n_excl: int          # nombre de '!'
    n_quest: int         # nombre de '?'
    upper_ratio: float   # part de lettres en MAJUSCULE (signal de sensationnalisme)

    def as_dict(self):
        return asdict(self)


def detect_lang(text: str) -> str:
    """Detecte la langue ; renvoie 'unknown' si echec."""
    if not _HAS_LANGDETECT or len(text.strip()) < 3:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def _upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    uppers = sum(1 for c in letters if c.isupper())
    return round(uppers / len(letters), 4)


def clean_text(text: str) -> CleanResult:
    """Nettoie un message et calcule ses features de surface."""
    raw = text if isinstance(text, str) else str(text)

    # features de surface AVANT nettoyage (on veut compter les vrais "!" du post)
    n_excl = raw.count("!")
    n_quest = raw.count("?")
    upper = _upper_ratio(raw)

    # nettoyage
    t = NON_PRINT_RE.sub(" ", raw)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(r"\1", t)          # garde le mot du hashtag, retire le #
    t = MULTISPACE_RE.sub(" ", t).strip()
    text_clean = t

    lang = detect_lang(text_clean)

    # normalisation pour le modele : minuscule + accents conserves (utile en FR)
    norm = text_clean.lower()
    norm = MULTISPACE_RE.sub(" ", norm).strip()
    text_norm = norm

    return CleanResult(
        text_raw=raw,
        text_clean=text_clean,
        text_norm=text_norm,
        lang=lang,
        n_chars=len(text_clean),
        n_excl=n_excl,
        n_quest=n_quest,
        upper_ratio=upper,
    )


def clean_batch(texts):
    """Nettoie une liste de messages -> liste de CleanResult."""
    return [clean_text(t) for t in texts]


if __name__ == "__main__":
    demo = [
        "URGENT !!! Ce REMEDE miracle soigne TOUT en 24h !! https://faux.site @compte",
        "Selon l'INSEE, l'inflation s'etablit a 2,1 % sur un an.",
        "BREAKING: they are HIDING the truth #wakeup",
    ]
    for r in clean_batch(demo):
        print(f"[{r.lang}] upper={r.upper_ratio} excl={r.n_excl} -> {r.text_clean}")
