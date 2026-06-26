"""
Collecte de messages Bluesky via l'API officielle (protocole AT / atproto).

Ce module fonctionne des que les identifiants sont fournis dans le .env :
    BLUESKY_HANDLE=ton-handle.bsky.social
    BLUESKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx   (App Password, PAS le mot de passe)

Genere un App Password ici : Bluesky > Settings > App Passwords.

Le collecteur :
 - se connecte,
 - recherche des posts par mots-cles et/ou langue,
 - normalise chaque post au format attendu par le pipeline,
 - peut sauvegarder le brut en CSV (data/raw) et/ou en base.

NB : l'environnement de build n'a pas acces a internet vers Bluesky ; ce code
est donc a executer sur ta machine. Un mode "demo" (sans reseau) est fourni
plus bas pour tester le pipeline a partir du dataset d'exemple.
"""
import os
import csv
from datetime import datetime, timezone

from config import settings
from src.preprocess.cleaner import clean_text

try:
    from atproto import Client
    _HAS_ATPROTO = True
except Exception:                      # pragma: no cover
    _HAS_ATPROTO = False


def _normalize_post(record, author_handle: str, uri: str) -> dict:
    """Met un post Bluesky au format interne commun."""
    text = getattr(record, "text", "") or ""
    created = getattr(record, "created_at", None)
    cr = clean_text(text)
    return {
        "id": uri,                      # l'URI AT sert d'identifiant unique
        "text_raw": text,
        "text_clean": cr.text_clean,
        "lang": cr.lang,
        "author": author_handle,
        "source": "bluesky",
        "created_at": created,
    }


def collect(search_terms=None, limit: int = None, lang: str = "fr") -> list[dict]:
    """Collecte des posts Bluesky par recherche de mots-cles.

    Renvoie une liste de dict normalises. Leve une erreur claire si les
    identifiants manquent ou si atproto n'est pas installe.
    """
    if not _HAS_ATPROTO:
        raise ImportError("atproto non installe : pip install atproto")

    handle = os.getenv("BLUESKY_HANDLE")
    app_pwd = os.getenv("BLUESKY_APP_PASSWORD")
    if not handle or not app_pwd:
        raise RuntimeError(
            "Identifiants manquants. Renseigne BLUESKY_HANDLE et "
            "BLUESKY_APP_PASSWORD dans le fichier .env (voir .env.example)."
        )

    search_terms = search_terms or settings.COLLECT_SEARCH_TERMS
    limit = limit or settings.COLLECT_MAX_POSTS

    client = Client()
    client.login(handle, app_pwd)

    collected = []
    per_term = max(1, limit // len(search_terms))
    for term in search_terms:
        # API de recherche de posts (app.bsky.feed.searchPosts)
        res = client.app.bsky.feed.search_posts({
            "q": term, "limit": min(per_term, 100), "lang": lang,
        })
        for post in res.posts:
            collected.append(_normalize_post(
                post.record, post.author.handle, post.uri
            ))
        if len(collected) >= limit:
            break

    return collected[:limit]


def save_raw_csv(posts: list[dict], filename: str = None) -> str:
    """Sauvegarde la collecte brute en CSV (data/raw)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = filename or f"bluesky_raw_{ts}.csv"
    path = settings.RAW_DIR / filename
    if not posts:
        return str(path)
    cols = ["id", "text_raw", "text_clean", "lang", "author", "source", "created_at"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in posts:
            w.writerow({c: p.get(c, "") for c in cols})
    return str(path)


def collect_demo(n: int = 30) -> list[dict]:
    """Mode DEMO sans reseau : tire des messages du dataset d'exemple
    et les met au format collecte. Utile pour tester le pipeline complet."""
    import pandas as pd
    csv_path = settings.SAMPLE_DIR / "sample_posts.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            "Genere d'abord le dataset : python -m scripts.make_sample_data"
        )
    df = pd.read_csv(csv_path).sample(n=min(n, 400), random_state=settings.RANDOM_STATE)
    posts = []
    for _, row in df.iterrows():
        cr = clean_text(row["text"])
        posts.append({
            "id": f"demo_{row['id']}",
            "text_raw": row["text"],
            "text_clean": cr.text_clean,
            "lang": cr.lang,
            "author": "demo.bsky.social",
            "source": "demo",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    return posts


if __name__ == "__main__":
    posts = collect_demo(5)
    for p in posts:
        print(f"[{p['lang']}] {p['text_clean'][:70]}")
