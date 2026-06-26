"""
Lancement du pipeline complet : collecte -> analyse -> stockage.

Usage :
    python -m scripts.run_pipeline --demo --n 30          # sans reseau (dataset exemple)
    python -m scripts.run_pipeline --terms info,urgent --lang fr --limit 50   # vrai Bluesky

En mode --demo, aucune connexion Bluesky n'est requise.
En mode reel, renseigne d'abord le .env (BLUESKY_HANDLE / BLUESKY_APP_PASSWORD).
Charge les variables du .env si python-dotenv est installe.
"""
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.collect import bluesky_collector as bc
from src.pipeline import run_batch
from src.storage import database as db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="mode sans reseau (dataset exemple)")
    ap.add_argument("--n", type=int, default=30, help="nb de posts en mode demo")
    ap.add_argument("--terms", default="info,urgent", help="mots-cles separes par des virgules")
    ap.add_argument("--lang", default="fr")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--no-store", action="store_true")
    args = ap.parse_args()

    if args.demo:
        print(f"Mode DEMO : {args.n} posts depuis le dataset d'exemple")
        posts = bc.collect_demo(args.n)
    else:
        terms = [t.strip() for t in args.terms.split(",") if t.strip()]
        print(f"Collecte Bluesky : termes={terms} lang={args.lang} limit={args.limit}")
        posts = bc.collect(search_terms=terms, lang=args.lang, limit=args.limit)
        path = bc.save_raw_csv(posts)
        print(f"Brut sauvegarde : {path}")

    res = run_batch(posts, store=not args.no_store)
    print(f"\n{res['n_analyses']} messages analyses.")
    print(f"Energie : {res['energie']}")

    # apercu
    doute = [r for r in res["resultats"] if r["credibility"]["niveau"] == "douteux"]
    print(f"Messages douteux detectes : {len(doute)} / {res['n_analyses']}")
    for r in doute[:3]:
        print(f"  [{r['credibility']['score']}/100] {r['text'][:60]}")

    if not args.no_store:
        print(f"\nTotal en base : {db.count_posts()} posts")


if __name__ == "__main__":
    main()
