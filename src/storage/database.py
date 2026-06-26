"""
Stockage des messages et de leurs analyses.

SQLite par defaut (zero config). Le schema est isole ici pour qu'on puisse
basculer vers PostgreSQL/MongoDB sans toucher au reste du code :
il suffit de reimplementer les memes fonctions (init_db, save_post,
save_analysis, fetch_*) avec un autre connecteur.

Tables :
 - posts    : message brut + metadonnees (langue, auteur, source...)
 - analyses : resultats (proba fake, score credibilite, emotion, explication...)
"""
import json
import sqlite3
from datetime import datetime, timezone
from config import settings


def get_conn():
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Cree les tables si elles n'existent pas."""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS posts (
            id           TEXT PRIMARY KEY,
            text_raw     TEXT NOT NULL,
            text_clean   TEXT,
            lang         TEXT,
            author       TEXT,
            source       TEXT,
            created_at   TEXT,
            collected_at TEXT
        );

        CREATE TABLE IF NOT EXISTS analyses (
            post_id           TEXT PRIMARY KEY,
            proba_fake        REAL,
            credibility_score INTEGER,
            niveau            TEXT,
            emotion_dominante TEXT,
            emotion_scores    TEXT,   -- JSON
            humour            INTEGER, -- 0/1
            explanation       TEXT,    -- JSON
            analyzed_at       TEXT,
            FOREIGN KEY(post_id) REFERENCES posts(id)
        );

        CREATE INDEX IF NOT EXISTS idx_cred ON analyses(credibility_score);
        CREATE INDEX IF NOT EXISTS idx_emo  ON analyses(emotion_dominante);
        """)


def _now():
    return datetime.now(timezone.utc).isoformat()


def save_post(post: dict):
    """post = {id, text_raw, text_clean, lang, author, source, created_at}"""
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO posts
            (id, text_raw, text_clean, lang, author, source, created_at, collected_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            post["id"], post["text_raw"], post.get("text_clean"),
            post.get("lang"), post.get("author"), post.get("source"),
            post.get("created_at"), _now(),
        ))


def save_analysis(post_id: str, analysis: dict):
    """analysis = sortie de pipeline.analyze_post() (champs ci-dessous)."""
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO analyses
            (post_id, proba_fake, credibility_score, niveau,
             emotion_dominante, emotion_scores, humour, explanation, analyzed_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            post_id,
            analysis["proba_fake"],
            analysis["credibility"]["score"],
            analysis["credibility"]["niveau"],
            analysis["emotion"]["dominante"],
            json.dumps(analysis["emotion"]["scores"], ensure_ascii=False),
            int(analysis["emotion"]["humour"]),
            json.dumps(analysis["explanation"], ensure_ascii=False),
            _now(),
        ))


def fetch_all_analyses():
    """Retourne la jointure posts + analyses (liste de dict)."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT p.id, p.text_clean, p.lang, p.author, p.source,
                   a.proba_fake, a.credibility_score, a.niveau,
                   a.emotion_dominante, a.humour, a.analyzed_at
            FROM posts p JOIN analyses a ON p.id = a.post_id
            ORDER BY a.analyzed_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def count_posts():
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]


if __name__ == "__main__":
    init_db()
    print(f"Base initialisee : {settings.DB_PATH}  ({count_posts()} posts)")
