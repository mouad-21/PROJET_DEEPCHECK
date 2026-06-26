"""
Telechargement + conversion de VRAIS datasets de fake news (EN + FR).

1) LIAR (Wang, 2017) -- ANGLAIS  [github.com/thiagorainmaker77/liar_dataset]
   12 836 courts enonces politiques (PolitiFact). Benchmark de reference.
   6 labels -> binaire :  FAKE = pants-fire/false/barely-true ;
                          FIABLE = half-true/mostly-true/true

2) OBSINFOX -- FRANCAIS  [github.com/obs-info/obsinfox]  (CC BY-NC 4.0)
   100 titres d'articles francais (17 sources peu fiables), annotes par 8
   experts. On agrege par vote majoritaire (anti-fuite). Colonne 'Fake News'.

3) X-FACT (Gupta & Srikumar, 2021) -- sous-ensemble FRANCAIS
   [github.com/utahnlp/x-fact]  198 claims francais fact-checkees.
   Labels -> binaire : FAKE = false/partly true ; FIABLE = true/mostly true.

Sortie : data/raw/combined_dataset.csv  (id, text, lang, label, split)
LIAR garde ses splits officiels ; le francais (obsinfox + x-fact) recoit un
split aleatoire 80/20 stratifie, sources melangees.

Usage : python -m scripts.download_dataset
"""
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from config import settings

# ---------- LIAR (EN) ----------
LIAR_BASE = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master"
LIAR_FILES = {"train": "train.tsv", "valid": "valid.tsv", "test": "test.tsv"}
LIAR_COLS = ["id", "label", "statement", "subject", "speaker", "job", "state",
             "party", "c_barely", "c_false", "c_halftrue", "c_mostlytrue",
             "c_pantsfire", "context"]
FAKE_LABELS = {"pants-fire", "false", "barely-true"}
REAL_LABELS = {"half-true", "mostly-true", "true"}

# ---------- OBSINFOX (FR) ----------
OBSINFOX_URL = "https://raw.githubusercontent.com/obs-info/obsinfox/main/obsinfox.csv"

# ---------- X-FACT (FR) ----------
XFACT_URL = "https://raw.githubusercontent.com/utahnlp/x-fact/main/data/x-fact-including-en/zeroshot.tsv"
XFACT_FAKE = {"false", "partly true/misleading", "mostly false"}
XFACT_REAL = {"true", "mostly true"}


def _cached_download(url: str, local):
    """Telecharge si absent du cache local (evite de re-DL les gros fichiers)."""
    if not local.exists() or local.stat().st_size == 0:
        urllib.request.urlretrieve(url, local)
    return local


def _load_liar() -> pd.DataFrame:
    parts = []
    for split, fname in LIAR_FILES.items():
        local = _cached_download(f"{LIAR_BASE}/{fname}", settings.RAW_DIR / f"liar_{split}.tsv")
        print(f"  [LIAR/EN]    {split:5s} ({fname})")
        df = pd.read_csv(local, sep="\t", header=None, names=LIAR_COLS, quoting=3)
        df["split"] = split
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["label"] = df["label"].map(
        lambda l: 1 if l in FAKE_LABELS else (0 if l in REAL_LABELS else None))
    df = df.dropna(subset=["label", "statement"]).copy()
    df["label"] = df["label"].astype(int)
    df = df.rename(columns={"statement": "text"})
    df["lang"] = "en"
    df["id"] = "liar_" + df["id"].astype(str)
    return df[["id", "text", "lang", "label", "split"]]   # garde le split officiel


def _load_obsinfox() -> pd.DataFrame:
    local = _cached_download(OBSINFOX_URL, settings.RAW_DIR / "obsinfox.csv")
    print(f"  [OBSINFOX/FR]      (100 articles x 8 annotateurs)")
    raw = pd.read_csv(local)
    # 100 articles x 8 annotateurs -> vote majoritaire (anti-fuite)
    agg = (raw.groupby("URL")
              .agg(text=("Title", "first"), vote=("Fake News", "mean"))
              .reset_index())
    agg["label"] = (agg["vote"] >= 0.5).astype(int)
    df = agg.dropna(subset=["text"]).reset_index(drop=True)
    df["id"] = "obsinfox_" + df.index.astype(str)
    return df[["id", "text", "label"]]


def _load_xfact_fr() -> pd.DataFrame:
    local = _cached_download(XFACT_URL, settings.RAW_DIR / "xfact_zeroshot.tsv")
    print(f"  [X-FACT/FR]        (claims fact-checkees)")
    raw = pd.read_csv(local, sep="\t", on_bad_lines="skip", quoting=3)
    fr = raw[raw["language"] == "fr"].copy()
    fr["label"] = fr["label"].map(
        lambda l: 1 if l in XFACT_FAKE else (0 if l in XFACT_REAL else None))
    fr = fr.dropna(subset=["label", "claim"]).copy()
    fr["label"] = fr["label"].astype(int)
    fr = fr.rename(columns={"claim": "text"}).reset_index(drop=True)
    fr["id"] = "xfact_" + fr.index.astype(str)
    return fr[["id", "text", "label"]]


def build_combined_csv() -> str:
    """Telecharge LIAR + obsinfox + x-fact FR, fusionne, ecrit combined_dataset.csv."""
    print("Telechargement des datasets reels (EN + FR)...")
    liar = _load_liar()                       # a deja une colonne split

    # --- pool francais (2 sources) ---
    fr = pd.concat([_load_obsinfox(), _load_xfact_fr()], ignore_index=True)
    fr = fr.drop_duplicates(subset=["text"]).reset_index(drop=True)
    fr["lang"] = "fr"
    tr, te = train_test_split(fr, test_size=0.2, random_state=settings.RANDOM_STATE,
                              stratify=fr["label"])
    fr.loc[tr.index, "split"] = "train"
    fr.loc[te.index, "split"] = "test"

    df = pd.concat([liar, fr], ignore_index=True)[["id", "text", "lang", "label", "split"]]
    path = settings.RAW_DIR / "combined_dataset.csv"
    df.to_csv(path, index=False)

    print(f"\nDataset combine pret : {path}")
    print(f"  TOTAL {len(df)} | fake={int(df.label.sum())} | fiable={int((df.label==0).sum())}")
    for lang in ("en", "fr"):
        sub = df[df.lang == lang]
        te_n = len(sub[sub.split == "test"])
        print(f"  {lang.upper()}: {len(sub):5d} enonces (test={te_n}, fake={int(sub.label.sum())})")
    return str(path)


if __name__ == "__main__":
    build_combined_csv()
