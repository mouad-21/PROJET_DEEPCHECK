"""
Generateur de donnees d'EXEMPLE (synthetiques).

>>> IMPORTANT <<<
Ces messages NE SONT PAS de vrais posts Bluesky. Ils sont generes
artificiellement pour pouvoir demontrer le pipeline de bout en bout
SANS dependre d'un acces API en direct.

Ils reproduisent neanmoins des marqueurs linguistiques reels :
 - cote "fake"  : sensationnalisme, urgence, MAJUSCULES, promesses miracles,
                  cadrage complotiste, absence de source.
 - cote "fiable": ton mesure, attribution/source, chiffres precis, nuances.

Pour la version finale, ce jeu de donnees doit etre remplace par :
 - un vrai dataset open source de fake news (voir docs/technical_doc.md),
 - et/ou des posts Bluesky reels collectes + annotes manuellement.
"""
import random
import pandas as pd
from config import settings

random.seed(settings.RANDOM_STATE)

# --- Briques pour generer du "fake" (FR) ---
FAKE_FR_AMORCES = [
    "URGENT !!! ", "ATTENTION : ", "Ils ne veulent pas que vous le sachiez : ",
    "SCANDALE : ", "INCROYABLE !! ", "On vous CACHE la verite : ",
    "PARTAGEZ avant suppression !! ", "BREAKING : ", "",
]
FAKE_FR_CORPS = [
    "un remede miracle soigne TOUTES les maladies en 24h, les medecins sont furieux",
    "le gouvernement prepare en secret une loi pour confisquer vos economies",
    "ce simple aliment fait fondre la graisse instantanement sans effort",
    "les vaccins contiennent des puces 5G, la preuve enfin reveleee",
    "une elite mondiale controle la meteo pour ruiner les agriculteurs",
    "boire ce melange chaque matin guerit le cancer, big pharma le cache",
    "ils vont couper internet la semaine prochaine, preparez-vous",
    "cette decouverte va changer le monde mais on l'etouffe deja",
]
FAKE_FR_FINS = [
    " !!!", " !! Reveillez-vous !", " (100% verifie)", " - partagez en masse !!",
    " la verite eclate ENFIN", "", " avant qu'il ne soit trop tard",
]

# --- Briques pour generer du "fiable" (FR) ---
REAL_FR_AMORCES = [
    "Selon l'INSEE, ", "D'apres une etude publiee dans Nature, ",
    "Le ministere a annonce que ", "Selon les chiffres officiels, ",
    "Une enquete de Le Monde indique que ", "L'OMS rappelle que ",
    "D'apres le rapport annuel, ", "",
]
REAL_FR_CORPS = [
    "le taux d'inflation s'est etabli a 2,1 % sur un an au mois dernier",
    "les temperatures moyennes ont augmente de 1,4 degre depuis 1900",
    "la frequentation des transports a progresse de 3 % au premier trimestre",
    "l'efficacite du traitement reste a confirmer par des essais complementaires",
    "le nombre de logements neufs a baisse de 5 % par rapport a l'an dernier",
    "les donnees suggerent une correlation, sans prouver de lien de cause a effet",
    "la consommation d'energie du secteur a diminue de 2 % cette annee",
    "des resultats preliminaires montrent une tendance, encore a verifier",
]
REAL_FR_FINS = [
    ".", ", precise le communique.", " selon les auteurs.",
    ", sous reserve de confirmation.", ".", " indiquent les chercheurs.",
]

# --- Anglais (versions simplifiees) ---
FAKE_EN = [
    "SHOCKING!!! This one weird trick cures everything doctors HATE it share now!!",
    "BREAKING: secret elite controls the weather to destroy farmers, wake up!!",
    "They are HIDING the truth: this miracle drink melts fat overnight no effort",
    "URGENT they will shut down the internet next week prepare yourself NOW!!",
    "100% PROVEN: vaccines contain microchips, finally the evidence revealed share!!",
]
REAL_EN = [
    "According to the WHO, the new guidelines recommend annual screening for adults.",
    "A study in Nature reports a 1.4 C rise in average temperatures since 1900.",
    "Official figures show inflation settled at 2.1 percent over the past year.",
    "Researchers note a correlation but caution it does not prove causation.",
    "The ministry announced a 3 percent increase in public transport ridership.",
]


def _gen_fake_fr() -> str:
    return (random.choice(FAKE_FR_AMORCES)
            + random.choice(FAKE_FR_CORPS)
            + random.choice(FAKE_FR_FINS))


def _gen_real_fr() -> str:
    return (random.choice(REAL_FR_AMORCES)
            + random.choice(REAL_FR_CORPS)
            + random.choice(REAL_FR_FINS))


def build_sample_dataset(n_per_class: int = 220) -> pd.DataFrame:
    """Construit un DataFrame equilibre fake/fiable, FR + EN.

    Colonnes : id, text, lang, label  (label 1 = fake, 0 = fiable)
    """
    rows = []
    pid = 0
    for _ in range(n_per_class):
        rows.append({"text": _gen_fake_fr(), "lang": "fr", "label": 1})
        rows.append({"text": _gen_real_fr(), "lang": "fr", "label": 0})
    # une portion en anglais pour montrer le multilingue
    for _ in range(n_per_class // 5):
        rows.append({"text": random.choice(FAKE_EN), "lang": "en", "label": 1})
        rows.append({"text": random.choice(REAL_EN), "lang": "en", "label": 0})

    random.shuffle(rows)
    for r in rows:
        r["id"] = f"sample_{pid:05d}"
        pid += 1
    df = pd.DataFrame(rows)[["id", "text", "lang", "label"]]
    return df


if __name__ == "__main__":
    df = build_sample_dataset()
    out = settings.SAMPLE_DIR / "sample_posts.csv"
    df.to_csv(out, index=False)
    print(f"Dataset d'exemple genere : {out}")
    print(f"  {len(df)} messages | fake={df.label.sum()} | fiable={(df.label==0).sum()}")
    print(df.head(6).to_string(index=False))
