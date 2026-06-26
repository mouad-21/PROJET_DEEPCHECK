# 🛰️ Thumalien — Détection de fake news sur Bluesky

MVP d'analyse automatisée des messages Bluesky : **détection de fake news**,
**score de crédibilité**, **analyse émotionnelle**, **explicabilité** (poids des
mots) et **suivi énergétique** (Green IT).

> Projet d'études — Mastère 1 Big Data & IA — SUP DE VINCI.
> Client fictif : **Thumalien**.

---

## Ce que fait l'outil

À partir d'un message (ou d'un lot collecté sur Bluesky), le pipeline produit :

| Sortie | Description |
|---|---|
| **Score de crédibilité (0-100)** | 100 = fiable, 0 = très douteux, avec code couleur |
| **Tonalité émotionnelle** | colère, peur, joie, tristesse, surprise, dégoût (+ humour/ironie) |
| **Explicabilité** | les mots qui poussent vers « fake » ou « fiable » |
| **Pénalités de surface** | sensationnalisme (MAJUSCULES, `!!!`), message trop court… |
| **Empreinte énergétique** | durée + CO₂ de chaque traitement (CodeCarbon) |

---

## Installation

```bash
# 1. Cloner / dézipper le projet, puis :
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt

# 2. (optionnel) identifiants Bluesky pour la collecte réelle
cp .env.example .env
#   puis éditer .env avec ton handle + App Password
```

## Démarrage rapide (sans compte Bluesky)

```bash
export PYTHONPATH=.          # Windows PowerShell : $env:PYTHONPATH="."

# a) télécharger les VRAIS datasets de fake news (LIAR EN + obsinfox/X-FACT FR)
python -m scripts.download_dataset

# b) entraîner le modèle de détection sur ces données réelles
python -m scripts.train_model

# c) lancer le pipeline complet en mode démo (collecte simulée)
python -m scripts.run_pipeline --demo --n 30

# d) ouvrir le dashboard
streamlit run dashboard/app.py
```

> `scripts.make_sample_data` reste disponible pour générer des messages
> d'exemple **en entrée** de démo (posts à analyser) — à ne pas confondre avec
> les données d'entraînement, qui sont réelles (étape a).

## Collecte réelle sur Bluesky

```bash
# après avoir renseigné le .env :
python -m scripts.run_pipeline --terms info,urgent,sante --lang fr --limit 50
```

## Avec Docker

```bash
docker compose up --build
# dashboard sur http://localhost:8501
```

---

## Architecture

```
thumalien/
├── config/settings.py          # configuration centrale (chemins, seuils, couleurs)
├── data/
│   ├── sample/                 # dataset d'exemple (démo)
│   ├── raw/                    # posts Bluesky bruts collectés
│   └── thumalien.db            # base SQLite
├── src/
│   ├── collect/                # collecte Bluesky (atproto) [Data Engineer]
│   ├── preprocess/             # nettoyage + langue + features  [Data Engineer]
│   ├── detect/                 # classifieur + score crédibilité [Data Scientist]
│   ├── emotion/                # analyse émotionnelle            [Analyste IA]
│   ├── explain/                # explicabilité (poids des mots)  [Analyste IA]
│   ├── energy/                 # suivi CodeCarbon                [Green IT]
│   ├── storage/                # base de données
│   └── pipeline.py             # orchestrateur
├── dashboard/app.py            # interface Streamlit
├── scripts/                    # make_sample_data, train_model, run_pipeline
├── tests/                      # tests unitaires (pytest)
└── docs/                       # doc technique + guide utilisateur
```

Le pipeline : **collecte → prétraitement → détection → crédibilité → émotion → explicabilité → stockage → dashboard**.

---

## Répartition des rôles (équipe)

- **Data Engineer** : `collect/`, `preprocess/`, `storage/` (API, ETL, base)
- **Data Scientist** : `detect/` (NLP, classification, score)
- **Analyste IA** : `emotion/`, `explain/`, `dashboard/`
- **Green IT** : `energy/`, qualité du code, rapport énergétique

---

## Tests

```bash
pytest -q                       # ou : python -m tests.test_pipeline
```

---

## 📊 Données réelles & performances (honnêteté)

Le modèle livré est entraîné sur de **vraies données de fake news**, pas du
synthétique. Trois sources publiques, fusionnées en binaire *fake / fiable* :

| Dataset | Langue | Taille utilisée | Source / licence |
|---|---|---|---|
| **LIAR** (Wang, 2017) | EN | 12 836 énoncés | PolitiFact — usage académique |
| **OBSINFOX** | FR | 100 articles | obs-info/obsinfox — CC BY-NC 4.0 |
| **X-FACT** (Gupta & Srikumar, 2021) | FR | 158 claims | utahnlp/x-fact — usage académique |

**Métriques réelles** (jeu de test, splits officiels pour LIAR) :

| | F1 | Accuracy | ROC-AUC | n test |
|---|---|---|---|---|
| 🇬🇧 **Anglais** (LIAR) | **0.56** | 0.62 | 0.67 | 1 283 |
| 🇫🇷 **Français** (obsinfox + X-FACT) | 0.63 | 0.52 | **0.46** | 52 |

**Lecture honnête :**
- L'anglais (F1 ≈ 0.56) est **conforme aux baselines publiées sur LIAR** : la
  détection de fake news sur texte court est un problème intrinsèquement
  difficile. Un F1 ≈ 1.0 serait au contraire le signe d'un artefact.
- Le français reste **faible** (AUC ≈ 0.46, le modèle ne discrimine pas) : avec
  seulement ~258 exemples FR hétérogènes, un TF-IDF classique n'a pas assez de
  matière. C'est la **rareté des données françaises** de fake news.
- **Piste prioritaire** : un **transformer multilingue** (CamemBERT / mBERT)
  exploite un pré-entraînement massif et nécessite bien moins de données
  labellisées → c'est la voie pour rendre le français exploitable. En attendant,
  les signaux *language-agnostic* (pénalités de sensationnalisme + émotion)
  servent de garde-fou pour le français.

> Détails complets dans `docs/technical_doc.md` (sections « Données » et
> « Limites »).

---

## Pour aller plus loin

Voir `docs/technical_doc.md` (dossier technique) et `docs/user_guide.md`
(guide utilisateur). Pistes : passage à CamemBERT/RoBERTa, lexique émotionnel
complet (FEEL/NRC), scalabilité Spark/Kafka.
