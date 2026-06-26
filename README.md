# 🛰️ Thumalien — DeepCheck
### Détection de fake news sur Bluesky · pipeline NLP complet sous contrainte Green IT

> **Projet Mastère 1 Big Data & IA — SUP DE VINCI**
> Équipe : Mouad Hocine · Inès Amiche · Ania Lachi · Yasmine Berrichi

---

## 🔗 Livrables

| Livrable | Accès |
|---|---|
| **MVP fonctionnel — pipeline complet (code)** | 👉 https://github.com/mouad-21/PROJET_DEEPCHECK |
| **Documentation technique** | dans ce README + fichier `DOCUMENTATION_TECHNIQUE.pdf` |
| **Guide utilisateur** | dans ce README + fichier `GUIDE_UTILISATEUR.pdf` |

---

## 📑 Sommaire

1. [Présentation](#-présentation)
2. [Documentation technique](#-documentation-technique)
3. [Guide utilisateur](#-guide-utilisateur)

---

## 🎯 Présentation

Thumalien analyse automatiquement les messages publiés sur le réseau social
**Bluesky** pour repérer la désinformation. Pour chaque message, l'outil fournit un
**verdict de crédibilité**, la **tonalité émotionnelle**, une **explication** des
mots qui rendent le contenu suspect, et une **mesure de consommation énergétique**
(démarche Green IT).

### Fonctionnalités
- **Collecte** via l'API officielle Bluesky (protocole AT).
- **Prétraitement** NLP : nettoyage, détection de langue (FR/EN), indices de sensationnalisme.
- **Détection** avec deux moteurs interchangeables : baseline **TF-IDF** et transformer **CamemBERT / DistilBERT**.
- **Score de crédibilité** transparent (0–100).
- **Analyse émotionnelle** bilingue et **explicabilité** (mots vers fake / fiable).
- **Suivi énergétique** (CodeCarbon) et **tableau de bord** Streamlit (3 onglets).

### Résultats (jeu de test réel)
| Critère | TF-IDF (baseline) | CamemBERT (transformer) |
|---|---|---|
| F1 global | 0,56 | 0,56 |
| F1 anglais | 0,56 | 0,55 |
| AUC français | 0,46 | 0,35 |
| Durée d'entraînement | ≈ 11 s | ≈ 738 s (~12 min) |

**Lecture honnête** : sur des données à 98 % anglaises, CamemBERT n'apporte pas de
gain pour ~70× plus de calcul. Le projet **chiffre** cet arbitrage performance / énergie.

---

## 📘 Documentation technique
---

### 1. Vue d'ensemble

Thumalien est un pipeline NLP modulaire. Chaque étape est isolée dans un module
dédié, ce qui permet de remplacer une brique sans toucher au reste — propriété qui
a notamment permis d'ajouter un transformer derrière la même interface que la
baseline.

```
Bluesky API → collect → preprocess → detect → score de crédibilité
                                          │
            ┌──────────────┬──────────────┼──────────────┐
         emotion        explain        storage         energy
            └──────────────┴──────────────┴──────────────┘
                                ▼
                          dashboard (Streamlit)
```

L'orchestration est assurée par `src/pipeline.py` (analyse d'un message ou d'un lot).

---

### 2. Modules

| Module | Fichier(s) | Responsabilité |
|---|---|---|
| Configuration | `config/settings.py` | chemins, seuils, **choix du moteur** (`MODEL_BACKEND`), hyperparamètres |
| Collecte | `src/collect/bluesky_collector.py` | requêtes API Bluesky (protocole AT), filtrage langue/mots-clés |
| Prétraitement | `src/preprocess/cleaner.py` | nettoyage, détection de langue, indices de surface |
| Détection (baseline) | `src/detect/classifier.py` | TF-IDF + régression logistique |
| Détection (transformer) | `src/detect/transformer_classifier.py` | fine-tuning CamemBERT/DistilBERT, occlusion |
| Façade modèle | `src/detect/model.py` | sélectionne le moteur selon `MODEL_BACKEND` |
| Émotion | `src/emotion/` | tonalité dominante (lexique bilingue) |
| Explicabilité | `src/explain/explainer.py` | mots vers « fake » / « fiable » |
| Énergie | `src/energy/carbon_tracker.py` | durée, énergie, CO₂ (CodeCarbon + estimation) |
| Stockage | `src/storage/` | persistance SQLite (posts, analyses) |
| Orchestration | `src/pipeline.py` | analyse unitaire / par lot |
| Interface | `dashboard/app.py` | tableau de bord Streamlit (3 onglets) |

---

### 3. Le pipeline étape par étape

1. **Collecte** — Les messages sont récupérés via l'API officielle Bluesky
   (bibliothèque `atproto`). L'authentification utilise un identifiant + App
   Password lus dans `.env`. Chaque post devient un enregistrement structuré
   (URI, texte brut, texte nettoyé, langue, auteur, date).
2. **Prétraitement** — Suppression des URLs/mentions/caractères de contrôle ;
   détection de langue ; calcul des **indices de surface** (ratio de majuscules,
   nombre de `!` et `?`, longueur) **avant** normalisation, car ce sont des
   signaux de sensationnalisme. Deux versions du texte sont conservées : lisible
   (affichage/explicabilité) et normalisée (modèle).
3. **Détection** — Le moteur actif (TF-IDF ou transformer) renvoie une
   probabilité de fake `p ∈ [0,1]`.
4. **Score de crédibilité** — `score = (1 − p) × 100 − pénalités`, borné [0,100]
   (pénalités de sensationnalisme, message trop court). Seuils : `<40` douteux
   (rouge), `40–65` à vérifier (orange), `>65` fiable (vert).
5. **Émotion** — Tonalité dominante via lexique FR/EN (peur, colère, joie,
   tristesse, surprise, dégoût ; heuristique humour/ironie).
6. **Explicabilité** — Mots qui poussent vers « fake » ou « fiable ».
7. **Stockage** — Écriture dans SQLite (tables `posts`, `analyses`).
8. **Énergie** — Mesure encadrant l'entraînement et les lots, écrite dans
   `energy_logs/emissions.csv`.
9. **Dashboard** — Restitution interactive.

---

### 4. Modèles de détection

#### 4.1 Baseline TF-IDF (`classifier.py`)
Vectorisation TF-IDF (unigrammes + bigrammes) + régression logistique à classes
équilibrées. Avantages : léger, rapide, **nativement interprétable** (les
coefficients donnent le poids de chaque mot). Interface :
`train(df)`, `predict_proba_fake(text)`, `load_model()`.

#### 4.2 Transformer (`transformer_classifier.py`)
Fine-tuning d'un modèle pré-entraîné Hugging Face (par défaut
`distilbert-base-multilingual-cased`, ou `camembert-base`, ou `xlm-roberta-base`).
Détails : longueur de séquence 128, 2 epochs, learning rate 2e-5, device
auto-détecté (CUDA › Apple MPS › CPU). Le modèle fine-tuné est sauvegardé dans
`models/transformer_clf/`. **Même interface** que la baseline.

#### 4.3 Façade (`model.py`)
Un commutateur unique sélectionne le moteur :
```python
MODEL_BACKEND = "tfidf"        # défaut
MODEL_BACKEND = "transformer"  # CamemBERT / DistilBERT
```
surchargeable par variable d'environnement. Le reste du code importe toujours
`from src.detect.model import train, predict_proba_fake, load_model`, ce qui permet
de comparer les deux moteurs sans rien réécrire.

---

### 5. Données

| Dataset | Langue | Taille | Source |
|---|---|---|---|
| LIAR (Wang, 2017) | EN | 12 836 | PolitiFact — usage académique |
| OBSINFOX | FR | 100 articles | obs-info/obsinfox — CC BY-NC 4.0 |
| X-FACT (Gupta & Srikumar, 2021) | FR | 158 | utahnlp/x-fact — académique |

Harmonisation en cible binaire fake / fiable. Pour LIAR, les 6 niveaux sont
projetés en binaire et les **splits officiels** conservés. Pour OBSINFOX, vote
majoritaire des 8 annotateurs **avant** découpage (pas de fuite de données).
Jeu combiné : 13 094 exemples (5 816 fake, 7 278 fiables ; 12 836 EN, 258 FR).

---

### 6. Explicabilité

- **TF-IDF** : contribution d'un mot = `tfidf(mot) × coefficient(mot)`.
- **Transformer** : par **occlusion** — on retire chaque mot et on mesure la
  variation de la probabilité de fake. Aucune dépendance lourde (ni SHAP ni LIME),
  format de sortie identique à la baseline (`mots_vers_fake`, `mots_vers_fiable`).

---

### 7. Suivi énergétique (Green IT)

`EnergyTracker` (context manager / décorateur) encadre l'entraînement et les lots.
Quand **CodeCarbon** dispose de capteurs, il fournit l'énergie et le CO₂. Sur une
machine sans capteur (Mac Apple Silicon), il bascule en mode dégradé ; le tracker
**estime** alors l'énergie à partir de la durée et d'une puissance CPU moyenne
(≈ 20 W), puis le CO₂ via l'intensité carbone du réseau (≈ 50 gCO₂/kWh). Chaque
exécution écrit une ligne dans `energy_logs/emissions.csv`
(`timestamp, task, duration, energy_consumed, emissions, mode`).

> Limite assumée : sur Mac le CO₂ est **estimé**, pas mesuré. La donnée fiable est
> la **durée**.

#### Comparaison mesurée
| | TF-IDF | CamemBERT |
|---|---|---|
| Durée d'entraînement | ≈ 11 s | ≈ 738 s |
| Énergie estimée | ≈ 6,4 × 10⁻⁵ kWh | ≈ 4,1 × 10⁻³ kWh |

---

### 8. Configuration (`settings.py`)

| Clé | Rôle |
|---|---|
| `MODEL_BACKEND` | `tfidf` ou `transformer` |
| `TRANSFORMER_MODEL_NAME` | modèle HF (`camembert-base`, …) |
| `TRANSFORMER_MAX_LEN` / `EPOCHS` / `BATCH_SIZE` / `LR` | hyperparamètres |
| `TRANSFORMER_MAX_TRAIN_SAMPLES` | plafond d'exemples (démo rapide) |
| `RANDOM_STATE` / `TEST_SIZE` | reproductibilité / split |
| `ENERGY_*` | projet, pays, dossier de logs énergie |

---

### 9. Stockage, tests, reproductibilité

- **SQLite** par défaut (tables `posts`, `analyses`), remplaçable par PostgreSQL/MongoDB.
- **Reproductibilité** : `random_state` fixé partout.
- **Tests** : `pytest` (`tests/`).
- **Conteneurisation** : `Dockerfile` + `docker-compose.yml`.
- **Secrets** : isolés dans `.env` (hors dépôt).

---

### 10. Métriques détaillées

**TF-IDF** — Global : F1 0,56 · Acc 0,62 · AUC 0,66 · n=1335.
EN : F1 0,56 · AUC 0,67. FR : F1 0,63 · AUC 0,46.

**CamemBERT** — Global : F1 0,56 · Acc 0,61 · AUC 0,65 · n=1335.
EN : F1 0,55 · AUC 0,66. FR : F1 0,59 · AUC 0,35.

---

### 11. Limites techniques

Données françaises rares (~258), tâche difficile sur texte court (état de l'art
LIAR ≈ 0,56), transformer non convergé (2 epochs CPU/MPS), CO₂ estimé sur Mac,
pas de temps réel ni de montée en charge à ce stade.

---

## 📗 Guide utilisateur
Ce guide explique comment **installer**, **configurer** et **utiliser** l'outil,
pas à pas. Aucune connaissance approfondie en programmation n'est requise, mais
quelques commandes de terminal sont nécessaires.

---

### 1. Prérequis

- **Python 3.12** (recommandé). Python 3.9 est trop ancien pour certaines dépendances.
- Le gestionnaire **[uv](https://github.com/astral-sh/uv)** (recommandé) ou `pip`.
- Un compte **Bluesky** + un **App Password** (pour la collecte uniquement).
- ~1 Go d'espace disque (le modèle CamemBERT pèse ~450 Mo).

---

### 2. Installation pas à pas

```bash
# Aller dans le dossier du projet (celui qui contient dashboard/, src/, scripts/)
cd chemin/vers/thumalien

# Créer l'environnement avec Python 3.12 (uv fournit un Python autonome, fiable)
uv venv --python 3.12
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# Installer les dépendances de base
uv pip install -r requirements.txt

# (Optionnel) Dépendances de la version transformer (CamemBERT / DistilBERT)
uv pip install -r requirements-transformer.txt
```

> 💡 **macOS** : si vous voyez une erreur `pyexpat ... Symbol not found` avec un
> Python installé via Homebrew, utilisez **uv** comme ci-dessus : il fournit un
> Python autonome qui contourne ce problème.

---

### 3. Configuration des identifiants

La collecte Bluesky nécessite des identifiants, à placer dans un fichier `.env`
(jamais partagé / jamais poussé sur GitHub) :

```bash
cp .env.example .env
```

Puis ouvrez `.env` et renseignez :
```
BLUESKY_HANDLE=votre_identifiant.bsky.social
BLUESKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
```

> L'analyse de messages et le dashboard fonctionnent **sans** identifiants ;
> ceux-ci ne servent qu'à la collecte en direct depuis Bluesky.

---

### 4. Préparer les données et entraîner

```bash
export PYTHONPATH=.                 # Windows PowerShell : $env:PYTHONPATH="."

# Télécharger et combiner les jeux de données réels (LIAR + OBSINFOX + X-FACT)
python -m scripts.download_dataset

# Entraîner le modèle par défaut (TF-IDF — rapide, ~11 s)
python -m scripts.train_model
```

#### Utiliser le transformer CamemBERT
```bash
TRANSFORMER_MODEL_NAME=camembert-base MODEL_BACKEND=transformer python -m scripts.train_model
```
> ⏱️ L'entraînement d'un transformer est long sur ordinateur portable (~12 min).
> Pour une démo rapide, ouvrez `config/settings.py` et mettez
> `TRANSFORMER_MAX_TRAIN_SAMPLES = 2000`.

#### Choisir le modèle
| `MODEL_BACKEND` | Modèle | Quand l'utiliser |
|---|---|---|
| `tfidf` (défaut) | TF-IDF | rapide, sobre, démonstration |
| `transformer` | CamemBERT / DistilBERT | tester la compréhension du contexte |

---

### 5. Lancer le tableau de bord

```bash
streamlit run dashboard/app.py
# ou avec le transformer :
MODEL_BACKEND=transformer streamlit run dashboard/app.py
```

Le dashboard s'ouvre sur `http://localhost:8501`. Pour l'arrêter : `Ctrl + C`.

---

### 6. Utiliser l'interface (3 onglets)

#### 🔎 Analyse en direct
Collez un message dans la zone de texte. L'outil affiche :
- **Verdict** : `FIABLE` ou `DOUTEUX` ;
- **Langue détectée** et **probabilité de fake** (en %) ;
- **Émotion dominante** (ou « neutre ») ;
- 🔴 **Mots qui poussent vers FAKE** / 🟢 **vers FIABLE** (un tiret « — » si aucun) ;
- **Détail technique (JSON complet)** dépliable.

*Exemple douteux :* « URGENT !! ce remède miracle soigne TOUT, les médecins furieux !! »
→ verdict DOUTEUX, mots « urgent / miracle / furieux » signalés.
*Exemple neutre :* « Le conseil municipal se réunira mardi pour voter le budget. »
→ verdict FIABLE, émotion neutre, aucun mot signalé.

#### 📊 Vue d'ensemble
Statistiques cumulées des messages analysés et stockés en base.

#### 🌱 Green IT
Nombre d'exécutions tracées, durée, énergie et CO₂ (estimé hors capteur).
Une mesure apparaît après chaque entraînement ou analyse par lot.

---

### 7. Dépannage (erreurs fréquentes)

| Message | Cause | Solution |
|---|---|---|
| `command not found: streamlit` | environnement non activé | `source .venv/bin/activate` |
| `No module named 'scripts'` | mauvais dossier ou `PYTHONPATH` manquant | se placer dans le dossier projet + `export PYTHONPATH=.` |
| `File does not exist: dashboard/app.py` | mauvais dossier courant | `cd` dans le dossier qui contient `dashboard/` |
| `Permission denied: .venv` | dossier en lecture seule | `chmod -R u+rwX .` puis recréer le venv |
| `pyexpat ... Symbol not found` (macOS) | Python Homebrew cassé | créer le venv avec **uv** (`uv venv --python 3.12`) |
| `No matching distribution scikit-learn` | Python < 3.11 | installer **Python 3.12** |
| `Aucune mesure énergétique` (onglet Green IT) | aucun entraînement encore lancé | lancer `python -m scripts.train_model` |
| push GitHub refusé (fichier > 100 Mo) | modèle versionné par erreur | vérifier que `models/` est dans `.gitignore` |

---

### 8. FAQ

**Le dashboard fonctionne sans identifiants Bluesky ?** Oui — ils ne servent qu'à
la collecte en direct. L'analyse et la démo marchent sans.

**Pourquoi CamemBERT n'est-il pas meilleur que le TF-IDF ?** Parce que les données
sont à 98 % anglaises et que CamemBERT est français, et que le fine-tuning (2 epochs)
est court. C'est documenté et assumé.

**Le CO₂ affiché est-il exact ?** Non, sur une machine sans capteur c'est une
**estimation** à partir de la durée. La durée, elle, est mesurée réellement.

**Comment revenir au TF-IDF après avoir testé le transformer ?** Lancez sans la
variable, ou `unset MODEL_BACKEND`, puis relancez.
