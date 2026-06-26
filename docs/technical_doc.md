# Dossier technique — Thumalien

Détection de fake news sur Bluesky · Mastère 1 Big Data & IA · SUP DE VINCI

---

## 1. Contexte et problématique

Le client fictif **Thumalien** veut analyser automatiquement les messages
Bluesky pour : repérer les contenus douteux, évaluer leur charge émotionnelle,
et aider au fact-checking. Contraintes clés issues du cahier des charges :
multilingue (FR/EN), transparence (expliquer **pourquoi** un contenu est jugé
douteux), et suivi énergétique (Green IT).

## 2. Architecture générale

Pipeline en étapes découplées, chacune isolée dans un module :

```
Bluesky API ──► Collecte ──► Prétraitement ──► Détection ──► Crédibilité
                                                   │
                                                   ├──► Analyse émotionnelle
                                                   ├──► Explicabilité
                                                   └──► Stockage ──► Dashboard
                          (le tout mesuré par CodeCarbon : Green IT)
```

Découpler permet de remplacer une brique (ex. modèle baseline → BERT) sans
toucher au reste, et de répartir le travail entre les 4 rôles.

## 3. Collecte des données (Data Engineer)

- **Source** : API officielle Bluesky via le protocole AT (`atproto`).
- **Auth** : handle + App Password, lus dans `.env` (jamais en dur).
- **Méthode** : `app.bsky.feed.search_posts` par mots-clés + filtre langue.
- **Normalisation** : chaque post → `{id (URI), text_raw, text_clean, lang,
  author, source, created_at}`.
- **Stockage brut** : CSV horodaté dans `data/raw/` + base SQLite.
- **Volumétrie** : garde-fou `COLLECT_MAX_POSTS` (config) pour ne pas saturer
  l'API. *[À COMPLÉTER : volumétrie réelle visée selon la campagne de collecte.]*

## 4. Prétraitement NLP (Data Engineer)

Module `src/preprocess/cleaner.py`. Étapes :
1. Suppression URLs, mentions `@`, caractères de contrôle ; les hashtags sont
   conservés sans le `#`.
2. **Détection de langue** (`langdetect`, rendue déterministe).
3. Deux versions du texte conservées :
   - `text_clean` (lisible, pour le dashboard et l'explicabilité),
   - `text_norm` (minuscule, pour les features du modèle).
4. **Features de surface** calculées AVANT normalisation, car ce sont des
   signaux utiles : ratio de MAJUSCULES, nombre de `!`, de `?`, longueur.

## 5. Détection de fake news (Data Scientist)

Module `src/detect/classifier.py`.

**Baseline (livrée)** : `TfidfVectorizer` (uni + bigrammes, accents conservés)
→ `LogisticRegression` (class_weight="balanced").

Justification du choix :
- léger et rapide (cohérent avec l'axe Green IT),
- **interprétable** : les coefficients donnent directement le poids de chaque
  mot → explicabilité native (cf. §7),
- baseline solide pour comparer à un transformer.

**Évolution (documentée, même interface `train()` / `predict_proba_fake()`)** :
- **CamemBERT** (FR) / **RoBERTa** (EN) via `transformers`, fine-tuné sur un
  vrai dataset. Activer les dépendances commentées dans `requirements.txt`.
- Bénéfice attendu : meilleure compréhension du sens (au-delà du vocabulaire),
  gain de F1 sur les cas ambigus. Coût : plus lourd → arbitrage Green IT à
  documenter (comparer CO₂ baseline vs BERT).

### Données (réelles)

Le modèle livré est entraîné sur de **vraies données de fake news**, téléchargées
et harmonisées par `scripts/download_dataset.py` au format
`text, lang, label (1=fake, 0=fiable), split` :

| Dataset | Langue | Taille | Nature | Source / licence |
|---|---|---|---|---|
| **LIAR** (Wang, 2017) | EN | 12 836 | énoncés politiques PolitiFact, 6 niveaux → binaire | `thiagorainmaker77/liar_dataset` — usage académique |
| **OBSINFOX** | FR | 100 | titres d'articles, annotés par 8 experts | `obs-info/obsinfox` — **CC BY-NC 4.0** |
| **X-FACT** (Gupta & Srikumar, 2021) | FR | 158 | claims fact-checkées (sous-ensemble FR) | `utahnlp/x-fact` — usage académique |

Choix de conception traçables :
- **Mapping binaire LIAR** : FAKE = {pants-fire, false, barely-true} ;
  FIABLE = {half-true, mostly-true, true}.
- **Anti-fuite obsinfox** : le fichier brut contient 100 articles × 8 annotateurs
  (800 lignes, titres répétés). On **agrège par vote majoritaire** par article
  avant de splitter → aucun titre partagé entre train et test.
- **Splits** : LIAR conserve ses **splits officiels** train/valid/test ; le
  français (obsinfox + X-FACT) reçoit un split 80/20 stratifié, sources mélangées.
- La colonne `lang` permet une **évaluation séparée par langue** (cf. Métriques).

> Démo uniquement : `scripts/make_sample_data.py` génère des messages d'exemple
> **en entrée** du pipeline (posts à analyser), distincts des données
> d'entraînement réelles ci-dessus.

### Métriques (KPIs du cahier des charges)

Calculées dans `train()` et sauvegardées dans `models/metrics.json` :
- **F1-score** (KPI principal), précision, rappel, ROC-AUC,
- **matrice de confusion** → taux de faux positifs / faux négatifs (FP/FN),
- **temps moyen d'analyse** : mesuré par CodeCarbon (durée du batch).

**Résultats réels obtenus** (jeu de test) :

| | F1 | Accuracy | ROC-AUC | n |
|---|---|---|---|---|
| 🇬🇧 Anglais (LIAR) | **0.56** | 0.62 | 0.67 | 1 283 |
| 🇫🇷 Français (obsinfox + X-FACT) | 0.63 | 0.52 | **0.46** | 52 |
| Global | 0.56 | 0.62 | 0.66 | 1 335 |

Interprétation (cf. §13 Limites) : l'anglais est cohérent avec les baselines
publiées sur LIAR (la tâche est difficile sur texte court) ; le français reste
sous-performant faute de données, ce qui motive le passage à un transformer
multilingue.

## 6. Score de crédibilité

Module `src/detect/credibility.py`. Transforme la probabilité de fake en score
0-100 **transparent** :

```
score_base  = (1 - proba_fake) × 100
pénalités   = sensationnalisme (MAJUSCULES, !!!), message trop court…
score_final = clamp(score_base − pénalités, 0, 100)
```

Chaque pénalité est tracée et restituée → l'utilisateur voit le détail.
Seuils (config) : < 40 douteux (rouge) · 40-65 à vérifier (orange) · > 65 fiable (vert).

## 7. Analyse émotionnelle (Analyste IA)

Module `src/emotion/emotion_analyzer.py`. Approche lexicale (FR+EN) couvrant
colère, peur, joie, tristesse, surprise, dégoût + heuristique humour/ironie.
Léger, sans téléchargement de modèle (cohérent Green IT).

**Évolution** *[À COMPLÉTER]* : lexique complet (FEEL pour le FR, NRC Emotion
Lexicon) ou modèle EmotionBERT, en gardant l'interface `analyze_emotions()`.

## 8. Explicabilité (Analyste IA)

Module `src/explain/explainer.py`. Le modèle étant linéaire, la contribution
d'un terme = `tfidf(terme) × coefficient(terme)`. On restitue le top des mots
poussant vers « fake » et vers « fiable » (mots vides filtrés de l'affichage).

**Évolution** : avec un transformer, remplacer par **SHAP** ou **LIME** (même
interface `explain()`).

## 9. Suivi énergétique — Green IT

Module `src/energy/carbon_tracker.py` (CodeCarbon). Context manager /
décorateur autour de l'entraînement et de l'inférence. Sortie cumulée dans
`energy_logs/emissions.csv` (durée, énergie kWh, CO₂ kg eq), affichée dans
l'onglet Green IT du dashboard.

> Sur une machine sans capteur (RAPL/NVML), CodeCarbon passe en mode dégradé :
> la durée est mesurée, le CO₂ peut être absent. Sur un poste standard, les
> colonnes énergie/CO₂ sont remplies.

Analyse attendue dans le dossier : comparer le coût énergétique baseline vs BERT,
et estimer l'économie d'énergie (KPI « économie énergétique »).

## 10. Stockage

Module `src/storage/database.py`. SQLite par défaut (zéro config). Schéma isolé
→ bascule PostgreSQL/MongoDB possible en réimplémentant les mêmes fonctions.
Tables : `posts` (brut + métadonnées) et `analyses` (résultats).

## 11. Dashboard (Analyste IA)

`dashboard/app.py` (Streamlit + Plotly). 3 onglets : analyse en direct (jauge
crédibilité, émotions, poids des mots, pénalités), vue d'ensemble (stats base),
Green IT (énergie).

## 12. Qualité, reproductibilité, industrialisation

- `random_state` fixé partout → reproductibilité.
- Tests unitaires (`pytest`).
- Dockerfile + docker-compose → exécution reproductible.
- Configuration centralisée, secrets dans `.env` (hors Git).
- **Pistes scalabilité** *[À COMPLÉTER]* : ingestion temps réel Kafka,
  traitement distribué Spark, suivi d'expériences MLflow, CI GitHub Actions.

## 13. Limites connues (analyse critique)

- **Faiblesse sur le français** : avec ~258 exemples FR hétérogènes (deux
  domaines : titres de presse + claims fact-checkées), le TF-IDF ne discrimine
  pas (AUC ≈ 0.46). C'est la **rareté des données françaises** de fake news, pas
  un bug. → *Correctif prioritaire :* **transformer multilingue (CamemBERT /
  mBERT)**, qui exploite un pré-entraînement massif et demande moins de données
  labellisées. Interface `train()` / `predict_proba_fake()` déjà prévue pour
  brancher ce modèle.
- **Difficulté intrinsèque de la tâche** : même en anglais sur un benchmark de
  référence (LIAR), un F1 ≈ 0.56 est l'état de l'art pour des approches texte
  seul — la véracité d'un énoncé court dépend souvent de connaissances externes.
  → *Piste :* enrichir avec des features de contexte / vérification de sources.
- **Modèle baseline limité au vocabulaire** : ne « comprend » pas le sens, d'où
  l'apport attendu des transformers.
- **Lexique émotionnel compact** : à étendre (FEEL, NRC, EmotionBERT).
- **Détection d'humour/ironie rudimentaire** (heuristique).
- **Pas encore de collecte temps réel ni de passage à l'échelle** (Kafka/Spark).

Ces limites sont assumées et chiffrées : le garde-fou *language-agnostic*
(pénalités de sensationnalisme + tonalité émotionnelle) maintient un signal utile
sur le français en attendant un meilleur modèle.

## 14. KPIs — synthèse

| KPI (cahier des charges) | Où / comment |
|---|---|
| F1-score, FP/FN | `train()` → `models/metrics.json` |
| Temps moyen d'analyse | CodeCarbon (durée batch) |
| Économie énergétique | `energy_logs/emissions.csv` (comparaison baseline vs BERT) |
