"""
Configuration centrale du projet Thumalien.
Toutes les constantes / chemins / hyperparametres passent par ici.
On evite les valeurs magiques eparpillees dans le code.
"""
from pathlib import Path

# --- Arborescence ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT_DIR / "models"

for _d in (RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Base de donnees ---
# SQLite par defaut = zero config pour le MVP / la demo.
# Pour passer en PostgreSQL : remplacer DB_URL par une URL postgresql://...
# et adapter src/storage/database.py (le code est deja structure pour ca).
DB_PATH = DATA_DIR / "thumalien.db"
DB_URL = f"sqlite:///{DB_PATH}"

# --- Collecte Bluesky ---
# Les identifiants ne sont JAMAIS en dur : on les lit dans le .env (voir .env.example)
BLUESKY_DEFAULT_LANGS = ("fr", "en")
COLLECT_MAX_POSTS = 200          # garde-fou pour ne pas surcharger l'API
COLLECT_SEARCH_TERMS = ["actualite", "info", "breaking", "urgent"]

# --- Modele de detection ---
MODEL_PATH = MODELS_DIR / "fake_news_clf.joblib"
RANDOM_STATE = 42                # reproductibilite
TEST_SIZE = 0.2

# --- Choix du moteur de detection ---
# "tfidf"       : baseline legere (defaut, aucune dependance lourde)
# "transformer" : CamemBERT / DistilBERT multilingue (necessite torch+transformers)
# Surchargeable par variable d'environnement : MODEL_BACKEND=transformer
import os as _os
MODEL_BACKEND = _os.getenv("MODEL_BACKEND", "tfidf").lower()

# --- Parametres du transformer (utilises si MODEL_BACKEND=transformer) ---
# Modeles 100 % open source et gratuits (Hugging Face) :
#   "distilbert-base-multilingual-cased"  -> leger, FR+EN (DEFAUT, conseille laptop)
#   "camembert-base"                      -> meilleur en francais pur, plus lourd
#   "xlm-roberta-base"                    -> multilingue fort, plus lourd
TRANSFORMER_MODEL_NAME = _os.getenv(
    "TRANSFORMER_MODEL_NAME", "distilbert-base-multilingual-cased"
)
TRANSFORMER_DIR = MODELS_DIR / "transformer_clf"   # dossier du modele fine-tune
TRANSFORMER_MAX_LEN = 128         # longueur max des messages (tokens)
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_EPOCHS = 2
TRANSFORMER_LR = 2e-5
# Garde-fou Green IT : plafonne le nb d'exemples d'entrainement (None = tout).
# Utile pour une demo rapide sur CPU : ex. 2000.
TRANSFORMER_MAX_TRAIN_SAMPLES = None

# --- Seuils de credibilite ---
# Le score de credibilite va de 0 (tres douteux) a 100 (tres fiable).
# Bornes pour le code couleur du dashboard.
CRED_SEUIL_DOUTEUX = 40          # < 40  -> rouge
CRED_SEUIL_VIGILANCE = 65        # 40-65 -> orange ; > 65 -> vert

# --- Green IT ---
ENERGY_LOG_DIR = ROOT_DIR / "energy_logs"
ENERGY_LOG_DIR.mkdir(parents=True, exist_ok=True)
ENERGY_PROJECT_NAME = "thumalien"
# Facteur d'emission par defaut (France, gCO2eq/kWh). CodeCarbon affine selon la region.
ENERGY_COUNTRY_ISO = "FRA"

# --- Charte graphique dashboard ---
COULEUR_OK = "#2E7D32"           # vert (fiable)
COULEUR_VIGILANCE = "#F9A825"    # orange (a verifier)
COULEUR_ALERTE = "#C62828"       # rouge (douteux)
COULEUR_PRIMAIRE = "#1565C0"     # bleu
