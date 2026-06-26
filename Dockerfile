FROM python:3.12-slim

WORKDIR /app

# Dependances systeme minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Installation des dependances Python (cache Docker optimise)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code du projet
COPY . .

ENV PYTHONPATH=/app

EXPOSE 8501

# Au demarrage : on s'assure d'avoir les vraies donnees + un modele, puis le dashboard
CMD bash -c "\
    [ -f data/raw/combined_dataset.csv ] || python -m scripts.download_dataset && \
    [ -f models/fake_news_clf.joblib ] || python -m scripts.train_model && \
    streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0"
