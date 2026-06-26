"""
Dashboard Thumalien (Streamlit).

Lancement :
    streamlit run dashboard/app.py

3 onglets :
 1. Analyse en direct : colle un message -> score de credibilite, emotion,
    explicabilite (poids des mots), penalites.
 2. Vue d'ensemble    : statistiques sur les messages deja analyses (base).
 3. Green IT          : consommation energetique des traitements.
"""
import sys
from pathlib import Path

# rend les imports du projet disponibles quel que soit le repertoire courant
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings
from src.pipeline import analyze_post
from src.storage import database as db

st.set_page_config(page_title="Thumalien – Detection Fake News",
                   page_icon="🛰️", layout="wide")


# ---------- helpers visuels ----------
def gauge_credibilite(score: int, couleur: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100"},
        title={"text": "Score de credibilite"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": couleur},
            "steps": [
                {"range": [0, settings.CRED_SEUIL_DOUTEUX], "color": "#FDECEA"},
                {"range": [settings.CRED_SEUIL_DOUTEUX, settings.CRED_SEUIL_VIGILANCE], "color": "#FFF8E1"},
                {"range": [settings.CRED_SEUIL_VIGILANCE, 100], "color": "#E8F5E9"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def bar_emotions(scores: dict):
    items = {k: v for k, v in scores.items() if v > 0}
    if not items:
        return None
    fig = go.Figure(go.Bar(
        x=list(items.values()), y=list(items.keys()), orientation="h",
        marker_color=settings.COULEUR_PRIMAIRE,
    ))
    fig.update_layout(height=240, margin=dict(t=20, b=20, l=20, r=20),
                      xaxis_title="intensite relative")
    return fig


# ---------- en-tete ----------
st.title("🛰️ Thumalien — Detection de fake news sur Bluesky")
st.caption("MVP — pipeline NLP : credibilite + analyse emotionnelle + explicabilite + suivi energetique")

tab1, tab2, tab3 = st.tabs(["🔎 Analyse en direct", "📊 Vue d'ensemble", "🌱 Green IT"])


# ============ ONGLET 1 : ANALYSE EN DIRECT ============
with tab1:
    st.subheader("Analyser un message")
    exemple = "URGENT !! Ce remede miracle soigne TOUT en 24h, les medecins sont furieux, partagez vite !!"
    txt = st.text_area("Texte du message", value=exemple, height=110)

    if st.button("Analyser", type="primary"):
        if not txt.strip():
            st.warning("Saisis un message.")
        else:
            res = analyze_post(txt)
            cred = res["credibility"]
            emo = res["emotion"]

            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(gauge_credibilite(cred["score"], cred["couleur"]),
                                use_container_width=True)
                st.markdown(
                    f"<h4 style='color:{cred['couleur']}'>Verdict : {cred['niveau'].upper()}</h4>",
                    unsafe_allow_html=True)
                st.write(f"Langue detectee : **{res['lang']}** · "
                         f"Probabilite fake (modele) : **{res['proba_fake']:.0%}**")

            with c2:
                st.markdown(f"**Emotion dominante : `{emo['dominante']}`**"
                            + (" · 😏 ton humoristique/ironique detecte" if emo["humour"] else ""))
                figE = bar_emotions(emo["scores"])
                if figE:
                    st.plotly_chart(figE, use_container_width=True)
                else:
                    st.info("Aucune charge emotionnelle marquee (ton neutre).")

            st.divider()
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**🔴 Mots qui poussent vers FAKE**")
                mf = res["explanation"].get("mots_vers_fake", [])
                if mf:
                    st.dataframe(pd.DataFrame(mf, columns=["mot", "poids"]),
                                 hide_index=True, use_container_width=True)
                else:
                    st.write("—")
            with cc2:
                st.markdown("**🟢 Mots qui poussent vers FIABLE**")
                mv = res["explanation"].get("mots_vers_fiable", [])
                if mv:
                    st.dataframe(pd.DataFrame(mv, columns=["mot", "poids"]),
                                 hide_index=True, use_container_width=True)
                else:
                    st.write("—")

            if cred["penalites"]:
                st.markdown("**⚠️ Penalites appliquees (signaux de surface)**")
                st.json(cred["penalites"])

            with st.expander("Detail technique (JSON complet)"):
                st.json(res)


# ============ ONGLET 2 : VUE D'ENSEMBLE ============
with tab2:
    st.subheader("Messages deja analyses (base de donnees)")
    try:
        rows = db.fetch_all_analyses()
    except Exception:
        rows = []

    if not rows:
        st.info("Aucune analyse en base. Lance : "
                "`python -m scripts.run_pipeline --demo --n 30`")
    else:
        df = pd.DataFrame(rows)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Messages analyses", len(df))
        k2.metric("Douteux", int((df["niveau"] == "douteux").sum()))
        k3.metric("A verifier", int((df["niveau"] == "a verifier").sum()))
        k4.metric("Credibilite moy.", f"{df['credibility_score'].mean():.0f}/100")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Repartition par niveau de credibilite**")
            vc = df["niveau"].value_counts()
            fig = go.Figure(go.Bar(x=vc.index, y=vc.values,
                                   marker_color=settings.COULEUR_PRIMAIRE))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            st.markdown("**Repartition des emotions dominantes**")
            ve = df["emotion_dominante"].value_counts()
            fig = go.Figure(go.Bar(x=ve.index, y=ve.values,
                                   marker_color="#6A1B9A"))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Derniers messages**")
        st.dataframe(
            df[["text_clean", "lang", "credibility_score", "niveau",
                "emotion_dominante"]].head(50),
            hide_index=True, use_container_width=True)


# ============ ONGLET 3 : GREEN IT ============
with tab3:
    st.subheader("Suivi energetique (CodeCarbon)")
    log = settings.ENERGY_LOG_DIR / "emissions.csv"
    if not log.exists():
        st.info("Aucune mesure energetique encore. Lance un entrainement ou "
                "un batch pour generer `energy_logs/emissions.csv`.")
    else:
        dfe = pd.read_csv(log)
        st.write(f"{len(dfe)} executions tracees.")
        cols = [c for c in ["project_name", "duration", "emissions",
                            "energy_consumed", "cpu_energy", "ram_energy"]
                if c in dfe.columns]
        st.dataframe(dfe[cols].tail(20), hide_index=True, use_container_width=True)
        if "emissions" in dfe.columns:
            tot = dfe["emissions"].sum()
            st.metric("CO2 cumule (kg eq)", f"{tot:.6f}")
        st.caption("En environnement sans capteur energetique (RAPL/NVML), "
                   "CodeCarbon bascule en mode degrade : la duree est mesuree, "
                   "le CO2 peut etre absent. Sur une machine standard, les "
                   "colonnes energie/CO2 sont remplies.")
