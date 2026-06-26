# Guide utilisateur — Thumalien

Outil d'aide à la détection de fake news sur Bluesky. Ce guide s'adresse aux
**utilisateurs** (journalistes, fact-checkers, citoyens) et explique comment
lire les résultats — sans connaissance technique.

---

## 1. À quoi sert l'outil ?

Vous collez un message (ou vous analysez un lot de messages collectés sur
Bluesky), et l'outil vous donne :

- un **score de crédibilité de 0 à 100**,
- l'**émotion dominante** du message,
- les **mots** qui ont conduit à ce jugement,
- des **signaux d'alerte** (sensationnalisme…).

⚠️ **L'outil est une aide, pas un verdict.** Un score bas signifie « à vérifier
en priorité », pas « c'est faux à coup sûr ». Le fact-checking humain reste
indispensable.

## 2. Lancer le dashboard

```bash
streamlit run dashboard/app.py
```
Une page web s'ouvre (http://localhost:8501).

## 3. Onglet « Analyse en direct »

1. Collez votre message dans la zone de texte.
2. Cliquez sur **Analyser**.
3. Lisez les résultats :

**La jauge de crédibilité**
- 🟢 **Vert (> 65)** : fiable — signaux plutôt rassurants.
- 🟠 **Orange (40-65)** : à vérifier — prudence.
- 🔴 **Rouge (< 40)** : douteux — à vérifier en priorité.

**L'émotion dominante** : indique si le message joue sur la peur, la colère, la
joie… Les fausses informations exploitent souvent des émotions fortes.

**Les mots vers FAKE / vers FIABLE** : la transparence de l'outil. Vous voyez
les termes qui ont pesé. Ex. « miracle », « urgent », « partagez » tirent vers
le douteux ; « selon », « étude », une source nommée tirent vers le fiable.

**Les pénalités** : signaux de forme repérés (trop de MAJUSCULES, de `!!!`,
message très court).

## 4. Onglet « Vue d'ensemble »

Statistiques sur tous les messages déjà analysés : combien de douteux, quelles
émotions dominent, crédibilité moyenne, et la liste des derniers messages.

## 5. Onglet « Green IT »

Montre la consommation électrique et le CO₂ des traitements. L'outil se veut
sobre : modèles légers et mesure de l'empreinte.

## 6. Comment bien interpréter un résultat

- Un **score bas + émotion forte (peur/colère)** = signal classique de
  désinformation → vérifiez la source originale.
- Un **score élevé** ne garantit pas la vérité : vérifiez quand même les faits
  importants.
- Regardez **les mots mis en avant** : ils expliquent le score et aident à
  repérer le procédé (sensationnalisme, absence de source…).

## 7. Bonnes pratiques de vérification

1. Remonter à la **source d'origine** (qui dit ça, quand ?).
2. Croiser avec des **médias / fact-checkers reconnus**.
3. Se méfier des messages qui **pressent de partager vite**.
4. Distinguer **fait** (vérifiable) et **opinion**.

## 8. Questions fréquentes

**L'outil peut-il se tromper ?** Oui. Il donne une probabilité, pas une
certitude. Utilisez-le comme un premier tri.

**Fonctionne-t-il en anglais ?** Oui (FR et EN). La langue est détectée
automatiquement.

**Mes données partent-elles quelque part ?** Non : l'analyse tourne en local.
Seule la collecte interroge l'API publique Bluesky.
