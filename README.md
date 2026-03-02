# 🫀 Détection et Classification des Arythmies Cardiaques  
### Projet Data Science – Hind Elawity | Groupe B

---

## 📖 Présentation

Ce projet consiste à développer un modèle de Machine Learning capable de détecter et classifier les arythmies cardiaques à partir de données médicales issues d’électrocardiogrammes (ECG).

Les arythmies cardiaques représentent un enjeu important en cardiologie. Une détection précoce permet d’améliorer le diagnostic et la prise en charge des patients.

Ce travail a été réalisé dans le cadre de la formation **Data Science – Hind Elawity (Groupe B)**.

---

## 🎯 Objectifs

- Analyser un dataset médical complexe
- Nettoyer et préparer les données
- Construire plusieurs modèles de classification
- Comparer leurs performances
- Sélectionner le modèle le plus performant

---

## 📊 Jeu de Données

Dataset utilisé : **Arrhythmia – UCI Machine Learning Repository**

Lien officiel :  
https://archive.ics.uci.edu/ml/datasets/arrhythmia

### Caractéristiques :

- 452 patients
- 279 variables
- 16 classes d’arythmie
- Données multivariées
- Présence de valeurs manquantes

Le dataset contient des mesures électrocardiographiques détaillées, ce qui en fait un problème de classification à haute dimensionnalité.

---

## 🧠 Méthodologie

### 1️⃣ Analyse des données
- Étude de la structure du dataset
- Analyse de la distribution des classes
- Identification des valeurs manquantes

### 2️⃣ Prétraitement
- Gestion des valeurs manquantes
- Suppression des variables peu pertinentes
- Normalisation des variables numériques
- Encodage des variables catégorielles

### 3️⃣ Analyse exploratoire (EDA)
- Visualisation des distributions
- Analyse des corrélations
- Détection du déséquilibre des classes

### 4️⃣ Modélisation
Implémentation et comparaison de plusieurs modèles :

- Régression Logistique
- Random Forest
- Support Vector Machine (SVM)

### 5️⃣ Évaluation
Les modèles ont été évalués à l’aide de :

- Accuracy
- Precision
- Recall
- F1-score
- Matrice de confusion

Le modèle final a été sélectionné selon ses performances globales et sa capacité de généralisation.

---

## 📈 Résultats

- Amélioration des performances après prétraitement
- Réduction des erreurs de classification
- Mise en évidence de l’importance du nettoyage des données dans les projets médicaux

---

## 🛠️ Technologies Utilisées

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 🚀 Exécution du Projet

```bash
git clone https://github.com/votre-username/arrhythmia-classification.git
cd arrhythmia-classification
pip install -r requirements.txt
jupyter notebook
