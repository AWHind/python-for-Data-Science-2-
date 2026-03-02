🫀 Classification des Arythmies Cardiaques
📌 Description du Projet

Ce projet a pour objectif de détecter et classifier les arythmies cardiaques à l’aide des techniques de Machine Learning.

L’objectif principal est de développer un modèle capable de :

Détecter la présence ou l’absence d’arythmie

Classifier les patients dans l’un des 16 groupes médicaux

Comparer les performances de plusieurs modèles de classification

Ce travail permet d’appliquer les concepts de Data Science sur un dataset médical réel et complexe.

🎓 Contexte Académique

Ce projet a été réalisé dans le cadre de la formation Data Science – Hind Elawity (Groupe B).

L’objectif pédagogique était de suivre un workflow complet de Data Science :

Compréhension des données → Prétraitement → Analyse exploratoire → Modélisation → Évaluation.

📊 Dataset

Le projet utilise le dataset Arrhythmia disponible sur le dépôt officiel UCI Machine Learning Repository.

🔗 Lien du dataset :
https://archive.ics.uci.edu/ml/datasets/arrhythmia

Informations sur les données :

Nombre d’instances : 452 patients

Nombre de variables : 279 attributs

Type : Classification multivariée

Variable cible : Classe d’arythmie (16 catégories)

Valeurs manquantes : Oui

Le dataset contient des mesures issues d’ECG (électrocardiogramme), incluant des variables numériques et catégorielles.

La forte dimensionnalité et la présence de valeurs manquantes ont nécessité un travail important de prétraitement.

🧠 Méthodologie

Le projet a été structuré selon les étapes suivantes :

1️⃣ Compréhension des données

Analyse de la structure du dataset

Étude de la distribution des classes

Identification des valeurs manquantes

2️⃣ Prétraitement des données

Traitement des valeurs manquantes

Suppression des variables peu pertinentes

Normalisation des données numériques

Encodage des variables catégorielles

3️⃣ Analyse Exploratoire (EDA)

Analyse des corrélations

Visualisation des distributions

Détection d’un éventuel déséquilibre des classes

4️⃣ Modélisation

Implémentation et comparaison de plusieurs modèles :

Régression Logistique

Random Forest

Support Vector Machine (SVM)

5️⃣ Évaluation des performances

Les modèles ont été évalués à l’aide de :

Accuracy

Precision

Recall

F1-score

Matrice de confusion

Le modèle final a été sélectionné en fonction de ses performances globales et de sa capacité de généralisation.

📈 Résultats

Amélioration significative des performances après le prétraitement

Réduction des erreurs de classification

Mise en évidence de l’importance du nettoyage des données dans les projets médicaux

(Vous pouvez ajouter ici vos scores réels si nécessaire)

🛠️ Technologies utilisées

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Jupyter Notebook

🚀 Exécution du Projet
# Cloner le repository
git clone https://github.com/votre-username/arrhythmia-classification.git

# Accéder au dossier
cd arrhythmia-classification

# Installer les dépendances
pip install -r requirements.txt

# Lancer Jupyter Notebook
jupyter notebook
👨‍💻 Auteur

Votre Nom
Étudiant en Data Science
Hind Elawity – Groupe B
