# python-for-Data-Science-2
# Détection des arythmies cardiaques à l’aide du Machine Learning

##  Description du projet
Ce projet vise à concevoir et entraîner un modèle de Machine Learning permettant de détecter automatiquement la présence d’arythmies cardiaques à partir de données issues de signaux ECG (électrocardiogramme). 
Les données utilisées sont des données médicales multivariées représentant différentes caractéristiques physiologiques du rythme cardiaque.

L’objectif principal est de proposer une approche d’aide au diagnostic médical, en distinguant un rythme cardiaque normal d’un rythme anormal.
Le projet couvre les principales étapes d’un processus de Data Science, incluant l’analyse exploratoire des données, le prétraitement, la modélisation et l’évaluation des performances,
dans le cadre du module Python for Data Science 2.

##  Dataset
- **Nom** : Arrhythmia Dataset  
- **Source** : UCI Machine Learning Repository  
- **Lien** : https://archive.ics.uci.edu/ml/datasets/Arrhythmia  
- **Nombre d’instances** : 452 patients  
- **Nombre de variables** : 279 attributs  
- **Type de données** : numériques et catégorielles  
- **Valeurs manquantes** : Oui  

###  Variable cible
- "class"
  - "1": ECG normal  
  - "2 à 16" : différents types d’arythmies cardiaques  

 Dans ce projet le problème est formulé comme une **classification binaire** :
- "0" : rythme cardiaque normal  
- "1" : présence d’une arythmie  


##  Objectifs du projet
- Comprendre et analyser des données ECG complexes
- Effectuer une analyse exploratoire des données (EDA)
- Gérer les valeurs manquantes et le déséquilibre des classes
- Construire et comparer plusieurs modèles de classification
- Évaluer les performances des modèles
- Proposer une solution fiable d’aide à la décision médicale

##  Méthodologie
1. Chargement et exploration des données
2. Nettoyage des données
   - traitement des valeurs manquantes
   - normalisation / standardisation
3. Préparation des données
   - transformation de la variable cible
   - sélection de caractéristiques
4. Modélisation
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Gradient Boosting
5. Évaluation
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Matrice de confusion


##  Technologies utilisées
- Python
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- Jupyter Notebook


## Équipe du projet
- Nom de l’étudiante : Hind Elawity  
- Groupe : B  
- Matière : Python for Data Science 2  


##  Conclusion
Ce projet permet de mettre en pratique les concepts fondamentaux de la Data Science et du Machine Learning 
sur un dataset médical réel et complexe, tout en soulignant l’importance de ces techniques dans le domaine de la santé.


