# Fonctionnalité de Téléchargement ECG et Génération de Rapport PDF

## Vue d'ensemble

La nouvelle fonctionnalité permet aux patients de :
1. **Télécharger des fichiers ECG** (CSV, JSON, Excel, TXT)
2. **Générer automatiquement un rapport PDF** contenant :
   - Informations du patient (nom, ID, âge, genre, date)
   - Résultats de la prédiction (classe, modèle utilisé, confiance)
   - Métriques du modèle (accuracy, F1-score, source)
   - En-têtes et pieds de page professionnels

## Architecture

### Frontend (`front-end/components/patient/ecg-upload.tsx`)

**Composant ECGUpload (267 lignes)**
- Sélection de fichiers (drag & drop support)
- Validation du type et de la taille
- Affichage des informations du fichier
- Boutons pour générer et télécharger le rapport PDF

**Intégration dans PatientPrediction**
- Ajoutée à la fin de la page de prédiction
- Disponible après les résultats de prédiction
- Communication bidirectionnelle avec le backend

### Backend (`app/main.py`)

**Nouvelle Endpoint 1: `/upload-ecg` (POST)**
```
POST /upload-ecg
Content-Type: application/json

Body:
{
  "patient_id": "string",
  "patient_name": "string",
  "file_content": "string (contenu du fichier)"
}

Response:
{
  "status": "success",
  "message": "ECG file uploaded successfully",
  "file_path": "string",
  "patient_id": "string",
  "patient_name": "string",
  "upload_time": "string"
}
```

**Nouvelle Endpoint 2: `/patient-report` (POST)**
```
POST /patient-report
Content-Type: application/json

Body:
{
  "patient_name": "string",
  "patient_id": "string",
  "age": "integer",
  "sex": "string (M/F)",
  "prediction_result": "integer (0 ou 1)",
  "model_used": "string",
  "confidence": "float",
  "data": {
    "features": [float, ...]
  }
}

Response: PDF file (application/pdf)
```

## Flux d'utilisation

### Étape 1: Navigation vers "Prédiction"
L'utilisateur clique sur l'onglet "Prediction" dans le menu patient

### Étape 2: Saisie des données ECG
L'utilisateur remplit les champs ECG manuellement ou télécharge un fichier

### Étape 3: Lancer la prédiction
L'utilisateur clique sur "Lancer la Prédiction"
- Le backend analyse les données avec le modèle ML sélectionné
- Les résultats s'affichent dans le panel de droite

### Étape 4: Télécharger le fichier ECG (optionnel)
L'utilisateur clique sur "Cliquez pour sélectionner un fichier" dans la section ECG Upload
- Sélectionne un fichier CSV/JSON/Excel
- Le fichier est validé (type, taille < 10MB)
- Le fichier est envoyé au backend via `/upload-ecg`

### Étape 5: Générer et télécharger le rapport PDF
L'utilisateur clique sur "Générer Rapport" ou "Télécharger PDF"
- Le backend génère un PDF professionnel via ReportLab
- Le PDF contient tous les informations pertinentes
- L'utilisateur peut télécharger le PDF directement

## Structure du Rapport PDF

```
┌─────────────────────────────────────────┐
│  Rapport d'Analyse ECG                  │
│  CardioSense - ECG Arrhythmia Detection │
└─────────────────────────────────────────┘

Informations du Patient
┌──────────────────────────────────────────┐
│ Nom Complet        | Jean Dupont         │
│ ID Patient         | PAT-001234          │
│ Âge                | 54 ans              │
│ Genre              | Masculin            │
│ Date du Rapport    | 05/03/2026 14:30    │
└──────────────────────────────────────────┘

Résultats de la Prédiction
┌──────────────────────────────────────────┐
│ Classe Prédite     | Normal (Classe 0)   │
│ Modèle Utilisé     | RandomForest        │
│ Confiance          | 95.23%              │
│ Nombre de Features | 278                 │
└──────────────────────────────────────────┘

Métriques du Modèle
┌──────────────────────────────────────────┐
│ Métrique           | Valeur              │
├──────────────────────────────────────────┤
│ Accuracy           | 96.20%              │
│ F1 Score           | 0.9470              │
│ Source             | Cache               │
└──────────────────────────────────────────┘

[Disclaimer légal du rapport]
```

## Formats de fichiers supportés

| Format | Extension | Support |
|--------|-----------|---------|
| CSV    | .csv      | ✅ Natif |
| JSON   | .json     | ✅ Natif |
| Excel  | .xlsx     | ✅ Via export |
| Excel  | .xls      | ✅ Via export |
| Texte  | .txt      | ✅ Natif |

## Validation des fichiers

**Validations effectuées:**
1. **Type de fichier**: CSV, JSON, Excel, TXT
2. **Taille maximale**: 10 MB
3. **Contenu**: Validation basique du format
4. **Détection d'erreurs**: Messages d'erreur clairs

## Gestion des erreurs

### Erreurs possibles

| Erreur | Cause | Solution |
|--------|-------|----------|
| Format non supporté | Type MIME non reconnu | Utiliser un format supporté |
| Fichier trop volumineux | Taille > 10 MB | Réduire la taille du fichier |
| Erreur d'upload | Problème réseau/serveur | Réessayer |
| Erreur de génération PDF | ReportLab non installé | Installer reportlab |

### Messages d'erreur

L'application affiche des alerts claires avec les icônes appropriées:
- 🔴 Erreurs en rouge
- ✅ Succès en vert
- ⚠️ Avertissements en orange

## Caractéristiques techniques

### Frontend
- **Composant**: React 19.2 + TypeScript
- **Validation**: Type et taille client-side
- **Icônes**: Lucide React
- **État**: React Hooks (useState)
- **Requêtes**: Fetch API

### Backend
- **Framework**: FastAPI
- **Report Generation**: ReportLab
- **Stockage**: Système de fichiers local
- **Sérialisation**: JSON
- **Timestamps**: ISO 8601 format

## Performance

- **Upload**: < 1 seconde (fichiers < 1 MB)
- **Génération PDF**: 2-3 secondes
- **Téléchargement PDF**: Immédiat (streaming)

## Sécurité

✅ **Mesures implémentées:**
- Validation de la taille des fichiers (max 10 MB)
- Validation du type MIME
- Timestamps uniques pour éviter les collisions
- Chemin de fichier sécurisé
- Gestion des exceptions

## Tests

### Tests manuels recommandés

1. **Test d'upload**
   ```bash
   # Sélectionner un fichier CSV
   # Vérifier: upload réussi, fichier sauvegardé
   ```

2. **Test de rapport**
   ```bash
   # Générer un rapport
   # Vérifier: PDF créé, données correctes
   ```

3. **Test d'erreur**
   ```bash
   # Essayer upload trop volumineux
   # Vérifier: message d'erreur clair
   ```

### Test API avec curl

```bash
# Test upload
curl -X POST "http://127.0.0.1:8000/upload-ecg" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAT-001",
    "patient_name": "Jean Dupont",
    "file_content": "age,sex,heartRate\n54,1,72"
  }'

# Test rapport
curl -X POST "http://127.0.0.1:8000/patient-report?patient_name=Jean+Dupont&patient_id=PAT-001&age=54&sex=M&prediction_result=0&model_used=RandomForest&confidence=0.95" \
  -H "Content-Type: application/json" \
  -d '{"features": [54, 1, 172, ...]}' \
  --output rapport.pdf
```

## Dépannage

### Le bouton de téléchargement ne fonctionne pas
**Solution**: Vérifier que l'API backend est en cours d'exécution sur le port 8000

### Le PDF est vide
**Solution**: Vérifier que ReportLab est installé: `pip install reportlab`

### Le fichier n'est pas accepté
**Solution**: Utiliser un format supporté (CSV, JSON, Excel, TXT)

## Extensions futures

✨ **Améliorations possibles:**
- [ ] Support des fichiers audio ECG brutes
- [ ] Prévisualisation du fichier avant upload
- [ ] Historique des uploads
- [ ] Partage de rapports par email
- [ ] Signature numérique du rapport
- [ ] Multi-langue pour les rapports

## Références

- **ReportLab Documentation**: https://www.reportlab.com/
- **FastAPI File Upload**: https://fastapi.tiangolo.com/
- **React File Handling**: https://react.dev/
