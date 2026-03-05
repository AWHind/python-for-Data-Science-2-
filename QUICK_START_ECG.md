# Guide de Démarrage Rapide - Fonctionnalité ECG & PDF

## 🚀 Démarrer en 5 minutes

### Prérequis
- Python 3.8+
- Node.js 18+
- pnpm (npm fonctionnera aussi)

### Étape 1: Démarrer l'API Backend (Terminal 1)

```bash
# Aller dans le répertoire de l'application
cd /vercel/share/v0-project

# Installer les dépendances Python (si nécessaire)
pip install -r requirements.txt

# Démarrer l'API
cd app
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Résultat attendu:**
```
✅ Uvicorn running on http://127.0.0.1:8000
✅ Application startup complete
```

### Étape 2: Démarrer le Frontend (Terminal 2)

```bash
# Aller dans le répertoire frontend
cd /vercel/share/v0-project/front-end

# Installer les dépendances
npm install
# ou
pnpm install

# Démarrer le serveur de développement
npm run dev
# ou
pnpm dev
```

**Résultat attendu:**
```
✅ Ready in 8.2s
✅ Local: http://localhost:3000
```

### Étape 3: Accéder à l'Application

Ouvrir dans le navigateur:
```
http://localhost:3000
```

### Étape 4: Tester la Fonctionnalité

1. **Se connecter**
   - Email: `patient@cardiosense.com`
   - Mot de passe: `patient123`

2. **Naviguer vers Prédiction**
   - Cliquer sur "Prediction" dans le menu

3. **Remplir les données ECG**
   - Les valeurs par défaut sont pré-remplies
   - Cliquer sur "Lancer la Prediction"

4. **Voir les résultats**
   - Les résultats s'affichent à droite
   - Confiance et détails visibles

5. **Télécharger un fichier ECG** (nouveau!)
   - Défiler vers le bas jusqu'à "Télécharger les Données ECG"
   - Cliquer sur la zone d'upload ou drag & drop
   - Sélectionner un fichier CSV/JSON/Excel/TXT

6. **Générer un rapport PDF** (nouveau!)
   - Après upload, cliquer sur "Télécharger PDF"
   - Le rapport PDF se télécharge automatiquement

---

## 📋 Fichiers de Test

### Exemple de fichier CSV

Créer un fichier `test_ecg.csv`:

```csv
age,sex,heartRate,qrsDuration,prInterval,qtInterval,tInterval,pInterval
54,1,72,80,160,370,180,100
60,0,68,78,155,365,175,98
45,1,75,82,165,375,185,102
```

### Exemple de fichier JSON

Créer un fichier `test_ecg.json`:

```json
{
  "patients": [
    {
      "age": 54,
      "sex": "M",
      "heartRate": 72,
      "qrsDuration": 80
    }
  ]
}
```

---

## 🔍 Vérifier que Tout Fonctionne

### API Swagger Documentation
```
http://127.0.0.1:8000/docs
```

Chercher les endpoints:
- `/upload-ecg` ✅
- `/patient-report` ✅

### Test d'Upload avec cURL

```bash
# Test d'upload simple
curl -X POST "http://127.0.0.1:8000/upload-ecg" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "TEST-001",
    "patient_name": "Test Patient",
    "file_content": "age,sex,heartRate\n54,1,72"
  }'
```

**Réponse attendue:**
```json
{
  "status": "success",
  "message": "ECG file uploaded successfully",
  "file_path": "/path/to/uploads/...",
  "patient_id": "TEST-001",
  "patient_name": "Test Patient",
  "upload_time": "20260305_143000"
}
```

---

## 🐛 Dépannage

### Erreur: "Cannot connect to API"
**Solution:** Vérifier que l'API est en cours d'exécution
```bash
curl http://127.0.0.1:8000/health
```

### Erreur: "ReportLab not installed"
**Solution:** Installer ReportLab
```bash
pip install reportlab
```

### L'upload de fichier échoue
**Solution:** Vérifier le format et la taille
- Format accepté: CSV, JSON, Excel, TXT
- Taille maximale: 10 MB

### Le PDF est vide
**Solution:** Réessayer, c'est peut-être un problème réseau

---

## ✅ Checklist de Vérification

- [ ] API démarrée (http://127.0.0.1:8000)
- [ ] Frontend démarré (http://localhost:3000)
- [ ] Connexion réussie
- [ ] Page "Prédiction" accessible
- [ ] Formulaire ECG visible
- [ ] Section "Télécharger les Données ECG" visible
- [ ] Upload de fichier fonctionne
- [ ] Rapport PDF généré avec succès
- [ ] PDF téléchargé correctement

---

## 📚 Documentation Complète

Pour plus de détails, consulter:
- **ECG_UPLOAD_PDF_FEATURE.md** - Guide complet de la fonctionnalité
- **FINAL_IMPLEMENTATION_SUMMARY.md** - Résumé des changements
- **API_DOCUMENTATION.md** - Référence API complète

---

## 💡 Astuces

**Pour développement:**
```bash
# Terminal 3: Monitorer les logs
tail -f /vercel/share/v0-project/uploads/*.csv
```

**Pour déboguer:**
```bash
# Ouvrir DevTools du navigateur
F12 → Console → Voir les logs [v0]
```

**Pour réinitialiser:**
```bash
# Supprimer les fichiers uploadés
rm -rf /vercel/share/v0-project/uploads/*
```

---

## 🎉 Vous Êtes Prêt!

La fonctionnalité d'upload ECG et de génération de rapport PDF est maintenant prête à l'emploi.

**Rendez-vous à l'étape 4 pour tester!** ✨
