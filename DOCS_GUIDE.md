# 📚 Guide de Navigation - Documentation Complète

Bienvenue ! Ce guide vous aide à trouver rapidement la documentation dont vous avez besoin.

---

## 🎯 Par Cas d'Usage

### 🚀 Je veux démarrer le projet immédiatement

1. **Lire** : `FRONTEND_README.md` (5 min)
2. **Exécuter** :
   ```bash
   cd front-end
   npm install
   npm run dev
   ```
3. **Accéder** : http://localhost:3000

### 🔧 Je veux comprendre la configuration technique

1. **Lire** : `FRONTEND_SETUP.md` (15 min)
2. **Comprendre** : Structure du projet, pages, authentification
3. **Consulter** : `next.config.mjs` et `tailwind.config.ts`

### 💻 Je veux développer/modifier le frontend

1. **Lire** : `FRONTEND_SETUP.md` - Section "Structure du projet"
2. **Voir** : `components/` pour les composants existants
3. **Référence** : `lib/auth-context.tsx` pour l'authentification

### 🌐 Je veux déployer sur Vercel

1. **Lire** : `FRONTEND_SETUP.md` - Section "Déploiement"
2. **Utiliser** : `vercel.json` (configuration automatique)
3. **Exécuter** : `npm run build`

### 🔗 Je veux connecter le backend

1. **Lire** : `FRONTEND_SETUP.md` - Section "Intégration Backend"
2. **Vérifier** : `components/dashboard/prediction-form.tsx`
3. **Backend** : Lancez `python -m uvicorn main:app --reload`

### 📊 Je veux tester l'API backend

1. **Lire** : `API_DOCUMENTATION.md` (dans le root)
2. **Voir** : Les endpoints et exemples
3. **Tester** : http://127.0.0.1:8000/docs (Swagger)

---

## 📖 Par Type de Documentation

### 🟢 Pour Commencer (Débutants)

| Document | Durée | Contenu |
|----------|-------|---------|
| **FRONTEND_README.md** | 5-10 min | Vue d'ensemble, commandes, pages |
| **00_START_HERE.md** | 10 min | Guide global du projet |
| **QUICK_START.md** | 5 min | Commandes copy-paste |

### 🟡 Documentations Techniques

| Document | Durée | Contenu |
|----------|-------|---------|
| **FRONTEND_SETUP.md** | 20 min | Structure, authentification, pages |
| **PREVIEW_OPTIMIZATION.md** | 15 min | Configuration pour v0 |
| **SETUP_GUIDE.md** | 30 min | Installation détaillée backend+frontend |
| **API_DOCUMENTATION.md** | 20 min | Endpoints, requêtes, réponses |

### 🔴 Références Complètes

| Document | Lignes | Contenu |
|----------|--------|---------|
| **README_COMPLETE.md** | 636 | Vue d'ensemble complète du projet |
| **INTEGRATION_COMPLETE.md** | 646 | Détails d'intégration complets |
| **DOCUMENTATION_INDEX.md** | 454 | Index complet de tous les fichiers |
| **CHANGELOG.md** | 381 | Historique de tous les changements |

---

## 🗺️ Carte Mentale du Projet

```
CardioSense/
├── 📘 Frontend (Next.js)
│   ├── Page d'accueil publique
│   ├── Authentification (Patient/Admin)
│   ├── Dashboard Patient
│   ├── Dashboard Admin
│   └── Intégration API Backend
│
├── 🐍 Backend (FastAPI)
│   ├── 4 Modèles ML (RF, SVM, LR, XGB)
│   ├── Endpoints de prédiction
│   ├── Génération de rapports PDF
│   └── Swagger documentation
│
├── 📊 MLflow Tracking
│   ├── Enregistrement des modèles
│   ├── Métriques de performance
│   └── Versioning des modèles
│
└── 📚 Documentation
    ├── Frontend
    ├── Backend
    ├── Configuration
    └── Intégration
```

---

## 📍 Localisation des Fichiers Clés

### Frontend

| Chemin | Fichier | Utilité |
|--------|---------|---------|
| `front-end/app/layout.tsx` | Layout racine | AuthProvider |
| `front-end/app/page.tsx` | Page principale | Routage basé sur auth |
| `front-end/app/globals.css` | Styles globaux | Design tokens |
| `front-end/lib/auth-context.tsx` | Authentification | Gestion login/register |
| `front-end/components/public-home.tsx` | Accueil | Page publique |
| `front-end/components/admin/` | Dashboard admin | Vue admin |
| `front-end/components/patient/` | Dashboard patient | Vue patient |
| `front-end/next.config.mjs` | Config Next.js | Optimisations |
| `front-end/package.json` | Dependencies | Scripts NPM |

### Backend

| Chemin | Fichier | Utilité |
|--------|---------|---------|
| `app/main.py` | FastAPI app | Routes et endpoints |
| `code/mlflow_tracking.py` | MLflow config | Configuration tracking |
| `code/modeling.py` | Modèles ML | Définition des modèles |
| `code/train_all_models.py` | Entraînement | Script d'entraînement |

### Documentation

| Chemin | Fichier | Lignes | Utilité |
|--------|---------|--------|---------|
| `FRONTEND_README.md` | Accueil frontend | 372 | Guide d'utilisation |
| `FRONTEND_SETUP.md` | Setup détaillé | 335 | Configuration technique |
| `PREVIEW_OPTIMIZATION.md` | Preview v0 | 322 | Optimisation pour v0 |
| `API_DOCUMENTATION.md` | API backend | 450 | Endpoints détaillés |
| `SETUP_GUIDE.md` | Installation | 522 | Guide complet |
| `README_COMPLETE.md` | Vue globale | 636 | Aperçu complet |
| `INTEGRATION_COMPLETE.md` | Intégration | 646 | Détails intégration |
| `DOCUMENTATION_INDEX.md` | Index | 454 | Index de navigation |
| `00_START_HERE.md` | Démarrage | 466 | Commencez ici |
| `CHANGELOG.md` | Changements | 381 | Historique |

---

## 🔀 Flux de Lectures Recommandés

### 👶 Pour Complètement Nouveau

```
1. 00_START_HERE.md          (5 min)   → Vue globale
2. FRONTEND_README.md        (5 min)   → Frontend specifique
3. QUICK_START.md            (3 min)   → Commandes
4. Essayer localement        (5 min)   → cd front-end && npm run dev
```

### 👨‍💻 Pour Développeur Frontend

```
1. FRONTEND_SETUP.md         (15 min)  → Structure détaillée
2. front-end/app/layout.tsx  (5 min)   → Layout principal
3. front-end/lib/auth-context.tsx (10 min) → Authentification
4. components/               (20 min)  → Explorer les composants
5. next.config.mjs           (5 min)   → Configuration Next.js
```

### 🔗 Pour Intégration Backend

```
1. API_DOCUMENTATION.md      (20 min)  → Endpoints
2. FRONTEND_SETUP.md         (10 min)  → "Intégration Backend"
3. prediction-form.tsx       (10 min)  → Exemple de requête API
4. admin-models.tsx          (10 min)  → Autre exemple
```

### 🚀 Pour Déploiement

```
1. FRONTEND_SETUP.md         (15 min)  → Section "Déploiement"
2. vercel.json               (2 min)   → Configuration
3. package.json              (2 min)   → Scripts de build
4. Déployer sur Vercel       (5 min)   → Connexion GitHub
```

### 📊 Pour Comprendre les Modèles ML

```
1. README_COMPLETE.md        (20 min)  → Aperçu ML
2. API_DOCUMENTATION.md      (15 min)  → Section "Modèles ML"
3. INTEGRATION_COMPLETE.md   (20 min)  → Détails des modèles
4. code/train_all_models.py  (15 min)  → Code d'entraînement
```

---

## 🆘 Troubleshooting Quick Access

### ❌ Le frontend ne démarre pas

→ Consulter : `FRONTEND_SETUP.md` - "Problèmes Courants"

### ❌ L'API ne répond pas

→ Consulter : `PREVIEW_OPTIMIZATION.md` - "Intégration Backend"

### ❌ Erreur de connexion

→ Consulter : `FRONTEND_SETUP.md` - "Configuration Authentification"

### ❌ Build échoue

→ Consulter : `FRONTEND_SETUP.md` - "Déploiement"

### ❌ Port occupé

→ Consulter : `PREVIEW_OPTIMIZATION.md` - "Troubleshooting"

### ❌ Module not found

→ Consulter : `FRONTEND_SETUP.md` - "Problèmes Courants"

---

## 📚 Table Complète des Documents

### 🟢 Documents à Lire en Premier

| # | Document | Type | Durée | Niveau |
|---|----------|------|-------|--------|
| 1 | **00_START_HERE.md** | Guide | 5 min | Débutant |
| 2 | **FRONTEND_README.md** | Guide | 5 min | Débutant |
| 3 | **QUICK_START.md** | Commandes | 3 min | Débutant |

### 🟡 Documents Techniques

| # | Document | Type | Durée | Niveau |
|---|----------|------|-------|--------|
| 4 | **FRONTEND_SETUP.md** | Technique | 20 min | Intermédiaire |
| 5 | **PREVIEW_OPTIMIZATION.md** | Technique | 15 min | Intermédiaire |
| 6 | **SETUP_GUIDE.md** | Technique | 30 min | Avancé |
| 7 | **API_DOCUMENTATION.md** | Technique | 20 min | Avancé |

### 🔴 Documents Complets

| # | Document | Type | Lignes | Niveau |
|---|----------|------|--------|--------|
| 8 | **README_COMPLETE.md** | Référence | 636 | Avancé |
| 9 | **INTEGRATION_COMPLETE.md** | Référence | 646 | Avancé |
| 10 | **DOCUMENTATION_INDEX.md** | Index | 454 | Tous |
| 11 | **CHANGELOG.md** | Historique | 381 | Tous |

### 🔧 Documents Spécialisés

| Document | Utilité |
|----------|---------|
| **CONFIGURATION_COMPLETE.md** | Résumé configuration |
| **DOCS_GUIDE.md** | Ce fichier |

---

## 💡 Tips Importants

### 📌 Astuce 1 : Démarrer Rapide

```bash
cd front-end
npm install
npm run dev
```

### 📌 Astuce 2 : Accès Comptes de Test

- Patient : `patient@cardiosense.com` / `patient123`
- Admin : `admin@cardiosense.com` / `admin123`

### 📌 Astuce 3 : Backend + Frontend

Terminal 1 :
```bash
cd front-end && npm run dev
```

Terminal 2 :
```bash
cd app && python -m uvicorn main:app --reload
```

### 📌 Astuce 4 : Accès Direct aux Pages

- Frontend : `http://localhost:3000`
- Backend API : `http://127.0.0.1:8000`
- Backend Docs : `http://127.0.0.1:8000/docs`
- MLflow : `http://127.0.0.1:5000`

### 📌 Astuce 5 : Nettoyage Complet

```bash
cd front-end
rm -rf node_modules .next pnpm-lock.yaml
npm install
npm run dev
```

---

## 🎯 Objectifs par Utilisateur

### 👤 Utilisateur Fin (Patient/Admin)
- Lire : `FRONTEND_README.md`
- Action : Se connecter et utiliser l'app
- Comptes : Utilisez les comptes de test

### 💼 Project Manager
- Lire : `00_START_HERE.md`, `README_COMPLETE.md`
- Comprendre : Vue d'ensemble complète
- Savoir : Status et contenu du projet

### 👨‍💻 Développeur Frontend
- Lire : `FRONTEND_SETUP.md`, `FRONTEND_README.md`
- Explorer : Structure dans `/front-end`
- Modifier : Composants et pages

### 🔧 DevOps/SysAdmin
- Lire : `SETUP_GUIDE.md`, `PREVIEW_OPTIMIZATION.md`
- Configurer : Déploiement, variables env
- Monitorer : Performance et logs

### 📊 Data Scientist
- Lire : `API_DOCUMENTATION.md`, `README_COMPLETE.md`
- Comprendre : Modèles ML, endpoints
- Intégrer : Nouveaux modèles

---

## 🚀 Checklist d'Utilisation

- [ ] Lire le document approprié pour votre cas
- [ ] Clôner/télécharger le repo
- [ ] Installer dépendances : `cd front-end && npm install`
- [ ] Démarrer : `npm run dev`
- [ ] Accéder : `http://localhost:3000`
- [ ] Tester : Utilisez les comptes de test
- [ ] Explorer : Les différentes pages et fonctionnalités
- [ ] Connecter backend (optionnel) : Lancez le backend FastAPI

---

## 📞 Support et Questions

### Pour Questions Générales
→ Consultez : `FRONTEND_README.md` ou `00_START_HERE.md`

### Pour Questions Techniques
→ Consultez : `FRONTEND_SETUP.md` ou `API_DOCUMENTATION.md`

### Pour Problèmes de Démarrage
→ Consultez : `PREVIEW_OPTIMIZATION.md` ou `SETUP_GUIDE.md`

### Pour Intégration Backend
→ Consultez : `API_DOCUMENTATION.md` ou `INTEGRATION_COMPLETE.md`

---

## 📈 Taille Totale de Documentation

```
Total des documents de doc : 2,358 lignes
Documents frontend spécifiques : 1,029 lignes
Documentation : 7 fichiers principaux
```

---

**✨ Vous êtes prêt à commencer !**

**Première étape** → Lire : `00_START_HERE.md`

**Puis** → Consulter ce guide (`DOCS_GUIDE.md`) pour naviguer vers le document approprié

**Enfin** → Lancer le projet : `cd front-end && npm run dev`

🚀 Bon développement !
