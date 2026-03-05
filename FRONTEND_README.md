# CardioSense - Frontend

Bienvenue dans le frontend de **CardioSense**, une application web intelligente pour l'analyse des arythmies cardiaques utilisant le Machine Learning.

## 🚀 Démarrage Rapide (2 minutes)

### 1️⃣ Installation des dépendances

```bash
cd front-end
npm install
```

### 2️⃣ Démarrage du serveur de développement

```bash
npm run dev
```

### 3️⃣ Accès à l'application

Ouvrez votre navigateur et allez à : **http://localhost:3000**

---

## 📚 Comptes de Démonstration

Vous pouvez vous connecter avec ces comptes pré-créés :

### Patient
```
Email : patient@cardiosense.com
Mot de passe : patient123
```

### Admin
```
Email : admin@cardiosense.com
Mot de passe : admin123
```

---

## 📁 Structure du Projet

```
front-end/
├── app/
│   ├── layout.tsx           # Layout racine + Authentification
│   ├── page.tsx             # Page principale (routage basé sur auth)
│   └── globals.css          # Design tokens et styles globaux
├── components/
│   ├── admin/               # Composants du dashboard admin
│   ├── patient/             # Composants du dashboard patient
│   ├── dashboard/           # Composants partagés
│   ├── ui/                  # Composants Shadcn/ui (boutons, cartes, etc.)
│   ├── login-page.tsx       # Page de connexion
│   ├── register-page.tsx    # Page d'inscription
│   ├── public-home.tsx      # Page d'accueil publique
│   └── theme-provider.tsx   # Provider de thème
├── lib/
│   ├── auth-context.tsx     # Contexte d'authentification
│   └── utils.ts             # Fonctions utilitaires
├── public/                  # Images et assets statiques
├── hooks/                   # Hooks personnalisés React
├── styles/                  # Styles CSS supplémentaires
├── package.json             # Dépendances npm
├── tsconfig.json            # Configuration TypeScript
├── tailwind.config.ts       # Configuration Tailwind CSS
├── postcss.config.mjs       # Configuration PostCSS
└── next.config.mjs          # Configuration Next.js
```

---

## 🎯 Pages Principales

### 1. Page d'Accueil Publique (`/`)
- Affichée si vous n'êtes pas connecté
- Présente les fonctionnalités de CardioSense
- Boutons "Connexion" et "Inscription"

### 2. Page Connexion
- Formulaire simple avec email et mot de passe
- Gestion du statut d'approbation
- Redirection automatique après connexion

### 3. Page Inscription
- Formulaire d'inscription complet
- Validation des données
- Demande d'approbation par l'admin

### 4. Dashboard Patient
- Accessible après connexion en tant que patient
- Contient :
  - Formulaire de prédiction ECG
  - Historique des prédictions
  - Statistiques personnelles
  - Graphiques et charts

### 5. Dashboard Admin
- Accessible après connexion en tant que admin
- Contient :
  - Vue d'ensemble des statistiques
  - Gestion des patients
  - Comparaison des modèles ML
  - Gestion des inscriptions en attente
  - Paramètres du système

---

## 🔧 Scripts disponibles

```bash
# Démarrer le serveur de développement
npm run dev              # Avec turbo (plus rapide)
npm run dev:fast        # Sans turbo

# Build pour la production
npm run build

# Démarrer le serveur de production
npm start

# Linter le code
npm run lint

# Aperçu complet (build + start)
npm run preview
```

---

## 🎨 Design et Styling

### Technologies utilisées
- **Tailwind CSS 4** : Utility-first CSS framework
- **Shadcn/ui** : Composants réutilisables de haute qualité
- **Lucide React** : Icônes modernes et cohérentes

### Couleurs du design

| Couleur | Code HSL | Utilisation |
|---------|----------|------------|
| **Primary** | `350 70% 55%` | Boutons, highlights, liens |
| **Secondary** | `30 15% 92%` | Backgrounds clairs |
| **Accent** | `350 65% 50%` | Accents, hover states |
| **Background** | `30 20% 96%` | Fond général |
| **Foreground** | `220 15% 15%` | Texte principal |
| **Sidebar** | `220 15% 15%` | Navigation et sidebars |

### Polices
- **Inter** : Pour le body text et texte général
- **Playfair Display** : Pour les headings et titres

---

## 🔗 Intégration Backend

Le frontend communique avec le backend ML via des appels API REST.

### Points d'intégration

#### Page de Prédiction
```
GET  http://127.0.0.1:8000/health          # Vérifier la connexion
GET  http://127.0.0.1:8000/models          # Charger les modèles
POST http://127.0.0.1:8000/predict         # Faire une prédiction
POST http://127.0.0.1:8000/report          # Générer un PDF
```

#### Admin Dashboard
```
GET  http://127.0.0.1:8000/models                    # Charger les modèles
GET  http://127.0.0.1:8000/models/{name}/metrics    # Métriques d'un modèle
```

### Configuration API

Le backend doit être lancé sur le port **8000** :
```bash
cd app
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Les adresses `localhost` et `127.0.0.1` sont autorisées par CORS.

---

## 🔐 Authentification

L'authentification est gérée côté client avec **React Context**.

### Fonctionnalités
- ✅ Connexion / Déconnexion
- ✅ Inscription avec validation
- ✅ Rôles utilisateur (Patient / Admin)
- ✅ Statuts de compte (actif / en attente / rejeté)
- ✅ Approbation des inscriptions par l'admin

### Utilisateurs pré-créés

**Patient**
- Email : `patient@cardiosense.com`
- Mot de passe : `patient123`
- Statut : Actif

**Admin**
- Email : `admin@cardiosense.com`
- Mot de passe : `admin123`
- Statut : Actif

### Fichier clé
`lib/auth-context.tsx` - Gère toute la logique d'authentification

---

## 📦 Dépendances Principales

### Framework & Runtime
- `next@16.1.6` - Framework React avec SSR
- `react@18.3.1` - Bibliothèque UI
- `typescript@5.7.3` - Type safety

### UI Components & Styling
- `tailwindcss@3.4.17` - CSS framework
- `shadcn/ui` - Composants accessibles
- `lucide-react@0.544.0` - Icônes
- `next-themes@0.4.6` - Gestion du thème

### Forms & Validation
- `react-hook-form@7.54.1` - Gestion de formulaires
- `@hookform/resolvers@3.9.1` - Résolveurs pour validation
- `zod@3.24.1` - Validation de schémas

### Data & Charts
- `recharts@2.15.0` - Graphiques et charts
- `date-fns@3.6.0` - Utilitaires date

### Autres
- `cmdk@1.1.1` - Command menu
- `sonner@1.7.1` - Notifications toast
- `vaul@1.1.2` - Drawer component
- `embla-carousel-react@8.5.1` - Carousel

---

## 🐛 Dépannage

### Le serveur ne démarre pas

**Error**: `Error: listen EADDRINUSE :::3000`

**Solution**: Le port 3000 est déjà utilisé. Tuez le processus ou utilisez un autre port:
```bash
npm run dev -- --port 3001
```

### Connexion API échoue

**Vérifiez**:
1. Backend lancé sur `http://127.0.0.1:8000`
2. Vérifiez l'URL dans le code (`const API_URL = "http://127.0.0.1:8000"`)
3. Accédez à `http://127.0.0.1:8000/docs` pour vérifier que l'API répond

### Build échoue avec erreurs TypeScript

```bash
npm run build          # Vérifier les erreurs
npm run lint           # Checker les erreurs de linting
```

### Module non trouvé

```bash
# Réinstaller les dépendances
rm -rf node_modules pnpm-lock.yaml
npm install
```

---

## 📱 Responsivité

Le design est **mobile-first** et responsive sur tous les appareils:
- **Mobile** : 320px+
- **Tablet** : 768px+
- **Desktop** : 1024px+

Tous les composants utilisent les breakpoints Tailwind (`sm:`, `md:`, `lg:`, etc.)

---

## 🚀 Déploiement sur Vercel

### 1. Connecter le repo GitHub

```bash
git push origin main
```

### 2. Sur Vercel Dashboard

- Importer le repo
- Configuration automatique (Next.js détecté)
- Variables d'environnement (si nécessaire)
- Déployer

### 3. Build Command (automatique)

```
cd front-end && npm run build
```

---

## 🔄 Mise à Jour des Dépendances

```bash
npm outdated           # Voir les versions disponibles
npm update             # Mettre à jour les dépendances
npm install @latest   # Installer les latest versions
```

---

## 📖 Ressources

- **Next.js Docs** : https://nextjs.org/docs
- **React Docs** : https://react.dev
- **Tailwind CSS** : https://tailwindcss.com
- **Shadcn/ui** : https://ui.shadcn.com
- **Lucide Icons** : https://lucide.dev

---

## 💡 Tips & Tricks

### Hot Reload
Les changements dans le code sont appliqués en temps réel pendant le développement.

### DevTools
Inspectez les composants React avec le React Developer Tools browser extension.

### Performance
Pour vérifier les performances:
```bash
npm run build          # Vérifier la taille du build
npm start              # Tester en production
```

---

## 📞 Support

Pour les problèmes avec le frontend, consultez :
1. **FRONTEND_SETUP.md** - Documentation technique détaillée
2. **DOCUMENTATION_INDEX.md** - Index de tous les documents
3. **00_START_HERE.md** - Guide de démarrage global

---

**Prêt à lancer ?** 🚀

```bash
cd front-end
npm install
npm run dev
```

L'application ouvrira automatiquement sur **http://localhost:3000** 🎉
