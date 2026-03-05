# Configuration du Frontend - CardioSense

## Vue d'ensemble

Le frontend est une application **Next.js 16** avec **React 18**, utilisant :
- **Framework** : Next.js 16 (App Router)
- **Styling** : Tailwind CSS 4
- **UI Components** : Shadcn/ui
- **Icons** : Lucide React
- **Authentification** : Context API (client-side)
- **Type Safety** : TypeScript

## Structure du projet

```
front-end/
├── app/
│   ├── layout.tsx          # Root layout avec AuthProvider
│   ├── page.tsx            # Page principale (routage basé sur auth)
│   └── globals.css         # Design tokens et styles globaux
├── components/
│   ├── admin/              # Composants admin (dashboard, modèles, stats)
│   ├── patient/            # Composants patient
│   ├── dashboard/          # Composants dashboard partagés
│   ├── ui/                 # Composants Shadcn/ui
│   ├── login-page.tsx      # Page de connexion
│   ├── register-page.tsx   # Page d'inscription
│   ├── public-home.tsx     # Page d'accueil publique
│   ├── chatbot.tsx         # Chatbot assistant
│   └── theme-provider.tsx  # Provider de thème
├── lib/
│   ├── auth-context.tsx    # Contexte d'authentification
│   └── utils.ts            # Fonctions utilitaires
├── public/                 # Assets statiques (images, logos)
├── hooks/                  # Hooks personnalisés
├── package.json            # Dépendances npm
├── tsconfig.json           # Configuration TypeScript
├── tailwind.config.ts      # Configuration Tailwind
├── postcss.config.mjs      # Configuration PostCSS
└── next.config.mjs         # Configuration Next.js

```

## Démarrage du Frontend

### Option 1 : Démarrage simple

```bash
cd front-end
npm install
npm run dev
```

Le frontend sera accessible à `http://localhost:3000`

### Option 2 : Démarrage optimisé

```bash
cd front-end
npm run dev:fast
```

### Option 3 : Mode production

```bash
cd front-end
npm run build
npm start
```

## Pages et Routes

### Page d'Accueil Publique (`/`)

Affichée par défaut si l'utilisateur n'est pas authentifié. Contient :
- Navigation avec boutons Login/Register
- Section Hero avec présentation
- Galerie d'images (technologie médicale)
- Section fonctionnalités (6 cartes)
- Guide "Comment ça marche" (3 étapes)
- Section "À propos du dataset"
- Section contact
- Footer avec liens

**Composant** : `components/public-home.tsx` (460 lignes)

### Compte de Démonstration

#### Patient
- **Email** : `patient@cardiosense.com`
- **Mot de passe** : `patient123`
- **Rôle** : Patient
- **Statut** : Actif

#### Admin
- **Email** : `admin@cardiosense.com`
- **Mot de passe** : `admin123`
- **Rôle** : Admin
- **Statut** : Actif

### Page Connexion

- Formulaire de connexion par email/mot de passe
- Gestion des statuts (actif, en attente, rejeté)
- Redirection automatique vers le dashboard approprié

**Composant** : `components/login-page.tsx`

### Page Inscription

- Formulaire d'inscription avec validation
- Demande d'approbation par l'admin
- Vérification des doublons email

**Composant** : `components/register-page.tsx`

### Dashboard Patient

Affiche après connexion en tant que patient. Contient :
- Informations du patient
- Formulaire de prédiction (connecté au backend)
- Historique des prédictions
- Statistiques personnelles

**Composant** : `components/patient/patient-dashboard.tsx`

### Dashboard Admin

Affiche après connexion en tant que admin. Contient :
- Sidebar de navigation
- Vue d'ensemble des statistiques
- Gestion des patients
- Comparaison des modèles ML
- Performance des modèles
- Gestion des validations

**Composant** : `components/admin/admin-dashboard-layout.tsx`

## Configuration Authentification

### AuthContext

Stocke en local (client-side) :
- Utilisateurs (email, mot de passe, profil)
- Demandes d'inscription en attente
- Rôles et statuts

**Fichier** : `lib/auth-context.tsx`

Utilisateurs pré-créés pour démo :
- Patient : patient@cardiosense.com / patient123
- Admin : admin@cardiosense.com / admin123

### Flux d'authentification

1. Utilisateur non authentifié → Page d'accueil publique
2. Clique sur "Connexion" ou "Inscription"
3. Après authentification :
   - Admin → Dashboard Admin
   - Patient → Dashboard Patient
4. Bouton "Déconnexion" dans chaque dashboard

## Intégration Backend

### Configuration API

L'API backend est accessible à `http://127.0.0.1:8000`

### Points de connexion

**Dans `components/dashboard/prediction-form.tsx` :**
- Charge la liste des modèles : `GET /models`
- Fait une prédiction : `POST /predict`
- Vérifie la santé : `GET /health`
- Génère un rapport : `POST /report`

**Dans `components/admin/admin-models.tsx` :**
- Charge les modèles : `GET /models`

### Variables d'environnement

Le frontend peut utiliser (optionnel) :
```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

## Design System

### Tokens de Couleurs (globals.css)

```css
--primary: 350 70% 55%         /* Rose/Magenta */
--secondary: 30 15% 92%        /* Beige clair */
--accent: 350 65% 50%          /* Rose accent */
--background: 30 20% 96%       /* Beige */
--foreground: 220 15% 15%      /* Bleu foncé */
--sidebar-background: 220 15% 15%  /* Bleu très foncé */
--sidebar-foreground: 220 10% 85%  /* Bleu clair */
```

### Polices

- **Sans-serif** : Inter (body text)
- **Serif** : Playfair Display (headings)

Appliquées avec les classes Tailwind `font-sans` et `font-serif`

## Composants Principaux

### Composants Admin
- `AdminDashboardLayout` - Conteneur principal avec sidebar
- `AdminDashboard` - Vue d'ensemble
- `AdminModels` - Comparaison des modèles ML
- `AdminPatients` - Gestion des patients
- `AdminPerformance` - Métriques de performance
- `AdminSettings` - Paramètres
- `AdminSidebar` - Navigation
- `AdminStatistics` - Statistiques globales
- `AdminValidations` - Gestion des inscriptions

### Composants Dashboard
- `PredictionForm` - Formulaire de prédiction (connecté à l'API)
- `AdminView` - Vue admin du dashboard
- `ClientView` - Vue patient du dashboard
- `HeartRateChart` - Graphique du rythme cardiaque
- `ModelPerformance` - Performance des modèles
- `MetricCards` - Cartes de métriques
- `RecentActivity` - Activité récente
- `PatientsTable` - Tableau des patients

### Composants Patient
- `PatientDashboard` - Dashboard patient complet
- `PatientHome` - Page d'accueil patient
- `PatientPrediction` - Interface de prédiction
- `PatientHistory` - Historique des prédictions
- `PatientSidebar` - Navigation patient

## Problèmes Courants et Solutions

### Frontend ne démarre pas

**Problème** : `npm run dev` échoue
**Solution** :
```bash
cd front-end
rm -rf node_modules pnpm-lock.yaml
npm install
npm run dev
```

### Port 3000 déjà utilisé

**Solution** : Le script utilise `--hostname 0.0.0.0` pour accepter les connexions externes

### Connexion API échoue

**Vérifiez** :
1. Backend lancé : `http://127.0.0.1:8000/docs`
2. CORS configuré dans le backend
3. URL API correcte dans le code (http://127.0.0.1:8000)

### Build échoue

**Solution** :
```bash
npm run lint           # Vérifier les erreurs TypeScript
npm run build          # Rebuilder
```

## Performance

### Optimisations appliquées

1. **Image Optimization** : Images non optimisées (unoptimized: true)
2. **Code Splitting** : Automatique avec Next.js
3. **CSS** : Tailwind avec optimisation de production
4. **JavaScript** : SWC minifier activé
5. **TypeScript** : Compilation TS optimisée

### Taille du build

- **Dev** : ~200-300 MB (avec node_modules)
- **Build** : ~50-80 MB (optimisé)

## Déploiement

### Sur Vercel

1. Connecter le repo GitHub
2. Configuration :
   - **Framework** : Next.js
   - **Build command** : `cd front-end && npm run build`
   - **Output directory** : `front-end/.next`
   - **Install command** : `npm install` (root)

3. Déployer

Le fichier `vercel.json` à la racine configure automatiquement.

### Localement

```bash
cd front-end
npm run build
npm start
```

## Support Navigateurs

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Fichiers importants

| Fichier | Utilité |
|---------|---------|
| `app/layout.tsx` | Root layout + AuthProvider |
| `app/page.tsx` | Routage principal (logique d'authentification) |
| `app/globals.css` | Design tokens + Tailwind |
| `lib/auth-context.tsx` | Gestion authentification |
| `components/public-home.tsx` | Page d'accueil publique |
| `components/login-page.tsx` | Formulaire connexion |
| `components/register-page.tsx` | Formulaire inscription |
| `components/patient/patient-dashboard.tsx` | Dashboard patient |
| `components/admin/admin-dashboard-layout.tsx` | Dashboard admin |
| `components/dashboard/prediction-form.tsx` | Formulaire prédiction (avec API) |
| `next.config.mjs` | Configuration Next.js |
| `tailwind.config.ts` | Configuration Tailwind |

---

**Prêt à démarrer ?** → `cd front-end && npm run dev`
