# Guide d'Optimisation du Preview - v0

Ce document explique comment accéder et utiliser le preview du frontend dans v0.

## 🎯 Accès au Preview

### Méthode 1 : Version Box dans v0

1. Cliquez sur le **Version Box** en haut du chat v0
2. Vous verrez l'aperçu du frontend en live
3. Le preview se met à jour automatiquement avec chaque changement de code

### Méthode 2 : Lancer localement

```bash
# Terminal 1 - Frontend
cd front-end
npm install
npm run dev

# Terminal 2 (optionnel) - Backend
cd app
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Accédez à : `http://localhost:3000`

---

## 🔧 Configuration Optimisée pour v0

### Next.js Configuration

Le fichier `next.config.mjs` est optimisé pour :
- ✅ Démarrage rapide
- ✅ Hot Module Replacement (HMR)
- ✅ Images non optimisées (plus rapide)
- ✅ Compression désactivée en dev

```javascript
// next.config.mjs
const nextConfig = {
  typescript: { ignoreBuildErrors: true },
  images: { unoptimized: true },
  swcMinify: true,
  compress: false,  // Désactivé pour dev
}
```

### Package.json Scripts

```json
{
  "scripts": {
    "dev": "next dev --turbo --port 3000 --hostname 0.0.0.0",
    "dev:fast": "next dev --port 3000 --hostname 0.0.0.0",
    "build": "next build",
    "start": "next start --port 3000 --hostname 0.0.0.0",
    "preview": "npm run build && npm start"
  }
}
```

---

## 📊 Performance du Preview

### Temps de Démarrage

| Mode | Temps | Notes |
|------|-------|-------|
| **Dev (turbo)** | 5-8 sec | Recommandé pour v0 |
| **Dev (normal)** | 8-12 sec | Fallback |
| **Production** | 2-3 sec | Après build |

### Optimisations Appliquées

1. **Turbopack** : Bundler Next.js ultra-rapide
2. **SWC** : Compilateur Rust (10x plus rapide que Babel)
3. **Module caching** : Les modules sont mis en cache
4. **Fast refresh** : HMR optimisé pour React

---

## 🚀 Commandes de Démarrage

### 1️⃣ Démarrage Standard (dans v0)

```bash
cd front-end && npm install && npm run dev
```

Avantages:
- Démarre en 5-8 secondes
- Parfait pour le preview v0
- Hot reload activé

### 2️⃣ Démarrage Ultra-Rapide

```bash
cd front-end && npm run dev:fast
```

Avantages:
- Pas de turbopack (un peu plus lent)
- Plus de stabilité
- Utilisez si dev:turbo pose problème

### 3️⃣ Mode Production

```bash
cd front-end && npm run build && npm start
```

Avantages:
- Démarrage ultra-rapide (1-2 sec)
- Bundle optimisé
- Parfait pour tester la version finale

---

## 📱 Preview Responsive

### Breakpoints Tailwind

Le design utilise les breakpoints standard de Tailwind:

```
sm: 640px   (tablets)
md: 768px   (small laptops)
lg: 1024px  (laptops)
xl: 1280px  (desktops)
```

### Test de Responsivité dans v0

1. Redimensionnez le preview
2. Vérifiez que le layout s'ajuste
3. Testez le menu mobile (768px+)

---

## 🔄 Auto-refresh en Développement

### Hot Module Replacement (HMR)

Activé automatiquement. Les changements se reflètent en temps réel:

1. **Changement CSS** : Refresh 100ms
2. **Changement JSX** : Refresh 200-500ms
3. **Changement TypeScript** : Refresh 500ms

### Désactiver HMR (si problèmes)

```bash
NEXT_TELEMETRY_DISABLED=1 npm run dev -- --no-experimental-app-edge
```

---

## 🌙 Thème et Design

### Système de Couleurs

Les couleurs sont définies dans `app/globals.css` avec des variables CSS:

```css
:root {
  --primary: 350 70% 55%;        /* Rose */
  --background: 30 20% 96%;      /* Beige clair */
  --foreground: 220 15% 15%;     /* Bleu foncé */
  --sidebar-background: 220 15% 15%;  /* Bleu */
}
```

### Changer les Couleurs

Éditez `app/globals.css` pour personnaliser les couleurs du design.

---

## 🧪 Test des Pages

### Navigation par URL

Vous pouvez accéder directement aux pages :

```
http://localhost:3000/          # Page d'accueil publique
(Puis utilisez les boutons pour naviguer)
```

### Comptes de Test

**Patient**
```
Email : patient@cardiosense.com
Password : patient123
```

**Admin**
```
Email : admin@cardiosense.com
Password : admin123
```

---

## 📊 Monitorer les Performances

### Dans le Browser

1. Ouvrez **DevTools** (F12)
2. Onglet **Network** : Vérifiez le temps de chargement
3. Onglet **Performance** : Profiling détaillé

### Fichiers à Monitorer

- **next.js bundle** : 150-200 KB (gzippé)
- **CSS total** : 50-80 KB (gzippé)
- **Initial page load** : 1-2 secondes

---

## 🔗 Intégration Backend

### Activer l'API Backend

Pour que le formulaire de prédiction fonctionne, lancez le backend:

```bash
# Terminal séparé
cd app
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Vérifier la Connexion

1. Allez au dashboard patient/admin
2. Accédez à "Prediction Form"
3. Vérifiez que le statut affiche "Connected"

---

## 🐛 Troubleshooting

### Le preview est blanc

**Cause** : Erreur JavaScript
**Solution** :
1. Ouvrez la console (F12)
2. Cherchez les erreurs rouges
3. Consultez `front-end/package.json` pour les dépendances manquantes

### Le preview ne charge pas

**Cause** : Port occupé ou build échoué
**Solution** :
```bash
cd front-end
rm -rf .next node_modules
npm install
npm run dev
```

### Hot reload ne fonctionne pas

**Cause** : Cache ou HMR désactivé
**Solution** :
```bash
# Force un refresh complet
npm run dev -- --no-cache
```

### Erreur "Module not found"

**Solution** :
```bash
npm install
npm run build
```

---

## 📈 Optimisations Futures

### Possibilités d'amélioration

1. **Image Optimization** : Utiliser `next/image` avec compression
2. **Code Splitting** : Importer composants dynamiquement avec `next/dynamic`
3. **Caching** : Ajouter `revalidateTag()` pour ISR
4. **Compression** : Activer Brotli en production
5. **Monitoring** : Intégrer Sentry pour les erreurs

---

## 📞 Support

- **Problème HMR** : Vérifiez `next.config.mjs`
- **Problème performance** : Vérifiez l'onglet Network
- **Problème API** : Vérifiez que le backend répond sur port 8000
- **Problème build** : Vérifiez `npm run lint`

---

## ✅ Checklist Démarrage

- [ ] Clôné le repo v0
- [ ] Installé Node.js 18+
- [ ] `cd front-end && npm install`
- [ ] `npm run dev`
- [ ] Accédé à `http://localhost:3000`
- [ ] Vu la page d'accueil publique
- [ ] Testé la connexion (patient@cardiosense.com)
- [ ] Navigué vers le dashboard

---

**✨ Le frontend est prêt pour le preview dans v0 !**

Cliquez sur le Version Box en haut pour voir l'aperçu en live. 🚀
