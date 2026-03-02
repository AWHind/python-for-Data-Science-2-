"use client";

import Image from "next/image";
import {
  Heart,
  Activity,
  Brain,
  Shield,
  ArrowRight,
  Sparkles,
  Users,
  BarChart3,
  Stethoscope,
  Zap,
  CheckCircle,
  ChevronRight,
} from "lucide-react";

interface PublicHomeProps {
  onGoToLogin: () => void;
  onGoToRegister: () => void;
}

export function PublicHome({ onGoToLogin, onGoToRegister }: PublicHomeProps) {
  return (
    <div className="min-h-screen bg-background">
      {/* Navbar */}
      <nav className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary">
              <Heart className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="text-lg font-bold tracking-wide text-foreground">
              CardioSense
            </span>
          </div>
          <div className="hidden items-center gap-6 md:flex">
            <a href="#features" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
              Fonctionnalites
            </a>
            <a href="#how-it-works" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
              Comment ca marche
            </a>
            <a href="#about" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
              A propos
            </a>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={onGoToLogin}
              className="rounded-xl border border-border bg-transparent px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-muted"
            >
              Connexion
            </button>
            <button
              onClick={onGoToRegister}
              className="rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90"
            >
              Inscription
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 sm:py-24 lg:px-8">
          <div className="flex flex-col items-center gap-12 lg:flex-row lg:items-start lg:gap-16">
            <div className="flex max-w-xl flex-1 flex-col items-center text-center lg:items-start lg:text-left">
              <div className="flex items-center gap-2 rounded-full border border-border bg-card px-4 py-1.5">
                <Sparkles className="h-3.5 w-3.5 text-primary" />
                <span className="text-xs font-medium text-muted-foreground">
                  Propulse par Intelligence Artificielle
                </span>
              </div>
              <h1 className="mt-6 font-serif text-4xl font-bold leading-tight tracking-tight text-foreground sm:text-5xl lg:text-6xl">
                <span className="text-balance">
                  Detectez les{" "}
                  <span className="text-primary">arythmies cardiaques</span>{" "}
                  avec precision
                </span>
              </h1>
              <p className="mt-5 text-base leading-relaxed text-muted-foreground sm:text-lg">
                CardioSense est une plateforme médicale d’aide à la décision, conçue pour
                assister les professionnels de santé dans l’analyse des données ECG et
                l’identification précoce des troubles du rythme cardiaque grâce à
                l’intelligence artificielle.
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <button
                  onClick={onGoToRegister}
                  className="flex items-center gap-2 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90"
                >
                  Accéder à la plateforme
                  <ArrowRight className="h-4 w-4" />
                </button>
                <button
                  onClick={onGoToLogin}
                  className="flex items-center gap-2 rounded-xl border border-border bg-card px-6 py-3 text-sm font-medium text-foreground transition-colors hover:bg-muted"
                >
                  En savoir plus
                </button>
              </div>
              <div className="mt-8 flex items-center gap-6">
                {[
                  { value: "452", label: "Patients analysés" },
                  { value: "96.2%", label: "Précision du modèle\n" },
                  { value: "16", label: "Classes d’arythmies détectées\n" },
                ].map((stat) => (
                  <div key={stat.label} className="text-center lg:text-left">
                    <p className="text-xl font-bold text-foreground">{stat.value}</p>
                    <p className="text-xs text-muted-foreground">{stat.label}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="relative w-full max-w-lg flex-1">
              <div className="relative aspect-[4/3] overflow-hidden rounded-3xl border border-border shadow-xl">
                <Image
                    src="/coeur.png"
                    alt="Analyse cardiaque par intelligence artificielle"
                    fill
                    className="object-cover"
                    priority


                />
              </div>
              {/* Floating card */}
              <div className="absolute -bottom-4 -left-4 rounded-2xl border border-border bg-card p-4 shadow-lg sm:-left-8">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-100">
                    <CheckCircle className="h-5 w-5 text-emerald-600" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-foreground">Rythme Normal</p>
                    <p className="text-xs text-muted-foreground">Confiance: 97.2%</p>
                  </div>
                </div>
              </div>
              <div className="absolute -right-2 -top-2 rounded-2xl border border-border bg-card p-3 shadow-lg sm:-right-6">
                <div className="flex items-center gap-2">
                  <Activity className="h-5 w-5 text-primary" />
                  <span className="text-sm font-bold text-foreground">72 bpm</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Image Gallery Section */}
      <section className="border-t border-border bg-card/50">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="font-serif text-3xl font-bold text-foreground">
              Technologie Medicale de Pointe
            </h2>
            <p className="mt-2 text-sm text-muted-foreground">
              Une plateforme integrant les dernieres avancees en cardiologie et intelligence artificielle
            </p>
          </div>
          <div className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <div className="group relative aspect-[4/3] overflow-hidden rounded-2xl border border-border">
              <Image
                src="/1.png"
                alt="Analyse ECG detaillee avec tracees electrocardiographiques"
                fill
                className="object-cover transition-transform duration-500 group-hover:scale-105"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-foreground/60 to-transparent" />
              <div className="absolute bottom-0 left-0 p-5">
                <p className="text-lg font-semibold text-primary-foreground">Analyse ECG</p>
                <p className="text-sm text-primary-foreground/80">
                  Lecture detaillee des electrocardiogrammes
                </p>
              </div>
            </div>
            <div className="group relative aspect-[4/3] overflow-hidden rounded-2xl border border-border">
              <Image
                src="/2.png"
                alt="Intelligence artificielle appliquee a la cardiologie"
                fill
                className="object-cover transition-transform duration-500 group-hover:scale-105"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-foreground/60 to-transparent" />
              <div className="absolute bottom-0 left-0 p-5">
                <p className="text-lg font-semibold text-primary-foreground">IA & Machine Learning</p>
                <p className="text-sm text-primary-foreground/80">
                  Classification intelligente des arythmies
                </p>
              </div>
            </div>
            <div className="group relative aspect-[4/3] overflow-hidden rounded-2xl border border-border sm:col-span-2 lg:col-span-1">
              <Image
                src="/36.png"
                alt="Equipe medicale specialisee en cardiologie"
                fill
                className="object-cover transition-transform duration-500 group-hover:scale-105"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-foreground/60 to-transparent" />
              <div className="absolute bottom-0 left-0 p-5">
                <p className="text-lg font-semibold text-primary-foreground">Equipe Medicale</p>
                <p className="text-sm text-primary-foreground/80">
                  Collaboration entre IA et professionnels de sante
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="border-t border-border">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="font-serif text-3xl font-bold text-foreground">
              Fonctionnalites Principales
            </h2>
            <p className="mt-2 text-sm text-muted-foreground">
              Un ecosysteme complet pour le diagnostic assiste des arythmies cardiaques
            </p>
          </div>
          <div className="mt-10 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {[
              {
                icon: Activity,
                iconBg: "bg-primary/10",
                iconColor: "text-primary",
                title: "Prediction en temps reel",
                desc: "Saisissez vos parametres ECG et obtenez une classification instantanee grace aux modeles de Machine Learning.",
              },
              {
                icon: Brain,
                iconBg: "bg-amber-500/10",
                iconColor: "text-amber-500",
                title: "Modeles ML avances",
                desc: "Random Forest, SVM, KNN, Naive Bayes - comparez les performances de plusieurs algorithmes.",
              },
              {
                icon: BarChart3,
                iconBg: "bg-sky-500/10",
                iconColor: "text-sky-500",
                title: "Analyse statistique",
                desc: "Explorez les donnees du dataset UCI Arrhythmia a travers des visualisations interactives.",
              },
              {
                icon: Users,
                iconBg: "bg-emerald-500/10",
                iconColor: "text-emerald-500",
                title: "Gestion des patients",
                desc: "Suivi complet des patients avec historique de predictions et tableau de bord personnalise.",
              },
              {
                icon: Shield,
                iconBg: "bg-foreground/10",
                iconColor: "text-foreground",
                title: "Securite et confidentialite",
                desc: "Authentification securisee avec roles (Patient/Administrateur) et protection des donnees medicales.",
              },
              {
                icon: Stethoscope,
                iconBg: "bg-primary/10",
                iconColor: "text-primary",
                title: "Aide a la decision",
                desc: "Outil complementaire pour les professionnels de sante, facilitant le diagnostic des arythmies.",
              },
            ].map((feature) => (
              <div
                key={feature.title}
                className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-6 transition-all hover:shadow-md"
              >
                <div className={`flex h-12 w-12 items-center justify-center rounded-xl ${feature.iconBg}`}>
                  <feature.icon className={`h-5 w-5 ${feature.iconColor}`} />
                </div>
                <h3 className="text-base font-semibold text-foreground">
                  {feature.title}
                </h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {feature.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="border-t border-border bg-card/50">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="font-serif text-3xl font-bold text-foreground">
              Comment ca fonctionne
            </h2>
            <p className="mt-2 text-sm text-muted-foreground">
              Un processus simple et efficace en trois etapes
            </p>
          </div>
          <div className="mt-10 grid grid-cols-1 gap-6 sm:grid-cols-3">
            {[
              {
                step: "01",
                icon: Users,
                title: "Creer votre compte",
                desc: "Inscrivez-vous en tant que patient. Votre compte sera valide par un administrateur pour garantir la securite.",
              },
              {
                step: "02",
                icon: Zap,
                title: "Saisir vos donnees ECG",
                desc: "Entrez vos parametres cardiaques dans le formulaire simplifie (age, frequence cardiaque, intervalles...).",
              },
              {
                step: "03",
                icon: CheckCircle,
                title: "Obtenir le resultat",
                desc: "Le modele IA analyse vos donnees et vous fournit la classification, le niveau de confiance et un rapport detaille.",
              },
            ].map((item, index) => (
              <div key={item.step} className="relative flex flex-col items-center text-center">
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
                  <item.icon className="h-7 w-7 text-primary" />
                </div>
                <span className="mt-4 text-3xl font-bold text-primary/20">{item.step}</span>
                <h3 className="mt-2 text-lg font-semibold text-foreground">{item.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{item.desc}</p>
                {index < 2 && (
                  <ChevronRight className="absolute -right-3 top-6 hidden h-6 w-6 text-border sm:block" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* About / Dataset Section */}
      <section id="about" className="border-t border-border">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="flex flex-col items-center gap-10 lg:flex-row lg:gap-16">
            <div className="flex-1">
              <h2 className="font-serif text-3xl font-bold text-foreground">
                A propos du dataset
              </h2>
              <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
                CardioSense repose sur le dataset UCI Arrhythmia, une reference
                dans le domaine de la classification des arythmies cardiaques.
                Ce jeu de donnees comprend 452 instances, 279 attributs et 16
                classes differentes, permettant un entrainement robuste des
                modeles de Machine Learning.
              </p>
              <div className="mt-6 grid grid-cols-2 gap-4">
                {[
                  { value: "452", label: "Instances", icon: Users },
                  { value: "279", label: "Attributs", icon: BarChart3 },
                  { value: "16", label: "Classes", icon: Activity },
                  { value: "96.2%", label: "Precision", icon: Brain },
                ].map((stat) => (
                  <div
                    key={stat.label}
                    className="flex items-center gap-3 rounded-2xl border border-border bg-card p-4"
                  >
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-muted">
                      <stat.icon className="h-4 w-4 text-foreground" />
                    </div>
                    <div>
                      <p className="text-lg font-bold text-foreground">{stat.value}</p>
                      <p className="text-xs text-muted-foreground">{stat.label}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="relative w-full max-w-md flex-1">
              <div className="relative aspect-square overflow-hidden rounded-3xl border border-border shadow-lg">
                <Image
                  src="/1.png"
                  alt="Electrocardiogramme analyse en detail"
                  fill
                  className="object-cover"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section className="border-t border-border">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="overflow-hidden rounded-3xl bg-sidebar p-8 sm:p-12 lg:p-16">
            <div className="mx-auto max-w-2xl text-center">
              <h2 className="font-serif text-3xl font-bold text-sidebar-foreground sm:text-4xl">
          <span className="text-balance">
            Contactez la Clinique CardioSense
          </span>
              </h2>

              <p className="mt-4 text-sm leading-relaxed text-sidebar-foreground/70">
                Pour toute information complémentaire ou assistance médicale, notre
                équipe est à votre écoute. N’hésitez pas à nous contacter via les
                coordonnées ci-dessous.
              </p>

              <div className="mt-8 space-y-3 text-sm text-sidebar-foreground">
                <p>
                  <span className="font-semibold">Adresse :</span>{" "}
                  Avenue de la Santé, 1000 Tunis, Tunisie
                </p>

                <p>
                  <span className="font-semibold">Téléphone :</span>{" "}
                  +216 70 000 000
                </p>

                <p>
                  <span className="font-semibold">Email :</span>{" "}
                  <a
                      href="mailto:contact@cardiosense.tn"
                      className="text-primary hover:underline"
                  >
                    contact@cardiosense.tn
                  </a>
                </p>

                <p>
                  <span className="font-semibold">Horaires :</span>{" "}
                  Lundi – Vendredi : 08h00 – 17h00
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>


      {/* Footer */}
      <footer className="border-t border-border">
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
          <div className="flex flex-col items-center justify-between gap-4 sm:flex-row">
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
                <Heart className="h-4 w-4 text-primary-foreground" />
              </div>
              <span className="text-sm font-bold text-foreground">CardioSense</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Cet outil constitue une aide a la decision medicale et ne remplace en aucun cas un diagnostic clinique.
            </p>
            <p className="text-xs text-muted-foreground">
              2026 CardioSense - UCI Arrhythmia Dataset
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
