"use client";

import React from "react"

import {
  Heart,
  Activity,
  Brain,
  Shield,
  ArrowRight,
  Sparkles,
  FileBarChart,
  Clock,
} from "lucide-react";
import { useAuth } from "@/lib/auth-context";

interface FeatureCardProps {
  icon: React.ReactNode;
  iconBg: string;
  title: string;
  description: string;
  onClick?: () => void;
}

function FeatureCard({ icon, iconBg, title, description, onClick }: FeatureCardProps) {
  return (
    <button
      onClick={onClick}
      className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-6 text-left transition-all hover:shadow-md"
    >
      <div
        className={`flex h-12 w-12 items-center justify-center rounded-xl ${iconBg}`}
      >
        {icon}
      </div>
      <div>
        <h3 className="text-base font-semibold text-foreground">{title}</h3>
        <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
          {description}
        </p>
      </div>
      <span className="flex items-center gap-1 text-sm font-medium text-primary">
        Acceder <ArrowRight className="h-3.5 w-3.5" />
      </span>
    </button>
  );
}

export function PatientHome({ onNavigate }: { onNavigate: (page: number) => void }) {
  const { user } = useAuth();

  return (
    <div className="flex flex-col gap-6">
      {/* Welcome Banner */}
      <div className="relative overflow-hidden rounded-3xl bg-sidebar p-8 lg:p-10">
        <div className="relative z-10 max-w-xl">
          <p className="text-sm font-medium text-sidebar-foreground/60">
            Bienvenue sur CardioSense
          </p>
          <h1 className="mt-2 font-serif text-3xl font-semibold text-sidebar-foreground lg:text-4xl">
            Bonjour, {user?.name || "Patient"}
          </h1>
          <p className="mt-3 text-sm leading-relaxed text-sidebar-foreground/70">
            Cette application vous permet d{"'"}analyser vos donnees cardiaques et
            d{"'"}obtenir une prediction sur la presence ou l{"'"}absence d{"'"}une arythmie
            cardiaque grace a l{"'"}intelligence artificielle.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <button
              onClick={() => onNavigate(1)}
              className="flex items-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90"
            >
              <Activity className="h-4 w-4" />
              Nouvelle Prediction
            </button>
            <button
              onClick={() => onNavigate(2)}
              className="flex items-center gap-2 rounded-xl bg-sidebar-accent px-5 py-2.5 text-sm font-medium text-sidebar-foreground transition-colors hover:bg-sidebar-accent/80"
            >
              <Clock className="h-4 w-4" />
              Mon Historique
            </button>
          </div>
        </div>
        {/* Decorative elements */}
        <div className="absolute -right-10 -top-10 h-48 w-48 rounded-full bg-primary/10 blur-3xl" />
        <div className="absolute -bottom-6 right-20 h-32 w-32 rounded-full bg-primary/5 blur-2xl" />
        <Heart className="absolute bottom-6 right-8 h-24 w-24 text-sidebar-foreground/5" />
      </div>

      {/* How it Works */}
      <div>
        <h2 className="text-lg font-semibold text-foreground">
          Comment ca fonctionne
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Trois etapes simples pour analyser vos donnees cardiaques
        </p>

        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-3">
          {[
            {
              step: "01",
              title: "Saisie des donnees",
              desc: "Entrez vos parametres ECG dans le formulaire simplifie (age, frequence cardiaque, intervalles...)",
            },
            {
              step: "02",
              title: "Analyse par IA",
              desc: "Le modele de Machine Learning (Random Forest, SVM...) analyse vos donnees en temps reel.",
            },
            {
              step: "03",
              title: "Resultat",
              desc: "Obtenez la classe predite, le niveau de confiance et un resume detaille de l'analyse.",
            },
          ].map((item) => (
            <div
              key={item.step}
              className="flex flex-col gap-3 rounded-2xl border border-border bg-card p-5"
            >
              <span className="text-2xl font-bold text-primary/30">
                {item.step}
              </span>
              <h3 className="text-sm font-semibold text-foreground">
                {item.title}
              </h3>
              <p className="text-xs leading-relaxed text-muted-foreground">
                {item.desc}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Feature Cards */}
      <div>
        <h2 className="text-lg font-semibold text-foreground">
          Fonctionnalites disponibles
        </h2>
        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <FeatureCard
            icon={<Activity className="h-5 w-5 text-primary" />}
            iconBg="bg-primary/10"
            title="Prediction d'arythmie"
            description="Saisissez vos donnees medicales et obtenez une prediction instantanee sur votre rythme cardiaque."
            onClick={() => onNavigate(1)}
          />
          <FeatureCard
            icon={<FileBarChart className="h-5 w-5 text-sky-500" />}
            iconBg="bg-sky-500/10"
            title="Historique des analyses"
            description="Consultez l'ensemble de vos predictions precedentes avec dates, resultats et niveaux de confiance."
            onClick={() => onNavigate(2)}
          />
          <FeatureCard
            icon={<Brain className="h-5 w-5 text-amber-500" />}
            iconBg="bg-amber-500/10"
            title="IA de pointe"
            description="Modeles entraines sur le dataset UCI Arrhythmia (452 instances, 279 attributs, 16 classes)."
          />
        </div>
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-3 rounded-2xl border border-border bg-card p-5">
        <Shield className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
        <div>
          <h3 className="text-sm font-semibold text-foreground">
            Avertissement medical
          </h3>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
            Cette application constitue un outil d{"'"}aide a la decision medicale et
            ne remplace en aucun cas un diagnostic clinique. Les resultats fournis
            sont bases sur des modeles de Machine Learning entraines sur le
            dataset UCI Arrhythmia et doivent etre interpretes par un professionnel
            de sante qualifie.
          </p>
        </div>
      </div>

      {/* Stats bar */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {[
          { icon: Heart, label: "Patients analyses", value: "452" },
          { icon: Sparkles, label: "Attributs", value: "279" },
          { icon: Brain, label: "Precision", value: "96.2%" },
          { icon: Activity, label: "Classes", value: "16" },
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
  );
}
