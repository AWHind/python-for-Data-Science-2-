"use client";

import { BarChart3, TrendingUp, Heart, Sparkles } from "lucide-react";

export function DatasetStats() {
  return (
    <div className="flex flex-col justify-between rounded-2xl border border-border bg-card p-5">
      <div className="flex items-center gap-2">
        <BarChart3 className="h-5 w-5 text-foreground" />
        <span className="text-sm font-semibold text-foreground">
          Dataset UCI
        </span>
      </div>

      <div className="mt-3 flex items-baseline gap-1.5">
        <span className="text-2xl font-bold text-foreground">452</span>
        <span className="text-sm text-muted-foreground">instances</span>
        <span className="ml-2 text-xs font-medium text-emerald-600">
          <TrendingUp className="mr-0.5 inline h-3 w-3" />
          279 attributs
        </span>
      </div>

      <div className="mt-3 flex gap-0.5">
        {Array.from({ length: 20 }).map((_, i) => (
          <div
            key={`bar-${i}`}
            className="w-2 rounded-sm bg-primary/70"
            style={{
              height: `${Math.random() * 24 + 8}px`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

export function HealthMessage() {
  return (
    <div className="flex flex-col justify-between rounded-2xl border border-border bg-card p-5">
      <div>
        <h3 className="text-balance text-lg font-semibold leading-snug text-foreground">
          La sante cardiaque, un enjeu vital.
        </h3>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
          Classification automatique des arythmies pour un diagnostic plus
          rapide et plus precis.
        </p>
      </div>

      <div className="mt-4 flex items-center justify-between">
        <button className="rounded-full bg-foreground px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-90">
          Analyser un ECG
        </button>
        <Heart className="h-12 w-12 text-primary/30" />
      </div>
    </div>
  );
}

export function AIPoweredPanel() {
  return (
    <div className="relative overflow-hidden rounded-2xl bg-foreground p-5 text-background">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4" />
          <span className="text-sm font-semibold">IA Classification</span>
        </div>
        <button className="text-background/60 transition-colors hover:text-background">
          ...
        </button>
      </div>

      <div className="mt-4 flex flex-col gap-3">
        <div className="relative overflow-hidden rounded-xl bg-background/10 p-3">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-full bg-primary/30" />
            <div className="flex flex-col gap-0.5">
              <span className="text-xs font-medium">Modele actif</span>
              <span className="text-xs text-background/60">
                Random Forest - 96.2%
              </span>
            </div>
          </div>
        </div>

        <div className="rounded-xl bg-background/10 px-3 py-2">
          <span className="text-xs text-background/80">
            Prediction en cours...
          </span>
        </div>
      </div>

      {/* Decorative glow */}
      <div className="absolute -bottom-8 -right-8 h-24 w-24 rounded-full bg-primary/20 blur-2xl" />
    </div>
  );
}
