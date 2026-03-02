"use client";

import React from "react"

import { useState } from "react";
import { Heart, Loader2, Eye, EyeOff, Activity, Shield, User } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import { cn } from "@/lib/utils";

interface LoginPageProps {
  onGoToRegister?: () => void;
  onGoToHome?: () => void;
}

export function LoginPage({ onGoToRegister, onGoToHome }: LoginPageProps) {
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    const result = await login(email, password);
    if (!result.success) {
      if (result.pending) {
        setError("Votre compte est en attente de validation par un administrateur.");
      } else {
        setError("Identifiants incorrects. Veuillez reessayer.");
      }
    }
    setLoading(false);
  };

  const handleQuickLogin = async (role: "patient" | "admin") => {
    setError("");
    setLoading(true);
    const creds =
      role === "patient"
        ? { email: "patient@cardiosense.com", password: "patient123" }
        : { email: "admin@cardiosense.com", password: "admin123" };
    setEmail(creds.email);
    setPassword(creds.password);
    await login(creds.email, creds.password);
    setLoading(false);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <div className="flex w-full max-w-[960px] overflow-hidden rounded-3xl border border-border bg-card shadow-xl">
        {/* Left Panel - Branding */}
        <div className="hidden w-[440px] shrink-0 flex-col justify-between bg-sidebar p-10 lg:flex">
          <div>
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary">
                <Heart className="h-6 w-6 text-primary-foreground" />
              </div>
              <span className="text-xl font-bold tracking-wide text-sidebar-foreground">
                CardioSense
              </span>
            </div>
            <h2 className="mt-10 font-serif text-3xl font-semibold leading-tight text-sidebar-foreground">
              Analyse et classification des arythmies cardiaques
            </h2>
            <p className="mt-4 text-sm leading-relaxed text-sidebar-foreground/60">
              Interface intelligente basee sur le dataset UCI Arrhythmia.
              Exploitez la puissance du Machine Learning pour un diagnostic
              assiste rapide et precis.
            </p>
          </div>

          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-3 rounded-xl bg-sidebar-accent p-4">
              <Activity className="h-5 w-5 text-primary" />
              <div>
                <p className="text-sm font-medium text-sidebar-foreground">
                  452 patients analyses
                </p>
                <p className="text-xs text-sidebar-foreground/50">
                  279 attributs - 16 classes
                </p>
              </div>
            </div>
            <p className="text-xs text-sidebar-foreground/40">
              Cet outil constitue une aide a la decision medicale et ne
              remplace en aucun cas un diagnostic clinique.
            </p>
          </div>
        </div>

        {/* Right Panel - Form */}
        <div className="flex flex-1 flex-col justify-center px-8 py-10 sm:px-12">
          {/* Mobile logo */}
          <div className="mb-8 flex items-center gap-3 lg:hidden">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary">
              <Heart className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="text-lg font-bold text-foreground">
              CardioSense
            </span>
          </div>

          <h1 className="text-2xl font-bold text-foreground">Connexion</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Accedez a votre espace personnel
          </p>

          <form onSubmit={handleSubmit} className="mt-8 flex flex-col gap-5">
            <div className="flex flex-col gap-1.5">
              <label
                htmlFor="email"
                className="text-sm font-medium text-foreground"
              >
                Adresse email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="exemple@cardiosense.com"
                required
                className="rounded-xl border border-border bg-background px-4 py-3 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground/50 focus:border-primary focus:ring-2 focus:ring-primary/20"
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label
                htmlFor="password"
                className="text-sm font-medium text-foreground"
              >
                Mot de passe
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Votre mot de passe"
                  required
                  className="w-full rounded-xl border border-border bg-background px-4 py-3 pr-11 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground/50 focus:border-primary focus:ring-2 focus:ring-primary/20"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground transition-colors hover:text-foreground"
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>

            {error && (
              <p className="rounded-lg bg-destructive/10 px-4 py-2.5 text-sm text-destructive">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="flex items-center justify-center gap-2 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-60"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : null}
              {loading ? "Connexion en cours..." : "Se connecter"}
            </button>
          </form>

          {/* Quick access */}
          <div className="mt-8">
            <div className="flex items-center gap-3">
              <div className="h-px flex-1 bg-border" />
              <span className="text-xs text-muted-foreground">
                Acces rapide (demo)
              </span>
              <div className="h-px flex-1 bg-border" />
            </div>
            <div className="mt-4 flex gap-3">


            </div>
          </div>

          {/* Registration link */}
          <p className="mt-6 text-center text-xs text-muted-foreground">
            Pas encore de compte ?{" "}
            <button
              onClick={onGoToRegister}
              className="font-medium text-primary transition-colors hover:text-primary/80"
            >
              Creer un compte patient
            </button>
          </p>

          {onGoToHome && (
            <button
              onClick={onGoToHome}
              className="mt-3 w-full text-center text-xs text-muted-foreground transition-colors hover:text-foreground"
            >
              Retour a l{"'"}accueil
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
