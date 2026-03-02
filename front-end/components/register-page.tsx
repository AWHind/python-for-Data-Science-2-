"use client";

import React from "react";
import { useState } from "react";
import {
  Heart,
  Loader2,
  Eye,
  EyeOff,
  Activity,
  ArrowLeft,
  User,
  Mail,
  Phone,
  Calendar,
  CheckCircle,
} from "lucide-react";
import { useAuth } from "@/lib/auth-context";

interface RegisterPageProps {
  onGoToLogin: () => void;
}

export function RegisterPage({ onGoToLogin }: RegisterPageProps) {
  const { register } = useAuth();
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
    phone: "",
    age: "",
    sex: "M",
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [success, setSuccess] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");

  const handleChange = (key: string, value: string) => {
    setFormData((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!formData.name || !formData.email || !formData.password || !formData.phone || !formData.age) {
      setError("Veuillez remplir tous les champs obligatoires.");
      return;
    }

    if (formData.password.length < 6) {
      setError("Le mot de passe doit contenir au moins 6 caracteres.");
      return;
    }

    if (formData.password !== formData.confirmPassword) {
      setError("Les mots de passe ne correspondent pas.");
      return;
    }

    setLoading(true);
    const result = await register({
      name: formData.name,
      email: formData.email,
      password: formData.password,
      phone: formData.phone,
      age: formData.age,
      sex: formData.sex,
    });

    if (result.success) {
      setSuccess(true);
      setSuccessMessage(result.message);
    } else {
      setError(result.message);
    }
    setLoading(false);
  };

  if (success) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background p-4">
        <div className="flex w-full max-w-md flex-col items-center rounded-3xl border border-border bg-card p-8 text-center shadow-xl">
          <div className="flex h-20 w-20 items-center justify-center rounded-full bg-emerald-100">
            <CheckCircle className="h-10 w-10 text-emerald-600" />
          </div>
          <h2 className="mt-6 text-2xl font-bold text-foreground">
            Inscription envoyee
          </h2>
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
            {successMessage}
          </p>
          <button
            onClick={onGoToLogin}
            className="mt-8 flex w-full items-center justify-center gap-2 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90"
          >
            Retour a la connexion
          </button>
        </div>
      </div>
    );
  }

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
              Creez votre compte patient
            </h2>
            <p className="mt-4 text-sm leading-relaxed text-sidebar-foreground/60">
              Rejoignez CardioSense pour acceder a l{"'"}outil de prediction des
              arythmies cardiaques. Votre compte sera valide par un
              administrateur avant activation.
            </p>
          </div>

          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-3 rounded-xl bg-sidebar-accent p-4">
              <Activity className="h-5 w-5 text-primary" />
              <div>
                <p className="text-sm font-medium text-sidebar-foreground">
                  Inscription securisee
                </p>
                <p className="text-xs text-sidebar-foreground/50">
                  Validation par un administrateur requise
                </p>
              </div>
            </div>
            <p className="text-xs text-sidebar-foreground/40">
              Vos donnees personnelles sont protegees et traitees conformement
              aux reglementations en vigueur.
            </p>
          </div>
        </div>

        {/* Right Panel - Form */}
        <div className="flex flex-1 flex-col justify-center px-8 py-8 sm:px-12">
          {/* Mobile logo */}
          <div className="mb-6 flex items-center gap-3 lg:hidden">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary">
              <Heart className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="text-lg font-bold text-foreground">
              CardioSense
            </span>
          </div>

          <button
            onClick={onGoToLogin}
            className="mb-4 flex w-fit items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="h-4 w-4" />
            Retour a la connexion
          </button>

          <h1 className="text-2xl font-bold text-foreground">Inscription</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Creez votre compte patient
          </p>

          <form onSubmit={handleSubmit} className="mt-6 flex flex-col gap-4">
            {/* Name */}
            <div className="flex flex-col gap-1.5">
              <label htmlFor="reg-name" className="text-sm font-medium text-foreground">
                Nom complet *
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground/50" />
                <input
                  id="reg-name"
                  type="text"
                  value={formData.name}
                  onChange={(e) => handleChange("name", e.target.value)}
                  placeholder="Votre nom et prenom"
                  required
                  className="w-full rounded-xl border border-border bg-background py-3 pl-10 pr-4 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground/50 focus:border-primary focus:ring-2 focus:ring-primary/20"
                />
              </div>
            </div>

            {/* Email */}
            <div className="flex flex-col gap-1.5">
              <label htmlFor="reg-email" className="text-sm font-medium text-foreground">
                Adresse email *
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground/50" />
                <input
                  id="reg-email"
                  type="email"
                  value={formData.email}
                  onChange={(e) => handleChange("email", e.target.value)}
                  placeholder="exemple@email.com"
                  required
                  className="w-full rounded-xl border border-border bg-background py-3 pl-10 pr-4 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground/50 focus:border-primary focus:ring-2 focus:ring-primary/20"
                />
              </div>
            </div>

            {/* Phone & Age row */}
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1.5">
                <label htmlFor="reg-phone" className="text-sm font-medium text-foreground">
                  Telephone *
                </label>
                <div className="relative">
                  <Phone className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground/50" />
                  <input
                    id="reg-phone"
                    type="tel"
                    value={formData.phone}
                    onChange={(e) => handleChange("phone", e.target.value)}
                    placeholder="+213 555..."
                    required
                    className="w-full rounded-xl border border-border bg-background py-3 pl-10 pr-4 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground/50 focus:border-primary focus:ring-2 focus:ring-primary/20"
                  />
                </div>
              </div>
              <div className="flex flex-col gap-1.5">
                <label htmlFor="reg-age" className="text-sm font-medium text-foreground">
                  Age *
                </label>
                <div className="relative">
                  <Calendar className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground/50" />
                  <input
                    id="reg-age"
                    type="number"
                    min="1"
                    max="120"
                    value={formData.age}
                    onChange={(e) => handleChange("age", e.target.value)}
                    placeholder="Age"
                    required
                    className="w-full rounded-xl border border-border bg-background py-3 pl-10 pr-4 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground/50 focus:border-primary focus:ring-2 focus:ring-primary/20"
                  />
                </div>
              </div>
            </div>

            {/* Sex */}
            <div className="flex flex-col gap-1.5">
              <label className="text-sm font-medium text-foreground">Sexe *</label>
              <div className="flex gap-3">
                {[
                  { value: "M", label: "Masculin" },
                  { value: "F", label: "Feminin" },
                ].map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => handleChange("sex", option.value)}
                    className={`flex-1 rounded-xl border px-4 py-2.5 text-sm font-medium transition-colors ${
                      formData.sex === option.value
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border bg-background text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Password */}
            <div className="flex flex-col gap-1.5">
              <label htmlFor="reg-password" className="text-sm font-medium text-foreground">
                Mot de passe *
              </label>
              <div className="relative">
                <input
                  id="reg-password"
                  type={showPassword ? "text" : "password"}
                  value={formData.password}
                  onChange={(e) => handleChange("password", e.target.value)}
                  placeholder="Minimum 6 caracteres"
                  required
                  className="w-full rounded-xl border border-border bg-background px-4 py-3 pr-11 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground/50 focus:border-primary focus:ring-2 focus:ring-primary/20"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground transition-colors hover:text-foreground"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {/* Confirm Password */}
            <div className="flex flex-col gap-1.5">
              <label htmlFor="reg-confirm" className="text-sm font-medium text-foreground">
                Confirmer le mot de passe *
              </label>
              <input
                id="reg-confirm"
                type="password"
                value={formData.confirmPassword}
                onChange={(e) => handleChange("confirmPassword", e.target.value)}
                placeholder="Repetez le mot de passe"
                required
                className="w-full rounded-xl border border-border bg-background px-4 py-3 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground/50 focus:border-primary focus:ring-2 focus:ring-primary/20"
              />
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
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              {loading ? "Inscription en cours..." : "Creer mon compte"}
            </button>
          </form>

          <p className="mt-4 text-center text-xs text-muted-foreground">
            Vous avez deja un compte ?{" "}
            <button
              onClick={onGoToLogin}
              className="font-medium text-primary transition-colors hover:text-primary/80"
            >
              Se connecter
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
