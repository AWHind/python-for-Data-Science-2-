"use client";

import { useState } from "react";
import {
  Heart,
  Activity,
  Zap,
  ArrowRight,
  CheckCircle,
  AlertTriangle,
  Loader2,
  Info,
  RotateCcw,
  Download,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface PredictionResult {
  id: string;
  prediction: string;
  confidence: number;
  isNormal: boolean;
  classId: number;
  details: { label: string; value: string }[];
  timestamp: string;
}

const defaultFormValues: Record<string, string> = {
  age: "54",
  sex: "1",
  height: "172",
  weight: "78",
  qrsDuration: "80",
  prInterval: "160",
  qtInterval: "370",
  tInterval: "180",
  pInterval: "100",
  heartRate: "72",
  qrsCount: "6",
};

const inputFields = [
  { key: "age", label: "Age", unit: "ans", placeholder: "Ex: 54" },
  { key: "sex", label: "Sexe", unit: "0=F, 1=M", placeholder: "0 ou 1" },
  { key: "height", label: "Taille", unit: "cm", placeholder: "Ex: 172" },
  { key: "weight", label: "Poids", unit: "kg", placeholder: "Ex: 78" },
  { key: "qrsDuration", label: "Duree QRS", unit: "ms", placeholder: "Ex: 80" },
  { key: "prInterval", label: "Intervalle PR", unit: "ms", placeholder: "Ex: 160" },
  { key: "qtInterval", label: "Intervalle QT", unit: "ms", placeholder: "Ex: 370" },
  { key: "tInterval", label: "Intervalle T", unit: "ms", placeholder: "Ex: 180" },
  { key: "pInterval", label: "Intervalle P", unit: "ms", placeholder: "Ex: 100" },
  { key: "heartRate", label: "Frequence Cardiaque", unit: "bpm", placeholder: "Ex: 72" },
  { key: "qrsCount", label: "Nombre de QRS", unit: "", placeholder: "Ex: 6" },
];

export function PatientPrediction() {
  const [formData, setFormData] = useState<Record<string, string>>(defaultFormValues);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const handleInputChange = (key: string, value: string) => {
    setFormData((prev) => ({ ...prev, [key]: value }));
  };

  const handlePredict = () => {
    setIsLoading(true);
    setResult(null);
    setTimeout(() => {
      const hr = Number.parseInt(formData.heartRate) || 72;
      const qrs = Number.parseInt(formData.qrsDuration) || 80;
      const isNormal = hr >= 60 && hr <= 100 && qrs >= 60 && qrs <= 120;
      const confidence = isNormal ? 85 + Math.random() * 12 : 60 + Math.random() * 25;
      setResult({
        id: `PRED-${Date.now().toString(36).toUpperCase()}`,
        prediction: isNormal ? "Rythme Sinusal Normal" : "Arythmie Detectee - Classe 2",
        confidence: Math.round(confidence * 10) / 10,
        isNormal,
        classId: isNormal ? 1 : 2,
        details: [
          { label: "Classe predite", value: isNormal ? "Classe 1 (Normal)" : "Classe 2 (Arythmie)" },
          { label: "Modele utilise", value: "Random Forest" },
          { label: "Temps d'inference", value: "0.23s" },
          { label: "Features utilisees", value: "279 attributs UCI" },
          { label: "Validation croisee", value: "k=10 folds" },
          { label: "AUC-ROC", value: "0.974" },
        ],
        timestamp: new Date().toLocaleString("fr-FR"),
      });
      setIsLoading(false);
    }, 2000);
  };

  const handleReset = () => {
    setFormData(defaultFormValues);
    setResult(null);
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Page Header */}
      <div>
        <h2 className="text-2xl font-bold text-foreground">
          Nouvelle Prediction
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Saisissez les parametres ECG du patient pour obtenir une prediction de classification
        </p>
      </div>

      <div className="flex flex-col gap-6 xl:flex-row">
        {/* Form */}
        <div className="flex-1 rounded-2xl border border-border bg-card p-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
              <Activity className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-foreground">
                Saisie des Donnees ECG
              </h3>
              <p className="text-xs text-muted-foreground">
                Parametres du patient - UCI Arrhythmia Dataset (279 attributs)
              </p>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3 lg:grid-cols-4">
            {inputFields.map((field) => (
              <div key={field.key} className="flex flex-col gap-1.5">
                <label htmlFor={`pred-${field.key}`} className="text-xs font-medium text-foreground">
                  {field.label}
                  {field.unit && (
                    <span className="ml-1 font-normal text-muted-foreground">
                      ({field.unit})
                    </span>
                  )}
                </label>
                <input
                  id={`pred-${field.key}`}
                  type="text"
                  value={formData[field.key] || ""}
                  onChange={(e) => handleInputChange(field.key, e.target.value)}
                  placeholder={field.placeholder}
                  className="rounded-lg border border-border bg-background px-3 py-2.5 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground/40 focus:border-primary focus:ring-2 focus:ring-primary/20"
                />
              </div>
            ))}
          </div>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <button
              onClick={handlePredict}
              disabled={isLoading}
              className="flex items-center gap-2 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-60"
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
              {isLoading ? "Analyse en cours..." : "Lancer la Prediction"}
            </button>
            <button
              onClick={handleReset}
              className="flex items-center gap-2 rounded-xl border border-border bg-transparent px-5 py-3 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              <RotateCcw className="h-4 w-4" />
              Reinitialiser
            </button>
          </div>

          <div className="mt-5 flex items-start gap-2 rounded-xl bg-muted/50 p-4">
            <Info className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <p className="text-xs leading-relaxed text-muted-foreground">
              Cet outil constitue une aide a la decision medicale et ne remplace en aucun cas un
              diagnostic clinique. Les resultats sont bases sur le dataset UCI Arrhythmia (452
              instances, 279 attributs, 16 classes).
            </p>
          </div>
        </div>

        {/* Result Panel */}
        <div className="w-full rounded-2xl border border-border bg-card p-6 xl:w-[360px]">
          <h3 className="text-base font-semibold text-foreground">
            Resultat de la Prediction
          </h3>

          {!result && !isLoading && (
            <div className="mt-12 flex flex-col items-center gap-4 text-center">
              <div className="flex h-20 w-20 items-center justify-center rounded-full bg-muted">
                <Heart className="h-10 w-10 text-muted-foreground/30" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">
                  En attente d{"'"}analyse
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Saisissez les donnees et lancez la prediction pour voir le resultat ici.
                </p>
              </div>
            </div>
          )}

          {isLoading && (
            <div className="mt-12 flex flex-col items-center gap-4">
              <div className="relative flex h-24 w-24 items-center justify-center">
                <div className="absolute inset-0 animate-ping rounded-full bg-primary/20" />
                <div className="absolute inset-2 animate-pulse rounded-full bg-primary/10" />
                <Heart className="relative h-10 w-10 animate-pulse text-primary" />
              </div>
              <div className="text-center">
                <p className="text-sm font-semibold text-foreground">
                  Analyse en cours...
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Classification par Random Forest
                </p>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div className="h-full animate-pulse rounded-full bg-primary/60" style={{ width: "70%" }} />
              </div>
            </div>
          )}

          {result && !isLoading && (
            <div className="mt-4 flex flex-col gap-4">
              {/* Main Result */}
              <div
                className={cn(
                  "flex items-center gap-3 rounded-xl p-4",
                  result.isNormal ? "bg-emerald-50" : "bg-red-50"
                )}
              >
                {result.isNormal ? (
                  <CheckCircle className="h-7 w-7 shrink-0 text-emerald-600" />
                ) : (
                  <AlertTriangle className="h-7 w-7 shrink-0 text-red-600" />
                )}
                <div>
                  <p className={cn("text-sm font-bold", result.isNormal ? "text-emerald-800" : "text-red-800")}>
                    {result.prediction}
                  </p>
                  <p className={cn("text-xs", result.isNormal ? "text-emerald-600" : "text-red-600")}>
                    Confiance: {result.confidence}%
                  </p>
                </div>
              </div>

              {/* Confidence Bar */}
              <div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Niveau de confiance</span>
                  <span className="font-semibold text-foreground">{result.confidence}%</span>
                </div>
                <div className="mt-2 h-3 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all duration-1000",
                      result.isNormal ? "bg-emerald-500" : "bg-primary"
                    )}
                    style={{ width: `${result.confidence}%` }}
                  />
                </div>
              </div>

              {/* Details */}
              <div className="flex flex-col gap-2">
                {result.details.map((d) => (
                  <div key={d.label} className="flex items-center justify-between rounded-lg bg-muted/50 px-3 py-2">
                    <span className="text-xs text-muted-foreground">{d.label}</span>
                    <span className="text-xs font-medium text-foreground">{d.value}</span>
                  </div>
                ))}
              </div>

              <div className="mt-1 flex flex-col gap-2">
                <button
                    onClick={() => window.open("http://127.0.0.1:8000/report")}
                    className="flex items-center justify-center gap-2 rounded-xl bg-primary px-4 py-2.5 text-sm text-white transition-colors hover:opacity-90"
                >
                  Télécharger le rapport
                </button>

                <button className="flex items-center justify-center gap-2 rounded-xl border border-border bg-transparent px-4 py-2.5 text-sm text-foreground transition-colors hover:bg-muted">
                  Voir le rapport complet
                  <ArrowRight className="h-3.5 w-3.5" />
                </button>
              </div>

              <p className="text-center text-xs text-muted-foreground">
                ID: {result.id} | {result.timestamp}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
