"use client";

import {
  Bar,
  BarChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";
import {
  Activity,
  Target,
  CheckCircle,
  Zap,
  TrendingUp,
  Brain,
  Award,
} from "lucide-react";
import { cn } from "@/lib/utils";

const performanceMetrics = [
  { label: "Accuracy", value: 96.2, icon: Target, description: "Proportion de predictions correctes sur l'ensemble des donnees" },
  { label: "Precision", value: 95.8, icon: CheckCircle, description: "Proportion de vrais positifs parmi les predictions positives" },
  { label: "Rappel (Recall)", value: 93.7, icon: Zap, description: "Proportion de vrais positifs detectes parmi tous les positifs reels" },
  { label: "F1-Score", value: 94.7, icon: Brain, description: "Moyenne harmonique de la precision et du rappel" },
];

const confusionMatrix = {
  tp: 89,
  tn: 94,
  fp: 6,
  fn: 11,
};

const classPerformance = [
  { name: "Classe 1 (Normal)", precision: 98.1, recall: 97.5, f1: 97.8, support: 245, color: "hsl(160, 60%, 45%)" },
  { name: "Classe 2 (Ischemie)", precision: 91.2, recall: 88.6, f1: 89.9, support: 44, color: "hsl(350, 70%, 55%)" },
  { name: "Classe 3 (PVC)", precision: 89.5, recall: 85.2, f1: 87.3, support: 25, color: "hsl(38, 80%, 55%)" },
  { name: "Classe 4 (BAV)", precision: 87.3, recall: 84.1, f1: 85.7, support: 19, color: "hsl(210, 60%, 50%)" },
  { name: "Classe 5 (BBG)", precision: 90.1, recall: 86.8, f1: 88.4, support: 22, color: "hsl(270, 50%, 55%)" },
  { name: "Classe 6 (BBD)", precision: 88.7, recall: 83.3, f1: 85.9, support: 18, color: "hsl(25, 70%, 55%)" },
];

const radarData = [
  { metric: "Accuracy", value: 96.2 },
  { metric: "Precision", value: 95.8 },
  { metric: "Rappel", value: 93.7 },
  { metric: "F1-Score", value: 94.7 },
  { metric: "Specificite", value: 95.1 },
  { metric: "AUC-ROC", value: 97.4 },
];

const crossValidation = [
  { fold: "Fold 1", accuracy: 95.8 },
  { fold: "Fold 2", accuracy: 96.5 },
  { fold: "Fold 3", accuracy: 95.2 },
  { fold: "Fold 4", accuracy: 97.1 },
  { fold: "Fold 5", accuracy: 96.0 },
  { fold: "Fold 6", accuracy: 95.9 },
  { fold: "Fold 7", accuracy: 96.8 },
  { fold: "Fold 8", accuracy: 96.3 },
  { fold: "Fold 9", accuracy: 95.5 },
  { fold: "Fold 10", accuracy: 97.0 },
];

function CVTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { fold: string; accuracy: number } }> }) {
  if (active && payload && payload.length) {
    return (
      <div className="rounded-xl border border-border bg-card px-4 py-3 shadow-lg">
        <p className="text-sm font-semibold text-foreground">{payload[0].payload.fold}</p>
        <p className="text-xs text-muted-foreground">Accuracy: {payload[0].payload.accuracy}%</p>
      </div>
    );
  }
  return null;
}

export function AdminPerformance() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-2xl font-bold text-foreground">Performances du Modele</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Evaluation detaillee du modele de classification Random Forest
        </p>
      </div>

      {/* Best Model Banner */}
      <div className="flex items-center gap-4 rounded-2xl border border-border bg-card p-5">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
          <Award className="h-7 w-7 text-primary" />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-bold text-foreground">Random Forest - Modele Principal</h3>
            <span className="rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-medium text-emerald-700">Actif</span>
          </div>
          <p className="mt-0.5 text-sm text-muted-foreground">
            Validation croisee k=10 | 279 features | 16 classes | Dataset UCI Arrhythmia
          </p>
        </div>
      </div>

      {/* Main Metrics */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {performanceMetrics.map((metric) => (
          <div key={metric.label} className="flex flex-col gap-3 rounded-2xl border border-border bg-card p-5">
            <div className="flex items-center gap-2">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10">
                <metric.icon className="h-4 w-4 text-primary" />
              </div>
              <span className="text-sm font-medium text-foreground">{metric.label}</span>
            </div>
            <div>
              <p className="text-3xl font-bold text-foreground">{metric.value}%</p>
              <p className="mt-1 text-xs text-muted-foreground">{metric.description}</p>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-all duration-1000"
                style={{ width: `${metric.value}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="flex flex-col gap-4 xl:flex-row">
        {/* Confusion Matrix */}
        <div className="flex-1 rounded-2xl border border-border bg-card p-5">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-foreground" />
            <h3 className="text-base font-semibold text-foreground">Matrice de Confusion</h3>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">Distribution des predictions vs realite</p>

          <div className="mx-auto mt-6 max-w-xs">
            {/* Matrix Header */}
            <div className="mb-2 ml-[100px] grid grid-cols-2 gap-2 text-center">
              <span className="text-xs font-medium text-muted-foreground">Predit Positif</span>
              <span className="text-xs font-medium text-muted-foreground">Predit Negatif</span>
            </div>
            {/* Matrix Row 1 */}
            <div className="flex items-center gap-2">
              <span className="w-[92px] text-right text-xs font-medium text-muted-foreground">Reel Positif</span>
              <div className="grid flex-1 grid-cols-2 gap-2">
                <div className="flex flex-col items-center gap-1 rounded-xl bg-emerald-50 p-4">
                  <span className="text-2xl font-bold text-emerald-700">{confusionMatrix.tp}</span>
                  <span className="text-xs text-emerald-600">VP</span>
                </div>
                <div className="flex flex-col items-center gap-1 rounded-xl bg-red-50 p-4">
                  <span className="text-2xl font-bold text-red-700">{confusionMatrix.fn}</span>
                  <span className="text-xs text-red-600">FN</span>
                </div>
              </div>
            </div>
            {/* Matrix Row 2 */}
            <div className="mt-2 flex items-center gap-2">
              <span className="w-[92px] text-right text-xs font-medium text-muted-foreground">Reel Negatif</span>
              <div className="grid flex-1 grid-cols-2 gap-2">
                <div className="flex flex-col items-center gap-1 rounded-xl bg-amber-50 p-4">
                  <span className="text-2xl font-bold text-amber-700">{confusionMatrix.fp}</span>
                  <span className="text-xs text-amber-600">FP</span>
                </div>
                <div className="flex flex-col items-center gap-1 rounded-xl bg-sky-50 p-4">
                  <span className="text-2xl font-bold text-sky-700">{confusionMatrix.tn}</span>
                  <span className="text-xs text-sky-600">VN</span>
                </div>
              </div>
            </div>
          </div>

          {/* Matrix Summary */}
          <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              { label: "Vrais Positifs", value: confusionMatrix.tp, color: "text-emerald-600", bg: "bg-emerald-50" },
              { label: "Faux Negatifs", value: confusionMatrix.fn, color: "text-red-600", bg: "bg-red-50" },
              { label: "Faux Positifs", value: confusionMatrix.fp, color: "text-amber-600", bg: "bg-amber-50" },
              { label: "Vrais Negatifs", value: confusionMatrix.tn, color: "text-sky-600", bg: "bg-sky-50" },
            ].map((item) => (
              <div key={item.label} className={cn("flex flex-col items-center gap-1 rounded-xl p-3", item.bg)}>
                <span className={cn("text-lg font-bold", item.color)}>{item.value}%</span>
                <span className="text-xs text-muted-foreground">{item.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Radar Chart */}
        <div className="w-full rounded-2xl border border-border bg-card p-5 xl:w-[380px]">
          <h3 className="text-base font-semibold text-foreground">Vue Radar des Metriques</h3>
          <p className="mt-1 text-xs text-muted-foreground">Profil de performance global du modele</p>
          <div className="mt-4 h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
                <PolarGrid stroke="hsl(30, 10%, 88%)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 11 }} />
                <PolarRadiusAxis angle={30} domain={[80, 100]} tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 10 }} />
                <Radar
                  name="Performance"
                  dataKey="value"
                  stroke="hsl(350, 70%, 55%)"
                  fill="hsl(350, 70%, 55%)"
                  fillOpacity={0.2}
                  strokeWidth={2}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Cross Validation */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-foreground" />
          <h3 className="text-base font-semibold text-foreground">Validation Croisee (k=10)</h3>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">Accuracy par fold - Moyenne: 96.2% | Ecart-type: 0.61%</p>
        <div className="mt-4 h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={crossValidation}>
              <XAxis dataKey="fold" axisLine={false} tickLine={false} tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 11 }} />
              <YAxis domain={[94, 98]} axisLine={false} tickLine={false} tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 11 }} />
              <Tooltip content={<CVTooltip />} />
              <Bar dataKey="accuracy" radius={[6, 6, 0, 0]} barSize={32}>
                {crossValidation.map((entry, i) => (
                  <Cell key={entry.fold} fill={entry.accuracy >= 96.2 ? "hsl(350, 70%, 55%)" : "hsl(350, 50%, 70%)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Per-Class Performance Table */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <h3 className="text-base font-semibold text-foreground">Performance par Classe</h3>
        <p className="mt-1 text-xs text-muted-foreground">Metriques detaillees pour chaque classe d{"'"}arythmie</p>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-border">
                <th className="pb-3 text-xs font-medium text-muted-foreground">Classe</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Precision</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Rappel</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">F1-Score</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Support</th>
              </tr>
            </thead>
            <tbody>
              {classPerformance.map((cls) => (
                <tr key={cls.name} className="border-b border-border/50 last:border-0">
                  <td className="py-3">
                    <div className="flex items-center gap-2">
                      <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: cls.color }} />
                      <span className="text-sm font-medium text-foreground">{cls.name}</span>
                    </div>
                  </td>
                  <td className="py-3">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
                        <div className="h-full rounded-full bg-primary" style={{ width: `${cls.precision}%` }} />
                      </div>
                      <span className="text-sm text-foreground">{cls.precision}%</span>
                    </div>
                  </td>
                  <td className="py-3">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
                        <div className="h-full rounded-full bg-emerald-500" style={{ width: `${cls.recall}%` }} />
                      </div>
                      <span className="text-sm text-foreground">{cls.recall}%</span>
                    </div>
                  </td>
                  <td className="py-3">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
                        <div className="h-full rounded-full bg-amber-500" style={{ width: `${cls.f1}%` }} />
                      </div>
                      <span className="text-sm text-foreground">{cls.f1}%</span>
                    </div>
                  </td>
                  <td className="py-3 text-sm text-muted-foreground">{cls.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
