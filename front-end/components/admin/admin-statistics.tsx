"use client";

import {
  Bar,
  BarChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  PieChart,
  Pie,
  Scatter,
  ScatterChart,
  ZAxis,
} from "recharts";
import { FileBarChart, AlertCircle, Database, CheckCircle } from "lucide-react";

const ageDistribution = [
  { range: "20-30", count: 32, color: "hsl(350, 50%, 70%)" },
  { range: "30-40", count: 58, color: "hsl(350, 60%, 63%)" },
  { range: "40-50", count: 95, color: "hsl(350, 65%, 57%)" },
  { range: "50-60", count: 128, color: "hsl(350, 70%, 55%)" },
  { range: "60-70", count: 89, color: "hsl(350, 65%, 57%)" },
  { range: "70-80", count: 38, color: "hsl(350, 60%, 63%)" },
  { range: "80+", count: 12, color: "hsl(350, 50%, 70%)" },
];

const sexDistribution = [
  { name: "Hommes", value: 305, fill: "hsl(220, 15%, 30%)" },
  { name: "Femmes", value: 147, fill: "hsl(350, 70%, 55%)" },
];

const classDistribution = [
  { name: "Normal", count: 245, pct: 54.2, color: "bg-emerald-500" },
  { name: "Arythmie ischemique", count: 44, pct: 9.7, color: "bg-primary" },
  { name: "PVC", count: 25, pct: 5.5, color: "bg-amber-500" },
  { name: "Bloc AV", count: 19, pct: 4.2, color: "bg-sky-500" },
  { name: "BBG", count: 22, pct: 4.9, color: "bg-violet-500" },
  { name: "BBD", count: 18, pct: 4.0, color: "bg-orange-500" },
  { name: "Autres (10 classes)", count: 79, pct: 17.5, color: "bg-muted-foreground/40" },
];

const missingValues = [
  { attr: "T wave", missing: 5, total: 452 },
  { attr: "P wave", missing: 12, total: 452 },
  { attr: "QRST angle", missing: 8, total: 452 },
  { attr: "Heart rate", missing: 0, total: 452 },
  { attr: "QRS duration", missing: 2, total: 452 },
  { attr: "PR interval", missing: 3, total: 452 },
];

const correlationData = [
  { x: 72, y: 80, z: 20 },
  { x: 88, y: 95, z: 25 },
  { x: 65, y: 78, z: 18 },
  { x: 102, y: 130, z: 30 },
  { x: 76, y: 82, z: 20 },
  { x: 91, y: 110, z: 28 },
  { x: 68, y: 75, z: 19 },
  { x: 55, y: 145, z: 35 },
  { x: 84, y: 88, z: 22 },
  { x: 70, y: 79, z: 18 },
];

const datasetStats = [
  { label: "Instances totales", value: "452" },
  { label: "Attributs", value: "279" },
  { label: "Classes de sortie", value: "16" },
  { label: "Attributs numeriques", value: "206" },
  { label: "Attributs lineaires", value: "73" },
  { label: "Valeurs manquantes", value: "~1.3%" },
];

function AgeTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { range: string; count: number } }> }) {
  if (active && payload && payload.length) {
    return (
      <div className="rounded-xl border border-border bg-card px-4 py-3 shadow-lg">
        <p className="text-sm font-semibold text-foreground">{payload[0].payload.range} ans</p>
        <p className="text-xs text-muted-foreground">{payload[0].payload.count} patients</p>
      </div>
    );
  }
  return null;
}

export function AdminStatistics() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-2xl font-bold text-foreground">Analyse Statistique</h2>
        <p className="mt-1 text-sm text-muted-foreground">Exploration des donnees du dataset UCI Arrhythmia</p>
      </div>

      {/* Dataset overview */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {datasetStats.map((s) => (
          <div key={s.label} className="flex flex-col items-center gap-1 rounded-2xl border border-border bg-card p-4">
            <p className="text-xl font-bold text-foreground">{s.value}</p>
            <p className="text-center text-xs text-muted-foreground">{s.label}</p>
          </div>
        ))}
      </div>

      <div className="flex flex-col gap-4 lg:flex-row">
        {/* Age Distribution */}
        <div className="flex-1 rounded-2xl border border-border bg-card p-5">
          <div className="flex items-center gap-2">
            <FileBarChart className="h-5 w-5 text-foreground" />
            <span className="text-base font-semibold text-foreground">Distribution par Age</span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">452 patients</p>
          <div className="mt-4 h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ageDistribution}>
                <XAxis dataKey="range" axisLine={false} tickLine={false} tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 11 }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 11 }} />
                <Tooltip content={<AgeTooltip />} />
                <Bar dataKey="count" radius={[6, 6, 0, 0]} barSize={36}>
                  {ageDistribution.map((e) => (<Cell key={e.range} fill={e.color} />))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Sex Distribution */}
        <div className="w-full rounded-2xl border border-border bg-card p-5 lg:w-[300px]">
          <h3 className="text-base font-semibold text-foreground">Distribution par Sexe</h3>
          <div className="mt-2 h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={sexDistribution} cx="50%" cy="50%" innerRadius={50} outerRadius={75} paddingAngle={4} dataKey="value">
                  {sexDistribution.map((e) => (<Cell key={e.name} fill={e.fill} />))}
                </Pie>
                <Tooltip formatter={(val: number) => [`${val} patients`]} contentStyle={{ borderRadius: 12, fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-6">
            {sexDistribution.map((s) => (
              <div key={s.name} className="flex items-center gap-2">
                <div className="h-3 w-3 rounded-full" style={{ backgroundColor: s.fill }} />
                <span className="text-xs text-foreground">{s.name} ({s.value})</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-4 lg:flex-row">
        {/* Class Distribution */}
        <div className="flex-1 rounded-2xl border border-border bg-card p-5">
          <h3 className="text-base font-semibold text-foreground">Distribution des Classes d{"'"}Arythmie</h3>
          <p className="mt-1 text-xs text-muted-foreground">16 classes, 7 principales affichees</p>
          <div className="mt-4 flex h-4 w-full overflow-hidden rounded-full">
            {classDistribution.map((cls) => (
              <div key={cls.name} className={cls.color} style={{ width: `${cls.pct}%` }} />
            ))}
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2">
            {classDistribution.map((cls) => (
              <div key={cls.name} className="flex items-center gap-2.5 rounded-lg px-2 py-1.5 transition-colors hover:bg-muted/50">
                <div className={`h-2.5 w-2.5 shrink-0 rounded-full ${cls.color}`} />
                <span className="flex-1 text-xs text-foreground">{cls.name}</span>
                <span className="text-xs font-medium text-foreground">{cls.count}</span>
                <span className="w-10 text-right text-xs text-muted-foreground">{cls.pct}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Correlation */}
        <div className="w-full rounded-2xl border border-border bg-card p-5 lg:w-[340px]">
          <h3 className="text-base font-semibold text-foreground">Correlation FC / QRS</h3>
          <p className="mt-1 text-xs text-muted-foreground">Frequence cardiaque vs duree QRS</p>
          <div className="mt-3 h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ bottom: 5 }}>
                <XAxis type="number" dataKey="x" name="FC" unit=" bpm" axisLine={false} tickLine={false} tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 10 }} />
                <YAxis type="number" dataKey="y" name="QRS" unit=" ms" axisLine={false} tickLine={false} tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 10 }} />
                <ZAxis type="number" dataKey="z" range={[40, 200]} />
                <Tooltip contentStyle={{ borderRadius: 12, fontSize: 12 }} />
                <Scatter data={correlationData} fill="hsl(350, 70%, 55%)" fillOpacity={0.6} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Missing Values */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <div className="flex items-center gap-2">
          <AlertCircle className="h-5 w-5 text-amber-500" />
          <h3 className="text-base font-semibold text-foreground">Analyse des Valeurs Manquantes</h3>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">Apercu des donnees incompletes dans le dataset</p>
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {missingValues.map((mv) => (
            <div key={mv.attr} className="flex items-center gap-3 rounded-xl border border-border bg-background p-4">
              {mv.missing === 0 ? (
                <CheckCircle className="h-5 w-5 shrink-0 text-emerald-500" />
              ) : (
                <Database className="h-5 w-5 shrink-0 text-amber-500" />
              )}
              <div className="flex-1">
                <p className="text-sm font-medium text-foreground">{mv.attr}</p>
                <p className="text-xs text-muted-foreground">
                  {mv.missing === 0 ? "Aucune valeur manquante" : `${mv.missing}/${mv.total} manquantes (${((mv.missing / mv.total) * 100).toFixed(1)}%)`}
                </p>
              </div>
              <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
                <div
                  className={mv.missing === 0 ? "bg-emerald-500" : "bg-amber-500"}
                  style={{ width: `${((mv.total - mv.missing) / mv.total) * 100}%`, height: "100%" }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
