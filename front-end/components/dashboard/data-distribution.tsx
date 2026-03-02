"use client";

import {
  Bar,
  BarChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from "recharts";
import { FileBarChart } from "lucide-react";

const ageDistribution = [
  { range: "20-30", count: 32, color: "hsl(350, 50%, 70%)" },
  { range: "30-40", count: 58, color: "hsl(350, 60%, 63%)" },
  { range: "40-50", count: 95, color: "hsl(350, 65%, 57%)" },
  { range: "50-60", count: 128, color: "hsl(350, 70%, 55%)" },
  { range: "60-70", count: 89, color: "hsl(350, 65%, 57%)" },
  { range: "70-80", count: 38, color: "hsl(350, 60%, 63%)" },
  { range: "80+", count: 12, color: "hsl(350, 50%, 70%)" },
];

const classDistribution = [
  { name: "Normal", count: 245, pct: "54.2%", color: "bg-emerald-500" },
  { name: "Arythmie ischemique", count: 44, pct: "9.7%", color: "bg-primary" },
  { name: "PVC", count: 25, pct: "5.5%", color: "bg-amber-500" },
  { name: "Bloc AV", count: 19, pct: "4.2%", color: "bg-sky-500" },
  { name: "Bloc branche gauche", count: 22, pct: "4.9%", color: "bg-violet-500" },
  { name: "Bloc branche droit", count: 18, pct: "4.0%", color: "bg-orange-500" },
  { name: "Autres classes", count: 79, pct: "17.5%", color: "bg-muted-foreground/50" },
];

function AgeTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: { range: string; count: number } }>;
}) {
  if (active && payload && payload.length) {
    return (
      <div className="rounded-xl border border-border bg-card px-4 py-3 shadow-lg">
        <p className="text-sm font-semibold text-foreground">
          {payload[0].payload.range} ans
        </p>
        <p className="text-xs text-muted-foreground">
          {payload[0].payload.count} patients
        </p>
      </div>
    );
  }
  return null;
}

export function DataDistribution() {
  return (
    <div className="flex flex-col gap-4 lg:flex-row">
      {/* Age Distribution */}
      <div className="flex-1 rounded-2xl border border-border bg-card p-5">
        <div className="flex items-center gap-2">
          <FileBarChart className="h-5 w-5 text-foreground" />
          <span className="text-base font-semibold text-foreground">
            Distribution par Age
          </span>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">
          452 patients - UCI Arrhythmia Dataset
        </p>

        <div className="mt-4 h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={ageDistribution}>
              <XAxis
                dataKey="range"
                axisLine={false}
                tickLine={false}
                tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 11 }}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 11 }}
              />
              <Tooltip content={<AgeTooltip />} />
              <Bar dataKey="count" radius={[6, 6, 0, 0]} barSize={32}>
                {ageDistribution.map((entry) => (
                  <Cell key={entry.range} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Class Distribution */}
      <div className="w-full rounded-2xl border border-border bg-card p-5 lg:w-[340px]">
        <h3 className="text-base font-semibold text-foreground">
          Distribution des Classes
        </h3>
        <p className="mt-1 text-xs text-muted-foreground">
          16 classes, 7 principales affichees
        </p>

        {/* Stacked bar */}
        <div className="mt-4 flex h-4 w-full overflow-hidden rounded-full">
          <div className="w-[54.2%] bg-emerald-500" />
          <div className="w-[9.7%] bg-primary" />
          <div className="w-[5.5%] bg-amber-500" />
          <div className="w-[4.2%] bg-sky-500" />
          <div className="w-[4.9%] bg-violet-500" />
          <div className="w-[4.0%] bg-orange-500" />
          <div className="w-[17.5%] bg-muted-foreground/30" />
        </div>

        <div className="mt-4 flex flex-col gap-2">
          {classDistribution.map((cls) => (
            <div
              key={cls.name}
              className="flex items-center gap-2.5 rounded-lg px-2 py-1.5 transition-colors hover:bg-muted/50"
            >
              <div className={`h-2.5 w-2.5 shrink-0 rounded-full ${cls.color}`} />
              <span className="flex-1 text-xs text-foreground">{cls.name}</span>
              <span className="text-xs font-medium text-foreground">
                {cls.count}
              </span>
              <span className="w-10 text-right text-xs text-muted-foreground">
                {cls.pct}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
