"use client";

import { useState } from "react";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";
import { Heart } from "lucide-react";

const monthlyData = [
  { month: "Jan", bpm: 72, bpmLow: 65 },
  { month: "Fev", bpm: 78, bpmLow: 70 },
  { month: "Mar", bpm: 74, bpmLow: 67 },
  { month: "Avr", bpm: 80, bpmLow: 72 },
  { month: "Mai", bpm: 85, bpmLow: 76 },
  { month: "Jun", bpm: 77, bpmLow: 69 },
  { month: "Jul", bpm: 92, bpmLow: 80 },
  { month: "Aou", bpm: 88, bpmLow: 78 },
  { month: "Sep", bpm: 95, bpmLow: 84 },
  { month: "Oct", bpm: 78, bpmLow: 70 },
  { month: "Nov", bpm: 74, bpmLow: 66 },
  { month: "Dec", bpm: 70, bpmLow: 63 },
];

const weeklyData = [
  { month: "Lun", bpm: 68, bpmLow: 62 },
  { month: "Mar", bpm: 75, bpmLow: 68 },
  { month: "Mer", bpm: 82, bpmLow: 74 },
  { month: "Jeu", bpm: 71, bpmLow: 64 },
  { month: "Ven", bpm: 89, bpmLow: 78 },
  { month: "Sam", bpm: 76, bpmLow: 69 },
  { month: "Dim", bpm: 72, bpmLow: 65 },
];

type Period = "Mensuel" | "Hebdomadaire";

function CustomTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
}) {
  if (active && payload && payload.length) {
    return (
      <div className="rounded-xl border border-border bg-card px-4 py-3 shadow-lg">
        <p className="text-xs font-semibold text-foreground">
          Excellent !
        </p>
        <p className="mt-0.5 text-xs text-muted-foreground">
          {payload[0].value} Bpm - Vous atteignez votre meilleur
        </p>
        <button className="mt-1 text-xs font-medium text-primary hover:underline">
          Voir details
        </button>
      </div>
    );
  }
  return null;
}

export function HeartRateChart() {
  const [period, setPeriod] = useState<Period>("Mensuel");
  const data = period === "Mensuel" ? monthlyData : weeklyData;

  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Heart className="h-5 w-5 text-primary" />
          <span className="text-base font-semibold text-foreground">
            Rythme Cardiaque
          </span>
        </div>
        <select
          value={period}
          onChange={(e) => setPeriod(e.target.value as Period)}
          className="rounded-lg border border-border bg-card px-3 py-1.5 text-xs text-foreground outline-none"
        >
          <option>Mensuel</option>
          <option>Hebdomadaire</option>
        </select>
      </div>

      <div className="mt-1">
        <span className="text-3xl font-bold text-foreground">72,56</span>
        <span className="ml-1.5 text-sm text-muted-foreground">Bpm</span>
      </div>

      <div className="mt-4 h-[200px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="bpmGradient" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor="hsl(350, 70%, 55%)"
                  stopOpacity={0.3}
                />
                <stop
                  offset="95%"
                  stopColor="hsl(350, 70%, 55%)"
                  stopOpacity={0}
                />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="month"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 12 }}
              dy={10}
            />
            <YAxis
              domain={[60, 100]}
              axisLine={false}
              tickLine={false}
              tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 12 }}
              dx={-10}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="bpm"
              stroke="hsl(350, 70%, 55%)"
              strokeWidth={2.5}
              fill="url(#bpmGradient)"
              dot={false}
              activeDot={{
                r: 5,
                fill: "hsl(350, 70%, 55%)",
                stroke: "hsl(0, 0%, 100%)",
                strokeWidth: 2,
              }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
