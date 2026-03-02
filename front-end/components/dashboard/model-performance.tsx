"use client";

import { useState } from "react";
import {
  Bar,
  BarChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  RadialBarChart,
  RadialBar,
  PieChart,
  Pie,
} from "recharts";
import { Brain, Target } from "lucide-react";
import { cn } from "@/lib/utils";

const modelComparison = [
  { name: "Random Forest", accuracy: 96.2, color: "hsl(350, 70%, 55%)" },
  { name: "SVM", accuracy: 93.8, color: "hsl(220, 15%, 30%)" },
  { name: "KNN", accuracy: 91.5, color: "hsl(30, 60%, 65%)" },
  { name: "Decision Tree", accuracy: 89.1, color: "hsl(170, 40%, 50%)" },
  { name: "Naive Bayes", accuracy: 84.3, color: "hsl(350, 50%, 70%)" },
];

const confusionData = [
  { name: "VP", value: 89, label: "Vrais Positifs" },
  { name: "VN", value: 94, label: "Vrais Negatifs" },
  { name: "FP", value: 6, label: "Faux Positifs" },
  { name: "FN", value: 11, label: "Faux Negatifs" },
];

const metricsRadial = [
  {
    name: "Precision",
    value: 96.2,
    fill: "hsl(350, 70%, 55%)",
  },
  {
    name: "Rappel",
    value: 93.7,
    fill: "hsl(170, 40%, 50%)",
  },
  {
    name: "F1-Score",
    value: 94.9,
    fill: "hsl(30, 60%, 65%)",
  },
];

type TabType = "comparaison" | "metriques" | "confusion";

function ModelBarTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: { name: string; accuracy: number } }>;
}) {
  if (active && payload && payload.length) {
    return (
      <div className="rounded-xl border border-border bg-card px-4 py-3 shadow-lg">
        <p className="text-sm font-semibold text-foreground">
          {payload[0].payload.name}
        </p>
        <p className="text-xs text-muted-foreground">
          Precision: {payload[0].payload.accuracy}%
        </p>
      </div>
    );
  }
  return null;
}

export function ModelPerformance() {
  const [activeTab, setActiveTab] = useState<TabType>("comparaison");

  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-foreground" />
          <span className="text-base font-semibold text-foreground">
            Performance des Modeles
          </span>
        </div>
        <div className="flex gap-1 rounded-lg bg-muted p-0.5">
          {(
            [
              { key: "comparaison", label: "Comparaison" },
              { key: "metriques", label: "Metriques" },
              { key: "confusion", label: "Matrice" },
            ] as const
          ).map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={cn(
                "rounded-md px-3 py-1.5 text-xs font-medium transition-all",
                activeTab === tab.key
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Comparison Tab */}
      {activeTab === "comparaison" && (
        <div className="mt-4 h-[260px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={modelComparison}
              layout="vertical"
              margin={{ left: 10, right: 20 }}
            >
              <XAxis
                type="number"
                domain={[80, 100]}
                axisLine={false}
                tickLine={false}
                tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 11 }}
              />
              <YAxis
                type="category"
                dataKey="name"
                axisLine={false}
                tickLine={false}
                tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 12 }}
                width={100}
              />
              <Tooltip content={<ModelBarTooltip />} />
              <Bar dataKey="accuracy" radius={[0, 6, 6, 0]} barSize={24}>
                {modelComparison.map((entry) => (
                  <Cell key={entry.name} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Metrics Tab */}
      {activeTab === "metriques" && (
        <div className="mt-4 flex flex-col items-center gap-4 sm:flex-row sm:items-start sm:justify-around">
          <div className="h-[200px] w-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <RadialBarChart
                cx="50%"
                cy="50%"
                innerRadius="30%"
                outerRadius="100%"
                data={metricsRadial}
                startAngle={180}
                endAngle={0}
              >
                <RadialBar
                  background
                  dataKey="value"
                  cornerRadius={6}
                />
              </RadialBarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-col gap-3">
            {metricsRadial.map((m) => (
              <div key={m.name} className="flex items-center gap-3">
                <div
                  className="h-3 w-3 rounded-full"
                  style={{ backgroundColor: m.fill }}
                />
                <div>
                  <p className="text-sm font-medium text-foreground">
                    {m.name}
                  </p>
                  <p className="text-lg font-bold text-foreground">
                    {m.value}%
                  </p>
                </div>
              </div>
            ))}
            <div className="mt-2 flex items-center gap-2 rounded-lg bg-muted/50 px-3 py-2">
              <Target className="h-4 w-4 text-primary" />
              <span className="text-xs text-muted-foreground">
                Modele: Random Forest (meilleur)
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Confusion Matrix Tab */}
      {activeTab === "confusion" && (
        <div className="mt-4 flex flex-col items-center gap-4 sm:flex-row sm:justify-around">
          <div className="h-[220px] w-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={confusionData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={90}
                  paddingAngle={3}
                  dataKey="value"
                >
                  <Cell fill="hsl(160, 60%, 45%)" />
                  <Cell fill="hsl(210, 60%, 50%)" />
                  <Cell fill="hsl(30, 80%, 55%)" />
                  <Cell fill="hsl(350, 70%, 55%)" />
                </Pie>
                <Tooltip
                  formatter={(value: number, name: string) => [
                    `${value}%`,
                    name,
                  ]}
                  contentStyle={{
                    borderRadius: "12px",
                    border: "1px solid hsl(30, 10%, 88%)",
                    fontSize: "12px",
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-col gap-2">
            {confusionData.map((item, i) => {
              const colors = [
                "bg-emerald-500",
                "bg-sky-500",
                "bg-amber-500",
                "bg-primary",
              ];
              return (
                <div
                  key={item.name}
                  className="flex items-center gap-3 rounded-lg bg-muted/50 px-3 py-2"
                >
                  <div
                    className={cn("h-2.5 w-2.5 rounded-full", colors[i])}
                  />
                  <div className="flex flex-col">
                    <span className="text-xs font-medium text-foreground">
                      {item.label}
                    </span>
                    <span className="text-lg font-bold text-foreground">
                      {item.value}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
