"use client";

import {
  Users,
  Database,
  Brain,
  TrendingUp,
  ArrowUpRight,
  ArrowDownRight,
  MoreHorizontal,
  ArrowRight,
  Activity,
  FileHeart,
  Scan,
} from "lucide-react";
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  Bar,
  BarChart,
  Cell,
} from "recharts";
import type { ReactNode } from "react";
import { useAuth } from "@/lib/auth-context";

interface StatCardProps {
  icon: ReactNode;
  iconBg: string;
  label: string;
  value: string;
  subValue: string;
  trend: number;
  trendLabel: string;
}

function StatCard({ icon, iconBg, label, value, subValue, trend, trendLabel }: StatCardProps) {
  const isPositive = trend >= 0;
  return (
    <div className="flex flex-col justify-between rounded-2xl border border-border bg-card p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`flex h-9 w-9 items-center justify-center rounded-xl ${iconBg}`}>
            {icon}
          </div>
          <span className="text-sm font-medium text-foreground">{label}</span>
        </div>
        <button className="text-muted-foreground transition-colors hover:text-foreground">
          <MoreHorizontal className="h-4 w-4" />
        </button>
      </div>
      <div className="mt-4">
        <div className="flex items-baseline gap-1.5">
          <span className="text-2xl font-bold text-foreground">{value}</span>
          <span className="text-sm text-muted-foreground">{subValue}</span>
        </div>
        <div className="mt-1 flex items-center gap-1">
          {isPositive ? (
            <ArrowUpRight className="h-3.5 w-3.5 text-emerald-600" />
          ) : (
            <ArrowDownRight className="h-3.5 w-3.5 text-red-500" />
          )}
          <span className={`text-xs font-medium ${isPositive ? "text-emerald-600" : "text-red-500"}`}>
            {isPositive ? "+" : ""}{trend}%
          </span>
          <span className="text-xs text-muted-foreground">{trendLabel}</span>
        </div>
      </div>
      <button className="mt-3 flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground">
        Voir details <ArrowRight className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

const trendData = [
  { month: "Jan", predictions: 85, accuracy: 94.1 },
  { month: "Fev", predictions: 102, accuracy: 95.3 },
  { month: "Mar", predictions: 98, accuracy: 94.8 },
  { month: "Avr", predictions: 130, accuracy: 95.9 },
  { month: "Mai", predictions: 145, accuracy: 96.0 },
  { month: "Jun", predictions: 128, accuracy: 95.5 },
  { month: "Jul", predictions: 155, accuracy: 96.2 },
  { month: "Aou", predictions: 140, accuracy: 96.1 },
  { month: "Sep", predictions: 162, accuracy: 96.2 },
  { month: "Oct", predictions: 150, accuracy: 96.0 },
  { month: "Nov", predictions: 138, accuracy: 95.8 },
  { month: "Dec", predictions: 131, accuracy: 96.2 },
];

const classBreakdown = [
  { name: "Normal", count: 245, color: "hsl(160, 60%, 45%)" },
  { name: "Isch.", count: 44, color: "hsl(350, 70%, 55%)" },
  { name: "PVC", count: 25, color: "hsl(38, 80%, 55%)" },
  { name: "BAV", count: 19, color: "hsl(210, 60%, 50%)" },
  { name: "BBG", count: 22, color: "hsl(270, 50%, 55%)" },
  { name: "BBD", count: 18, color: "hsl(25, 70%, 55%)" },
  { name: "Autres", count: 79, color: "hsl(220, 10%, 60%)" },
];

const recentActivities = [
  { icon: Activity, iconColor: "text-primary", bgColor: "bg-primary/10", time: "14:20", title: "Prediction patient P-247", desc: "Rythme sinusal normal - 97.2%" },
  { icon: FileHeart, iconColor: "text-amber-500", bgColor: "bg-amber-500/10", time: "11:30", title: "Lot #38 classe", desc: "12 enregistrements analyses" },
  { icon: Brain, iconColor: "text-sky-500", bgColor: "bg-sky-500/10", time: "09:00", title: "Modele re-entraine", desc: "Random Forest - Accuracy 96.2%" },
  { icon: Scan, iconColor: "text-emerald-500", bgColor: "bg-emerald-500/10", time: "08:00", title: "Import donnees", desc: "279 attributs - 452 patients" },
];

function TrendTooltip({ active, payload }: { active?: boolean; payload?: Array<{ value: number; dataKey: string }> }) {
  if (active && payload && payload.length) {
    return (
      <div className="rounded-xl border border-border bg-card px-4 py-3 shadow-lg">
        <p className="text-xs text-muted-foreground">Predictions: <span className="font-semibold text-foreground">{payload[0]?.value}</span></p>
        {payload[1] && <p className="text-xs text-muted-foreground">Accuracy: <span className="font-semibold text-foreground">{payload[1].value}%</span></p>}
      </div>
    );
  }
  return null;
}

export function AdminDashboard() {
  const { user } = useAuth();

  return (
    <div className="flex flex-col gap-6">
      {/* Welcome */}
      <div>
        <h1 className="font-serif text-3xl font-semibold text-foreground">
          Tableau de Bord
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Vue globale du systeme - Bienvenue, {user?.name || "Administrateur"}
        </p>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard icon={<Users className="h-4 w-4 text-primary" />} iconBg="bg-primary/10" label="Total Patients" value="452" subValue="instances" trend={12} trendLabel="ce mois" />
        <StatCard icon={<Database className="h-4 w-4 text-sky-500" />} iconBg="bg-sky-500/10" label="Attributs" value="279" subValue="features" trend={0} trendLabel="UCI Dataset" />
        <StatCard icon={<Brain className="h-4 w-4 text-amber-500" />} iconBg="bg-amber-500/10" label="Precision Modele" value="96.2" subValue="%" trend={2.4} trendLabel="vs. precedent" />
        <StatCard icon={<TrendingUp className="h-4 w-4 text-emerald-500" />} iconBg="bg-emerald-500/10" label="Predictions" value="1,284" subValue="total" trend={18} trendLabel="cette semaine" />
      </div>

      <div className="flex flex-col gap-4 xl:flex-row">
        {/* Predictions Trend */}
        <div className="flex-1 rounded-2xl border border-border bg-card p-5">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-semibold text-foreground">Evolution des Predictions</h3>
              <p className="text-xs text-muted-foreground">Nombre de predictions mensuelles</p>
            </div>
          </div>
          <div className="mt-4 h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="predGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(350, 70%, 55%)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="hsl(350, 70%, 55%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="month" axisLine={false} tickLine={false} tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 11 }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 11 }} />
                <Tooltip content={<TrendTooltip />} />
                <Area type="monotone" dataKey="predictions" stroke="hsl(350, 70%, 55%)" strokeWidth={2.5} fill="url(#predGrad)" dot={false} activeDot={{ r: 4, fill: "hsl(350, 70%, 55%)", stroke: "#fff", strokeWidth: 2 }} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Class Breakdown */}
        <div className="w-full rounded-2xl border border-border bg-card p-5 xl:w-[320px]">
          <h3 className="text-base font-semibold text-foreground">Repartition des Classes</h3>
          <p className="mt-1 text-xs text-muted-foreground">16 classes, 7 principales</p>
          <div className="mt-4 h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classBreakdown}>
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 10 }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fill: "hsl(220, 10%, 50%)", fontSize: 10 }} />
                <Bar dataKey="count" radius={[6, 6, 0, 0]} barSize={28}>
                  {classBreakdown.map((e) => (<Cell key={e.name} fill={e.color} />))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <h3 className="text-base font-semibold text-foreground">Activite Recente</h3>
        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {recentActivities.map((a) => (
            <div key={a.title} className="flex items-start gap-3 rounded-xl border border-border bg-background p-4">
              <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${a.bgColor}`}>
                <a.icon className={`h-5 w-5 ${a.iconColor}`} />
              </div>
              <div className="min-w-0">
                <p className="text-xs text-muted-foreground">{a.time}</p>
                <p className="text-sm font-medium text-foreground">{a.title}</p>
                <p className="truncate text-xs text-muted-foreground">{a.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
