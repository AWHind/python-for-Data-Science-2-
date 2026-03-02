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
} from "lucide-react";
import type { ReactNode } from "react";

interface StatCardProps {
  icon: ReactNode;
  iconBg: string;
  label: string;
  value: string;
  subValue: string;
  trend: number;
  trendLabel: string;
}

function StatCard({
  icon,
  iconBg,
  label,
  value,
  subValue,
  trend,
  trendLabel,
}: StatCardProps) {
  const isPositive = trend >= 0;
  return (
    <div className="flex flex-col justify-between rounded-2xl border border-border bg-card p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className={`flex h-9 w-9 items-center justify-center rounded-xl ${iconBg}`}
          >
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
          <span
            className={`text-xs font-medium ${isPositive ? "text-emerald-600" : "text-red-500"}`}
          >
            {isPositive ? "+" : ""}
            {trend}%
          </span>
          <span className="text-xs text-muted-foreground">{trendLabel}</span>
        </div>
      </div>

      <button className="mt-3 flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground">
        Voir details
        <ArrowRight className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

export function AdminStats() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <StatCard
        icon={<Users className="h-4 w-4 text-primary" />}
        iconBg="bg-primary/10"
        label="Total Patients"
        value="452"
        subValue="instances"
        trend={12}
        trendLabel="ce mois"
      />
      <StatCard
        icon={<Database className="h-4 w-4 text-sky-500" />}
        iconBg="bg-sky-500/10"
        label="Attributs"
        value="279"
        subValue="features"
        trend={0}
        trendLabel="UCI Dataset"
      />
      <StatCard
        icon={<Brain className="h-4 w-4 text-amber-500" />}
        iconBg="bg-amber-500/10"
        label="Precision Modele"
        value="96.2"
        subValue="%"
        trend={2.4}
        trendLabel="vs. precedent"
      />
      <StatCard
        icon={<TrendingUp className="h-4 w-4 text-emerald-500" />}
        iconBg="bg-emerald-500/10"
        label="Predictions"
        value="1,284"
        subValue="total"
        trend={18}
        trendLabel="cette semaine"
      />
    </div>
  );
}
