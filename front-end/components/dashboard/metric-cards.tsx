"use client";

import { Heart, Timer, Waves, MoreHorizontal, ArrowRight } from "lucide-react";
import type { ReactNode } from "react";

interface MetricCardProps {
  icon: ReactNode;
  iconColor: string;
  title: string;
  value: string;
  unit: string;
  status: string;
  statusColor: string;
}

function MetricCard({
  icon,
  iconColor,
  title,
  value,
  unit,
  status,
  statusColor,
}: MetricCardProps) {
  return (
    <div className="flex flex-col justify-between rounded-2xl border border-border bg-card p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={iconColor}>{icon}</span>
          <span className="text-sm font-medium text-foreground">{title}</span>
        </div>
        <button className="text-muted-foreground transition-colors hover:text-foreground">
          <MoreHorizontal className="h-4 w-4" />
        </button>
      </div>

      <div className="mt-4">
        <div className="flex items-baseline gap-1.5">
          <span className="text-2xl font-bold text-foreground">{value}</span>
          <span className="text-sm text-muted-foreground">{unit}</span>
          <span className={`ml-2 text-xs font-medium ${statusColor}`}>
            {status}
          </span>
        </div>
      </div>

      <button className="mt-4 flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground">
        Voir details
        <ArrowRight className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

export function MetricCards() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      <MetricCard
        icon={<Heart className="h-4 w-4" />}
        iconColor="text-primary"
        title="Fr\u00e9quence Cardiaque"
        value="72,56"
        unit="Bpm"
        status="Normal"
        statusColor="text-emerald-600"
      />
      <MetricCard
        icon={<Timer className="h-4 w-4" />}
        iconColor="text-amber-500"
        title="Intervalle QRS"
        value="0.08"
        unit="sec"
        status="Normal"
        statusColor="text-emerald-600"
      />
      <MetricCard
        icon={<Waves className="h-4 w-4" />}
        iconColor="text-sky-500"
        title="Intervalle PR"
        value="0.16"
        unit="sec"
        status="Normal"
        statusColor="text-emerald-600"
      />
    </div>
  );
}
