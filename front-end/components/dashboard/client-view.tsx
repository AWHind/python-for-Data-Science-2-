"use client";

import { ClientHeader } from "./client-header";
import { MetricCards } from "./metric-cards";
import { HeartRateChart } from "./heart-rate-chart";
import { PredictionForm } from "./prediction-form";
import { RecentActivity } from "./recent-activity";
import { AIPoweredPanel } from "./bottom-panels";
import { Stethoscope, Calendar, Share2 } from "lucide-react";

export function ClientView() {
  return (
    <div className="flex flex-1 flex-col gap-4 overflow-y-auto">
      <ClientHeader />

      {/* Action Bar */}
      <div className="flex flex-wrap items-center gap-3">
        <button className="flex items-center gap-2 rounded-full bg-foreground px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-90">
          <Stethoscope className="h-4 w-4" />
          Nouvelle Analyse
        </button>
        <button className="flex items-center gap-2 rounded-full border border-border bg-card px-4 py-2 text-sm text-foreground">
          <Calendar className="h-4 w-4" />
          10 Fevrier 2026
        </button>
        <button className="flex items-center gap-2 rounded-full border border-border bg-card px-4 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground">
          <Share2 className="h-4 w-4" />
          Exporter PDF
        </button>
      </div>

      <div className="flex flex-1 gap-4">
        {/* Center Content */}
        <div className="flex flex-1 flex-col gap-4">
          <MetricCards />
          <PredictionForm />
          <HeartRateChart />
        </div>

        {/* Right Panel */}
        <div className="hidden w-[280px] shrink-0 flex-col gap-4 xl:flex">
          <RecentActivity />
          <AIPoweredPanel />
        </div>
      </div>
    </div>
  );
}
