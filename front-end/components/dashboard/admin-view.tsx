"use client";

import { AdminHeader } from "./admin-header";
import { AdminStats } from "./admin-stats";
import { ModelPerformance } from "./model-performance";
import { HeartRateChart } from "./heart-rate-chart";
import { DataDistribution } from "./data-distribution";
import { ClassificationPanel } from "./classification-panel";
import { PatientsTable } from "./patients-table";
import {
  DatasetStats,
  HealthMessage,
  AIPoweredPanel,
} from "./bottom-panels";
import { Calendar, Download, RefreshCw } from "lucide-react";

export function AdminView() {
  return (
    <div className="flex flex-1 flex-col gap-4 overflow-y-auto">
      <AdminHeader />

      {/* Action Bar */}
      <div className="flex flex-wrap items-center gap-3">
        <button className="flex items-center gap-2 rounded-full bg-foreground px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-90">
          <RefreshCw className="h-4 w-4" />
          Actualiser les modeles
        </button>
        <button className="flex items-center gap-2 rounded-full border border-border bg-card px-4 py-2 text-sm text-foreground">
          <Calendar className="h-4 w-4" />
          10 Fevrier 2026
        </button>
        <button className="flex items-center gap-2 rounded-full border border-border bg-card px-4 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground">
          <Download className="h-4 w-4" />
          Exporter Rapport
        </button>
      </div>

      <div className="flex flex-1 gap-4">
        {/* Center Content */}
        <div className="flex flex-1 flex-col gap-4">
          <AdminStats />
          <ModelPerformance />
          <DataDistribution />
          <HeartRateChart />
          <PatientsTable />
          <ClassificationPanel />

          {/* Bottom Row */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <DatasetStats />
            <HealthMessage />
          </div>
        </div>

        {/* Right Panel */}
        <div className="hidden w-[280px] shrink-0 flex-col gap-4 xl:flex">
          {/* Quick KPIs */}
          <div className="rounded-2xl border border-border bg-card p-5">
            <h3 className="text-base font-semibold text-foreground">
              Indicateurs Cles
            </h3>
            <div className="mt-4 flex flex-col gap-3">
              {[
                {
                  label: "Sensibilite",
                  value: "93.7%",
                  color: "bg-emerald-500",
                },
                {
                  label: "Specificite",
                  value: "95.1%",
                  color: "bg-sky-500",
                },
                {
                  label: "Valeur Pred. Pos.",
                  value: "91.8%",
                  color: "bg-amber-500",
                },
                {
                  label: "Valeur Pred. Neg.",
                  value: "96.3%",
                  color: "bg-primary",
                },
                {
                  label: "AUC-ROC",
                  value: "0.974",
                  color: "bg-foreground",
                },
              ].map((kpi) => (
                <div
                  key={kpi.label}
                  className="flex items-center justify-between"
                >
                  <div className="flex items-center gap-2">
                    <div
                      className={`h-2 w-2 rounded-full ${kpi.color}`}
                    />
                    <span className="text-xs text-muted-foreground">
                      {kpi.label}
                    </span>
                  </div>
                  <span className="text-sm font-semibold text-foreground">
                    {kpi.value}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Model Log */}
          <div className="rounded-2xl border border-border bg-card p-5">
            <h3 className="text-base font-semibold text-foreground">
              Journal des Modeles
            </h3>
            <div className="mt-3 flex flex-col gap-2.5">
              {[
                {
                  time: "15:42",
                  event: "Random Forest entraine",
                  status: "success",
                },
                {
                  time: "14:18",
                  event: "Validation croisee (k=10)",
                  status: "success",
                },
                {
                  time: "12:05",
                  event: "Preprocessing des donnees",
                  status: "success",
                },
                {
                  time: "10:30",
                  event: "SVM - hyperparametres",
                  status: "warning",
                },
              ].map((log) => (
                <div
                  key={log.time}
                  className="flex items-start gap-2"
                >
                  <span className="mt-0.5 text-xs text-muted-foreground">
                    {log.time}
                  </span>
                  <div className="flex flex-1 items-center gap-1.5">
                    <div
                      className={`mt-1 h-1.5 w-1.5 shrink-0 rounded-full ${
                        log.status === "success"
                          ? "bg-emerald-500"
                          : "bg-amber-500"
                      }`}
                    />
                    <span className="text-xs text-foreground">
                      {log.event}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <AIPoweredPanel />
        </div>
      </div>
    </div>
  );
}
