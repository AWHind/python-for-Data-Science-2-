"use client";

import { useState } from "react";
import {
  CheckCircle,
  AlertTriangle,
  Search,
  ChevronLeft,
  ChevronRight,
  Eye,
  Calendar,
  Filter,
  Download,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface HistoryEntry {
  id: string;
  date: string;
  time: string;
  classId: number;
  classification: string;
  confidence: number;
  isNormal: boolean;
  heartRate: number;
  qrs: number;
  model: string;
}

const historyData: HistoryEntry[] = [
  { id: "PRED-A1B2C3", date: "10/02/2026", time: "14:32", classId: 1, classification: "Rythme Sinusal Normal", confidence: 97.2, isNormal: true, heartRate: 72, qrs: 80, model: "Random Forest" },
  { id: "PRED-D4E5F6", date: "08/02/2026", time: "09:15", classId: 2, classification: "Arythmie Ischemique", confidence: 87.5, isNormal: false, heartRate: 95, qrs: 112, model: "Random Forest" },
  { id: "PRED-G7H8I9", date: "05/02/2026", time: "16:48", classId: 1, classification: "Rythme Sinusal Normal", confidence: 94.1, isNormal: true, heartRate: 68, qrs: 78, model: "SVM" },
  { id: "PRED-J0K1L2", date: "01/02/2026", time: "11:22", classId: 1, classification: "Rythme Sinusal Normal", confidence: 96.8, isNormal: true, heartRate: 74, qrs: 82, model: "Random Forest" },
  { id: "PRED-M3N4O5", date: "28/01/2026", time: "08:05", classId: 3, classification: "PVC Detecte", confidence: 78.3, isNormal: false, heartRate: 88, qrs: 135, model: "KNN" },
  { id: "PRED-P6Q7R8", date: "25/01/2026", time: "13:50", classId: 1, classification: "Rythme Sinusal Normal", confidence: 98.1, isNormal: true, heartRate: 65, qrs: 76, model: "Random Forest" },
  { id: "PRED-S9T0U1", date: "20/01/2026", time: "10:30", classId: 4, classification: "Bloc AV Suspecte", confidence: 72.4, isNormal: false, heartRate: 52, qrs: 148, model: "SVM" },
  { id: "PRED-V2W3X4", date: "15/01/2026", time: "15:12", classId: 1, classification: "Rythme Sinusal Normal", confidence: 95.6, isNormal: true, heartRate: 70, qrs: 79, model: "Random Forest" },
];

export function PatientHistory() {
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedEntry, setSelectedEntry] = useState<HistoryEntry | null>(null);
  const [filterNormal, setFilterNormal] = useState<"all" | "normal" | "arythmie">("all");
  const pageSize = 5;

  const filtered = historyData.filter((e) => {
    const matchSearch =
      e.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      e.classification.toLowerCase().includes(searchTerm.toLowerCase());
    const matchFilter =
      filterNormal === "all" ||
      (filterNormal === "normal" && e.isNormal) ||
      (filterNormal === "arythmie" && !e.isNormal);
    return matchSearch && matchFilter;
  });

  const totalPages = Math.ceil(filtered.length / pageSize);
  const paginated = filtered.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const normalCount = historyData.filter((e) => e.isNormal).length;
  const abnormalCount = historyData.filter((e) => !e.isNormal).length;

  return (
    <div className="flex flex-col gap-6">
      {/* Page Header */}
      <div>
        <h2 className="text-2xl font-bold text-foreground">
          Historique des Predictions
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Consultez l{"'"}ensemble de vos analyses precedentes
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="flex items-center gap-4 rounded-2xl border border-border bg-card p-5">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-muted">
            <Calendar className="h-5 w-5 text-foreground" />
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">{historyData.length}</p>
            <p className="text-xs text-muted-foreground">Total predictions</p>
          </div>
        </div>
        <div className="flex items-center gap-4 rounded-2xl border border-border bg-card p-5">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-100">
            <CheckCircle className="h-5 w-5 text-emerald-600" />
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">{normalCount}</p>
            <p className="text-xs text-muted-foreground">Rythme normal</p>
          </div>
        </div>
        <div className="flex items-center gap-4 rounded-2xl border border-border bg-card p-5">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-red-100">
            <AlertTriangle className="h-5 w-5 text-red-600" />
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">{abnormalCount}</p>
            <p className="text-xs text-muted-foreground">Arythmie detectee</p>
          </div>
        </div>
      </div>

      {/* Table Card */}
      <div className="rounded-2xl border border-border bg-card p-5">
        {/* Toolbar */}
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-2">
            <Search className="h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Rechercher par ID ou classification..."
              value={searchTerm}
              onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }}
              className="w-64 bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground/50"
            />
          </div>
          <div className="flex items-center gap-2">
            <div className="flex rounded-lg border border-border">
              {(["all", "normal", "arythmie"] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => { setFilterNormal(f); setCurrentPage(1); }}
                  className={cn(
                    "px-3 py-1.5 text-xs font-medium transition-colors",
                    filterNormal === f ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {f === "all" ? "Tous" : f === "normal" ? "Normal" : "Arythmie"}
                </button>
              ))}
            </div>
            <button className="flex items-center gap-1.5 rounded-lg border border-border bg-transparent px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground">
              <Download className="h-3.5 w-3.5" />
              Exporter
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-border">
                <th className="pb-3 text-xs font-medium text-muted-foreground">ID</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Date</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Resultat</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Confiance</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">FC</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Modele</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Actions</th>
              </tr>
            </thead>
            <tbody>
              {paginated.map((entry) => (
                <tr key={entry.id} className="border-b border-border/50 transition-colors last:border-0 hover:bg-muted/30">
                  <td className="py-3 text-sm font-medium text-foreground">{entry.id}</td>
                  <td className="py-3 text-sm text-muted-foreground">{entry.date} {entry.time}</td>
                  <td className="py-3">
                    <span className={cn(
                      "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium",
                      entry.isNormal ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"
                    )}>
                      {entry.isNormal ? <CheckCircle className="h-3 w-3" /> : <AlertTriangle className="h-3 w-3" />}
                      {entry.classification}
                    </span>
                  </td>
                  <td className="py-3">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-14 overflow-hidden rounded-full bg-muted">
                        <div
                          className={cn("h-full rounded-full", entry.confidence >= 90 ? "bg-emerald-500" : entry.confidence >= 75 ? "bg-amber-500" : "bg-primary")}
                          style={{ width: `${entry.confidence}%` }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground">{entry.confidence}%</span>
                    </div>
                  </td>
                  <td className="py-3 text-sm text-foreground">{entry.heartRate} bpm</td>
                  <td className="py-3 text-xs text-muted-foreground">{entry.model}</td>
                  <td className="py-3">
                    <button
                      onClick={() => setSelectedEntry(entry)}
                      className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                    >
                      <Eye className="h-4 w-4" />
                      <span className="sr-only">Voir details</span>
                    </button>
                  </td>
                </tr>
              ))}
              {paginated.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-sm text-muted-foreground">
                    Aucun resultat trouve.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="mt-4 flex items-center justify-between">
          <span className="text-xs text-muted-foreground">{filtered.length} resultat(s)</span>
          <div className="flex items-center gap-1">
            <button onClick={() => setCurrentPage((p) => Math.max(1, p - 1))} disabled={currentPage === 1} className="flex h-8 w-8 items-center justify-center rounded-lg border border-border bg-transparent text-muted-foreground hover:text-foreground disabled:opacity-40">
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="px-3 text-xs text-foreground">{currentPage} / {totalPages || 1}</span>
            <button onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))} disabled={currentPage >= totalPages} className="flex h-8 w-8 items-center justify-center rounded-lg border border-border bg-transparent text-muted-foreground hover:text-foreground disabled:opacity-40">
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Detail Modal */}
      {selectedEntry && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/40 p-4">
          <div className="w-full max-w-md rounded-2xl border border-border bg-card p-6 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-foreground">
                Detail de la Prediction
              </h3>
              <button onClick={() => setSelectedEntry(null)} className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className={cn(
              "mt-4 flex items-center gap-3 rounded-xl p-4",
              selectedEntry.isNormal ? "bg-emerald-50" : "bg-red-50"
            )}>
              {selectedEntry.isNormal ? (
                <CheckCircle className="h-6 w-6 text-emerald-600" />
              ) : (
                <AlertTriangle className="h-6 w-6 text-red-600" />
              )}
              <div>
                <p className={cn("text-sm font-bold", selectedEntry.isNormal ? "text-emerald-800" : "text-red-800")}>
                  {selectedEntry.classification}
                </p>
                <p className={cn("text-xs", selectedEntry.isNormal ? "text-emerald-600" : "text-red-600")}>
                  Confiance: {selectedEntry.confidence}%
                </p>
              </div>
            </div>

            <div className="mt-4 flex flex-col gap-2">
              {[
                { label: "ID Prediction", value: selectedEntry.id },
                { label: "Date", value: `${selectedEntry.date} a ${selectedEntry.time}` },
                { label: "Frequence cardiaque", value: `${selectedEntry.heartRate} bpm` },
                { label: "Duree QRS", value: `${selectedEntry.qrs} ms` },
                { label: "Modele", value: selectedEntry.model },
                { label: "Classe", value: `Classe ${selectedEntry.classId}` },
              ].map((item) => (
                <div key={item.label} className="flex justify-between rounded-lg bg-muted/50 px-3 py-2">
                  <span className="text-xs text-muted-foreground">{item.label}</span>
                  <span className="text-xs font-medium text-foreground">{item.value}</span>
                </div>
              ))}
            </div>

            <button
              onClick={() => setSelectedEntry(null)}
              className="mt-5 w-full rounded-xl bg-foreground py-2.5 text-sm font-medium text-background transition-opacity hover:opacity-90"
            >
              Fermer
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
