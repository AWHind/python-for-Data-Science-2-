"use client";

import { useState } from "react";
import { Search, Filter, ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface PatientRow {
  id: string;
  age: number;
  sex: string;
  heartRate: number;
  qrs: number;
  classification: string;
  classId: number;
  confidence: number;
}

const patients: PatientRow[] = [
  { id: "P-001", age: 54, sex: "M", heartRate: 72, qrs: 80, classification: "Normal", classId: 1, confidence: 97.2 },
  { id: "P-002", age: 67, sex: "F", heartRate: 88, qrs: 95, classification: "Arythmie ischemique", classId: 2, confidence: 91.5 },
  { id: "P-003", age: 43, sex: "M", heartRate: 65, qrs: 78, classification: "Normal", classId: 1, confidence: 98.1 },
  { id: "P-004", age: 71, sex: "M", heartRate: 102, qrs: 130, classification: "BBG", classId: 5, confidence: 87.3 },
  { id: "P-005", age: 58, sex: "F", heartRate: 76, qrs: 82, classification: "Normal", classId: 1, confidence: 95.8 },
  { id: "P-006", age: 62, sex: "M", heartRate: 91, qrs: 110, classification: "PVC", classId: 3, confidence: 89.6 },
  { id: "P-007", age: 49, sex: "F", heartRate: 68, qrs: 75, classification: "Normal", classId: 1, confidence: 96.4 },
  { id: "P-008", age: 75, sex: "M", heartRate: 55, qrs: 145, classification: "Bloc AV", classId: 4, confidence: 82.1 },
];

function getClassBadge(classId: number) {
  const styles: Record<number, string> = {
    1: "bg-emerald-100 text-emerald-700",
    2: "bg-red-100 text-red-700",
    3: "bg-amber-100 text-amber-700",
    4: "bg-sky-100 text-sky-700",
    5: "bg-violet-100 text-violet-700",
  };
  return styles[classId] || "bg-muted text-muted-foreground";
}

export function PatientsTable() {
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 6;

  const filtered = patients.filter(
    (p) =>
      p.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.classification.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const totalPages = Math.ceil(filtered.length / pageSize);
  const paginated = filtered.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );

  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <h3 className="text-base font-semibold text-foreground">
          Gestion des Patients
        </h3>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-1.5">
            <Search className="h-3.5 w-3.5 text-muted-foreground" />
            <input
              type="text"
              placeholder="Rechercher..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              className="w-32 bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground/50"
            />
          </div>
          <button className="flex items-center gap-1.5 rounded-lg border border-border bg-transparent px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground">
            <Filter className="h-3.5 w-3.5" />
            Filtrer
          </button>
        </div>
      </div>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-border">
              <th className="pb-3 text-xs font-medium text-muted-foreground">
                ID
              </th>
              <th className="pb-3 text-xs font-medium text-muted-foreground">
                Age
              </th>
              <th className="pb-3 text-xs font-medium text-muted-foreground">
                Sexe
              </th>
              <th className="pb-3 text-xs font-medium text-muted-foreground">
                FC (bpm)
              </th>
              <th className="pb-3 text-xs font-medium text-muted-foreground">
                QRS (ms)
              </th>
              <th className="pb-3 text-xs font-medium text-muted-foreground">
                Classification
              </th>
              <th className="pb-3 text-xs font-medium text-muted-foreground">
                Confiance
              </th>
            </tr>
          </thead>
          <tbody>
            {paginated.map((patient) => (
              <tr
                key={patient.id}
                className="border-b border-border/50 transition-colors last:border-0 hover:bg-muted/30"
              >
                <td className="py-3 text-sm font-medium text-foreground">
                  {patient.id}
                </td>
                <td className="py-3 text-sm text-foreground">{patient.age}</td>
                <td className="py-3 text-sm text-foreground">{patient.sex}</td>
                <td className="py-3 text-sm text-foreground">
                  {patient.heartRate}
                </td>
                <td className="py-3 text-sm text-foreground">{patient.qrs}</td>
                <td className="py-3">
                  <span
                    className={cn(
                      "rounded-full px-2.5 py-1 text-xs font-medium",
                      getClassBadge(patient.classId)
                    )}
                  >
                    {patient.classification}
                  </span>
                </td>
                <td className="py-3">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
                      <div
                        className={cn(
                          "h-full rounded-full",
                          patient.confidence >= 95
                            ? "bg-emerald-500"
                            : patient.confidence >= 85
                              ? "bg-amber-500"
                              : "bg-primary"
                        )}
                        style={{ width: `${patient.confidence}%` }}
                      />
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {patient.confidence}%
                    </span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="mt-4 flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {filtered.length} patient(s) trouve(s)
        </span>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-border bg-transparent text-muted-foreground transition-colors hover:text-foreground disabled:opacity-40"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="px-3 text-xs text-foreground">
            {currentPage} / {totalPages || 1}
          </span>
          <button
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage >= totalPages}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-border bg-transparent text-muted-foreground transition-colors hover:text-foreground disabled:opacity-40"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
