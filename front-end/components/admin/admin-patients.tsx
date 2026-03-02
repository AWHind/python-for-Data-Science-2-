"use client";

import { useState } from "react";
import {
  Search,
  Filter,
  ChevronLeft,
  ChevronRight,
  Eye,
  X,
  User,
  Heart,
  Activity,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface PatientRow {
  id: string;
  name: string;
  age: number;
  sex: string;
  heartRate: number;
  qrs: number;
  classification: string;
  classId: number;
  confidence: number;
  predictions: number;
  lastVisit: string;
}

const patients: PatientRow[] = [
  { id: "P-001", name: "Ahmed B.", age: 54, sex: "M", heartRate: 72, qrs: 80, classification: "Normal", classId: 1, confidence: 97.2, predictions: 5, lastVisit: "10/02/2026" },
  { id: "P-002", name: "Fatima Z.", age: 67, sex: "F", heartRate: 88, qrs: 95, classification: "Arythmie ischemique", classId: 2, confidence: 91.5, predictions: 3, lastVisit: "08/02/2026" },
  { id: "P-003", name: "Karim M.", age: 43, sex: "M", heartRate: 65, qrs: 78, classification: "Normal", classId: 1, confidence: 98.1, predictions: 7, lastVisit: "07/02/2026" },
  { id: "P-004", name: "Youssef H.", age: 71, sex: "M", heartRate: 102, qrs: 130, classification: "BBG", classId: 5, confidence: 87.3, predictions: 2, lastVisit: "05/02/2026" },
  { id: "P-005", name: "Amina L.", age: 58, sex: "F", heartRate: 76, qrs: 82, classification: "Normal", classId: 1, confidence: 95.8, predictions: 4, lastVisit: "04/02/2026" },
  { id: "P-006", name: "Omar R.", age: 62, sex: "M", heartRate: 91, qrs: 110, classification: "PVC", classId: 3, confidence: 89.6, predictions: 6, lastVisit: "03/02/2026" },
  { id: "P-007", name: "Leila K.", age: 49, sex: "F", heartRate: 68, qrs: 75, classification: "Normal", classId: 1, confidence: 96.4, predictions: 8, lastVisit: "02/02/2026" },
  { id: "P-008", name: "Hassan T.", age: 75, sex: "M", heartRate: 55, qrs: 145, classification: "Bloc AV", classId: 4, confidence: 82.1, predictions: 3, lastVisit: "01/02/2026" },
  { id: "P-009", name: "Nadia S.", age: 39, sex: "F", heartRate: 78, qrs: 84, classification: "Normal", classId: 1, confidence: 94.5, predictions: 2, lastVisit: "30/01/2026" },
  { id: "P-010", name: "Rachid D.", age: 66, sex: "M", heartRate: 96, qrs: 118, classification: "BBD", classId: 6, confidence: 85.2, predictions: 4, lastVisit: "28/01/2026" },
];

function getClassBadge(classId: number) {
  const styles: Record<number, string> = {
    1: "bg-emerald-100 text-emerald-700",
    2: "bg-red-100 text-red-700",
    3: "bg-amber-100 text-amber-700",
    4: "bg-sky-100 text-sky-700",
    5: "bg-violet-100 text-violet-700",
    6: "bg-orange-100 text-orange-700",
  };
  return styles[classId] || "bg-muted text-muted-foreground";
}

export function AdminPatients() {
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedPatient, setSelectedPatient] = useState<PatientRow | null>(null);
  const pageSize = 7;

  const filtered = patients.filter(
    (p) =>
      p.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.classification.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const totalPages = Math.ceil(filtered.length / pageSize);
  const paginated = filtered.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-2xl font-bold text-foreground">Gestion des Patients</h2>
        <p className="mt-1 text-sm text-muted-foreground">Consultez les informations et resultats de prediction des patients</p>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="flex items-center gap-4 rounded-2xl border border-border bg-card p-5">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
            <User className="h-5 w-5 text-primary" />
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">{patients.length}</p>
            <p className="text-xs text-muted-foreground">Patients enregistres</p>
          </div>
        </div>
        <div className="flex items-center gap-4 rounded-2xl border border-border bg-card p-5">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-100">
            <Heart className="h-5 w-5 text-emerald-600" />
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">{patients.filter((p) => p.classId === 1).length}</p>
            <p className="text-xs text-muted-foreground">Rythme normal</p>
          </div>
        </div>
        <div className="flex items-center gap-4 rounded-2xl border border-border bg-card p-5">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-amber-100">
            <Activity className="h-5 w-5 text-amber-600" />
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">{patients.reduce((s, p) => s + p.predictions, 0)}</p>
            <p className="text-xs text-muted-foreground">Total predictions</p>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <h3 className="text-base font-semibold text-foreground">Liste des Patients</h3>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-2">
              <Search className="h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="Rechercher..."
                value={searchTerm}
                onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }}
                className="w-48 bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground/50"
              />
            </div>
            <button className="flex items-center gap-1.5 rounded-lg border border-border bg-transparent px-3 py-2 text-xs text-muted-foreground transition-colors hover:text-foreground">
              <Filter className="h-3.5 w-3.5" />
              Filtrer
            </button>
          </div>
        </div>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-border">
                <th className="pb-3 text-xs font-medium text-muted-foreground">ID</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Nom</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Age</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Sexe</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">FC</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Classification</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Confiance</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Analyses</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Derniere visite</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Actions</th>
              </tr>
            </thead>
            <tbody>
              {paginated.map((p) => (
                <tr key={p.id} className="border-b border-border/50 transition-colors last:border-0 hover:bg-muted/30">
                  <td className="py-3 text-sm font-medium text-foreground">{p.id}</td>
                  <td className="py-3 text-sm text-foreground">{p.name}</td>
                  <td className="py-3 text-sm text-foreground">{p.age}</td>
                  <td className="py-3 text-sm text-foreground">{p.sex}</td>
                  <td className="py-3 text-sm text-foreground">{p.heartRate} bpm</td>
                  <td className="py-3">
                    <span className={cn("rounded-full px-2.5 py-1 text-xs font-medium", getClassBadge(p.classId))}>
                      {p.classification}
                    </span>
                  </td>
                  <td className="py-3">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-14 overflow-hidden rounded-full bg-muted">
                        <div className={cn("h-full rounded-full", p.confidence >= 95 ? "bg-emerald-500" : p.confidence >= 85 ? "bg-amber-500" : "bg-primary")} style={{ width: `${p.confidence}%` }} />
                      </div>
                      <span className="text-xs text-muted-foreground">{p.confidence}%</span>
                    </div>
                  </td>
                  <td className="py-3 text-sm text-muted-foreground">{p.predictions}</td>
                  <td className="py-3 text-xs text-muted-foreground">{p.lastVisit}</td>
                  <td className="py-3">
                    <button onClick={() => setSelectedPatient(p)} className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">
                      <Eye className="h-4 w-4" />
                      <span className="sr-only">Voir patient</span>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-4 flex items-center justify-between">
          <span className="text-xs text-muted-foreground">{filtered.length} patient(s)</span>
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
      {selectedPatient && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/40 p-4">
          <div className="w-full max-w-lg rounded-2xl border border-border bg-card p-6 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-foreground">Fiche Patient</h3>
              <button onClick={() => setSelectedPatient(null)} className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-4 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
                <User className="h-7 w-7 text-primary" />
              </div>
              <div>
                <p className="text-lg font-bold text-foreground">{selectedPatient.name}</p>
                <p className="text-sm text-muted-foreground">ID: {selectedPatient.id} | {selectedPatient.age} ans | {selectedPatient.sex === "M" ? "Masculin" : "Feminin"}</p>
              </div>
            </div>
            <div className={cn("mt-4 flex items-center gap-3 rounded-xl p-3", selectedPatient.classId === 1 ? "bg-emerald-50" : "bg-red-50")}>
              <Heart className={cn("h-5 w-5", selectedPatient.classId === 1 ? "text-emerald-600" : "text-red-600")} />
              <div>
                <p className={cn("text-sm font-semibold", selectedPatient.classId === 1 ? "text-emerald-800" : "text-red-800")}>{selectedPatient.classification}</p>
                <p className={cn("text-xs", selectedPatient.classId === 1 ? "text-emerald-600" : "text-red-600")}>Confiance: {selectedPatient.confidence}%</p>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-2">
              {[
                { label: "Frequence cardiaque", value: `${selectedPatient.heartRate} bpm` },
                { label: "Duree QRS", value: `${selectedPatient.qrs} ms` },
                { label: "Nombre d'analyses", value: `${selectedPatient.predictions}` },
                { label: "Derniere visite", value: selectedPatient.lastVisit },
              ].map((item) => (
                <div key={item.label} className="rounded-lg bg-muted/50 px-3 py-2">
                  <p className="text-xs text-muted-foreground">{item.label}</p>
                  <p className="text-sm font-medium text-foreground">{item.value}</p>
                </div>
              ))}
            </div>
            <button onClick={() => setSelectedPatient(null)} className="mt-5 w-full rounded-xl bg-foreground py-2.5 text-sm font-medium text-background hover:opacity-90">
              Fermer
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
