"use client";

import { useState } from "react";
import {
  UserCheck,
  UserX,
  Clock,
  CheckCircle,
  XCircle,
  Search,
  User,
  Mail,
  Phone,
  Calendar,
  X,
  Eye,
} from "lucide-react";
import { useAuth, type PendingRegistration } from "@/lib/auth-context";
import { cn } from "@/lib/utils";

export function AdminValidations() {
  const { pendingRegistrations, approveRegistration, rejectRegistration } = useAuth();
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedReg, setSelectedReg] = useState<PendingRegistration | null>(null);
  const [filterStatus, setFilterStatus] = useState<"all" | "pending" | "active" | "rejected">("all");

  const filtered = pendingRegistrations.filter((r) => {
    const matchSearch =
      r.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      r.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchFilter =
      filterStatus === "all" || r.status === filterStatus;
    return matchSearch && matchFilter;
  });

  const pendingCount = pendingRegistrations.filter((r) => r.status === "pending").length;
  const approvedCount = pendingRegistrations.filter((r) => r.status === "active").length;
  const rejectedCount = pendingRegistrations.filter((r) => r.status === "rejected").length;

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "pending":
        return (
          <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-100 px-2.5 py-1 text-xs font-medium text-amber-700">
            <Clock className="h-3 w-3" />
            En attente
          </span>
        );
      case "active":
        return (
          <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-100 px-2.5 py-1 text-xs font-medium text-emerald-700">
            <CheckCircle className="h-3 w-3" />
            Approuve
          </span>
        );
      case "rejected":
        return (
          <span className="inline-flex items-center gap-1.5 rounded-full bg-red-100 px-2.5 py-1 text-xs font-medium text-red-700">
            <XCircle className="h-3 w-3" />
            Refuse
          </span>
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-2xl font-bold text-foreground">
          Validation des Inscriptions
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Consultez et validez les demandes d{"'"}inscription des patients
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="flex items-center gap-4 rounded-2xl border border-border bg-card p-5">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-amber-100">
            <Clock className="h-5 w-5 text-amber-600" />
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">{pendingCount}</p>
            <p className="text-xs text-muted-foreground">En attente</p>
          </div>
        </div>
        <div className="flex items-center gap-4 rounded-2xl border border-border bg-card p-5">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-100">
            <CheckCircle className="h-5 w-5 text-emerald-600" />
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">{approvedCount}</p>
            <p className="text-xs text-muted-foreground">Approuvees</p>
          </div>
        </div>
        <div className="flex items-center gap-4 rounded-2xl border border-border bg-card p-5">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-red-100">
            <XCircle className="h-5 w-5 text-red-600" />
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">{rejectedCount}</p>
            <p className="text-xs text-muted-foreground">Refusees</p>
          </div>
        </div>
      </div>

      {/* Table Card */}
      <div className="rounded-2xl border border-border bg-card p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <h3 className="text-base font-semibold text-foreground">
            Demandes d{"'"}inscription
          </h3>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-2">
              <Search className="h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="Rechercher..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-48 bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground/50"
              />
            </div>
            <div className="flex rounded-lg border border-border">
              {(["all", "pending", "active", "rejected"] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => setFilterStatus(f)}
                  className={cn(
                    "px-3 py-1.5 text-xs font-medium transition-colors",
                    filterStatus === f
                      ? "bg-foreground text-background"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {f === "all"
                    ? "Tous"
                    : f === "pending"
                      ? "En attente"
                      : f === "active"
                        ? "Approuve"
                        : "Refuse"}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-border">
                <th className="pb-3 text-xs font-medium text-muted-foreground">Nom</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Email</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Telephone</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Age / Sexe</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Date inscription</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Statut</th>
                <th className="pb-3 text-xs font-medium text-muted-foreground">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((reg) => (
                <tr
                  key={reg.id}
                  className="border-b border-border/50 transition-colors last:border-0 hover:bg-muted/30"
                >
                  <td className="py-3 text-sm font-medium text-foreground">
                    {reg.name}
                  </td>
                  <td className="py-3 text-sm text-muted-foreground">
                    {reg.email}
                  </td>
                  <td className="py-3 text-sm text-muted-foreground">
                    {reg.phone}
                  </td>
                  <td className="py-3 text-sm text-foreground">
                    {reg.age} ans / {reg.sex === "M" ? "Masculin" : "Feminin"}
                  </td>
                  <td className="py-3 text-xs text-muted-foreground">
                    {reg.registeredAt}
                  </td>
                  <td className="py-3">{getStatusBadge(reg.status)}</td>
                  <td className="py-3">
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setSelectedReg(reg)}
                        className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                        title="Voir details"
                      >
                        <Eye className="h-4 w-4" />
                        <span className="sr-only">Voir details</span>
                      </button>
                      {reg.status === "pending" && (
                        <>
                          <button
                            onClick={() => approveRegistration(reg.id)}
                            className="flex h-8 w-8 items-center justify-center rounded-lg text-emerald-600 transition-colors hover:bg-emerald-100"
                            title="Approuver"
                          >
                            <UserCheck className="h-4 w-4" />
                            <span className="sr-only">Approuver</span>
                          </button>
                          <button
                            onClick={() => rejectRegistration(reg.id)}
                            className="flex h-8 w-8 items-center justify-center rounded-lg text-red-600 transition-colors hover:bg-red-100"
                            title="Refuser"
                          >
                            <UserX className="h-4 w-4" />
                            <span className="sr-only">Refuser</span>
                          </button>
                        </>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-sm text-muted-foreground">
                    Aucune demande d{"'"}inscription trouvee.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="mt-4">
          <span className="text-xs text-muted-foreground">
            {filtered.length} demande(s)
          </span>
        </div>
      </div>

      {/* Detail Modal */}
      {selectedReg && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/40 p-4">
          <div className="w-full max-w-md rounded-2xl border border-border bg-card p-6 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-foreground">
                Detail de l{"'"}inscription
              </h3>
              <button
                onClick={() => setSelectedReg(null)}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="mt-4 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
                <User className="h-7 w-7 text-primary" />
              </div>
              <div>
                <p className="text-lg font-bold text-foreground">{selectedReg.name}</p>
                <div className="mt-1">{getStatusBadge(selectedReg.status)}</div>
              </div>
            </div>

            <div className="mt-4 flex flex-col gap-2">
              {[
                { icon: Mail, label: "Email", value: selectedReg.email },
                { icon: Phone, label: "Telephone", value: selectedReg.phone },
                { icon: Calendar, label: "Age", value: `${selectedReg.age} ans` },
                { icon: User, label: "Sexe", value: selectedReg.sex === "M" ? "Masculin" : "Feminin" },
                { icon: Clock, label: "Date inscription", value: selectedReg.registeredAt },
              ].map((item) => (
                <div key={item.label} className="flex items-center gap-3 rounded-lg bg-muted/50 px-3 py-2">
                  <item.icon className="h-4 w-4 text-muted-foreground" />
                  <div className="flex flex-1 justify-between">
                    <span className="text-xs text-muted-foreground">{item.label}</span>
                    <span className="text-xs font-medium text-foreground">{item.value}</span>
                  </div>
                </div>
              ))}
            </div>

            {selectedReg.status === "pending" && (
              <div className="mt-5 flex gap-3">
                <button
                  onClick={() => {
                    approveRegistration(selectedReg.id);
                    setSelectedReg(null);
                  }}
                  className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-emerald-600 py-2.5 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90"
                >
                  <UserCheck className="h-4 w-4" />
                  Approuver
                </button>
                <button
                  onClick={() => {
                    rejectRegistration(selectedReg.id);
                    setSelectedReg(null);
                  }}
                  className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-red-600 py-2.5 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90"
                >
                  <UserX className="h-4 w-4" />
                  Refuser
                </button>
              </div>
            )}

            <button
              onClick={() => setSelectedReg(null)}
              className="mt-3 w-full rounded-xl border border-border bg-transparent py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-muted"
            >
              Fermer
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
