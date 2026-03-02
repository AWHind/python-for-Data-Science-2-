"use client";

import { useState } from "react";
import {
  Brain,
  Database,
  Shield,
  Bell,
  RefreshCw,
  Save,
  CheckCircle,
  AlertTriangle,
  Server,
  Key,
  Clock,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/lib/auth-context";


interface SettingToggleProps {
  label: string;
  description: string;
  enabled: boolean;
  onToggle: () => void;
}

function SettingToggle({
                         label,
                         description,
                         enabled,
                         onToggle,
                       }: SettingToggleProps) {
  return (
      <div className="flex items-center justify-between rounded-xl border border-border bg-background p-4">
        <div className="flex-1 pr-4">
          <p className="text-sm font-medium text-foreground">{label}</p>
          <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>
        </div>
        <button
            onClick={onToggle}
            className={cn(
                "flex h-6 w-11 items-center rounded-full transition-colors",
                enabled ? "bg-primary" : "bg-muted"
            )}
        >
          <div
              className={cn(
                  "h-5 w-5 rounded-full bg-card shadow transition-transform",
                  enabled ? "translate-x-[22px]" : "translate-x-0.5"
              )}
          />
        </button>
      </div>
  );
}

export function AdminSettings()
 {
  const { user } = useAuth();
  const [saved, setSaved] = useState(false);

  /* ================== STATES ================== */
  const [settings, setSettings] = useState({
    autoRetrain: true,
    notifications: true,
    emailAlerts: false,
    auditLog: true,
    twoFactor: false,
    dataBackup: true,
  });

  const [adminInfo, setAdminInfo] = useState({
    name: user?.name || "Dr. Martin",
    email: user?.email || "admin@cardiosense.com",
  });

  const [modelConfig, setModelConfig] = useState({
    model: "random-forest",
    cv: "10",
    threshold: "0.5",
  });

  /* ================== HANDLERS ================== */
  const toggleSetting = (key: keyof typeof settings) => {
    setSettings((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleSave = () => {
    console.log("ADMIN SETTINGS:", {
      settings,
      adminInfo,
      modelConfig,
    });
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
      <div className="flex flex-col gap-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-foreground">Paramètres</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Configuration générale du système CardioSense
            </p>
          </div>
          <button
              onClick={handleSave}
              className={cn(
                  "flex items-center gap-2 rounded-xl px-5 py-2.5 text-sm font-semibold",
                  saved
                      ? "bg-emerald-500 text-white"
                      : "bg-primary text-primary-foreground hover:opacity-90"
              )}
          >
            {saved ? <CheckCircle className="h-4 w-4" /> : <Save className="h-4 w-4" />}
            {saved ? "Enregistré" : "Sauvegarder"}
          </button>
        </div>

        <div className="flex flex-col gap-6 xl:flex-row">
          {/* LEFT */}
          <div className="flex flex-1 flex-col gap-6">
            {/* MODEL */}
            <div className="rounded-2xl border border-border bg-card p-5">
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                <h3 className="text-base font-semibold">Configuration du Modèle</h3>
              </div>

              <div className="mt-4 flex flex-col gap-4">
                <select
                    value={modelConfig.model}
                    onChange={(e) =>
                        setModelConfig({ ...modelConfig, model: e.target.value })
                    }
                    className="rounded-lg border border-border bg-background px-3 py-2.5 text-sm"
                >
                  <option value="random-forest">Random Forest</option>
                  <option value="svm">SVM</option>
                  <option value="knn">KNN</option>
                  <option value="decision-tree">Decision Tree</option>
                </select>

                <select
                    value={modelConfig.cv}
                    onChange={(e) =>
                        setModelConfig({ ...modelConfig, cv: e.target.value })
                    }
                    className="rounded-lg border border-border bg-background px-3 py-2.5 text-sm"
                >
                  <option value="5">k = 5</option>
                  <option value="10">k = 10</option>
                  <option value="15">k = 15</option>
                </select>

                <input
                    value={modelConfig.threshold}
                    onChange={(e) =>
                        setModelConfig({ ...modelConfig, threshold: e.target.value })
                    }
                    className="rounded-lg border border-border bg-background px-3 py-2.5 text-sm"
                />

                <button className="flex items-center gap-2 self-start rounded-xl bg-foreground px-5 py-2.5 text-sm text-background">
                  <RefreshCw className="h-4 w-4" />
                  Ré-entraîner le modèle
                </button>
              </div>
            </div>

            {/* DATASET */}
            <div className="rounded-2xl border border-border bg-card p-5">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                {[
                  { icon: Server, label: "Dataset", value: "UCI Arrhythmia" },
                  { icon: Database, label: "Instances", value: "452" },
                  { icon: Brain, label: "Classes", value: "16" },
                  { icon: Clock, label: "Dernière MAJ", value: "10/02/2026" },
                ].map((item) => (
                    <div
                        key={item.label}
                        className="flex items-center gap-3 rounded-xl border border-border bg-background p-3"
                    >
                      <item.icon className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <p className="text-xs text-muted-foreground">{item.label}</p>
                        <p className="text-sm font-medium">{item.value}</p>
                      </div>
                    </div>
                ))}
              </div>
            </div>
          </div>

          {/* RIGHT */}
          <div className="flex w-full flex-col gap-6 xl:w-[400px]">
            {/* SECURITY */}
            <div className="rounded-2xl border border-border bg-card p-5">
              <Shield className="mb-3 h-5 w-5" />
              <SettingToggle
                  label="Double authentification"
                  description="Sécurité renforcée"
                  enabled={settings.twoFactor}
                  onToggle={() => toggleSetting("twoFactor")}
              />
              <SettingToggle
                  label="Journal d'audit"
                  description="Tracer les actions admin"
                  enabled={settings.auditLog}
                  onToggle={() => toggleSetting("auditLog")}
              />
              <SettingToggle
                  label="Sauvegarde automatique"
                  description="Backup quotidien"
                  enabled={settings.dataBackup}
                  onToggle={() => toggleSetting("dataBackup")}
              />
            </div>

            {/* NOTIFS */}
            <div className="rounded-2xl border border-border bg-card p-5">
              <Bell className="mb-3 h-5 w-5" />
              <SettingToggle
                  label="Notifications système"
                  description="Alertes importantes"
                  enabled={settings.notifications}
                  onToggle={() => toggleSetting("notifications")}
              />
              <SettingToggle
                  label="Alertes email"
                  description="Envoi par email"
                  enabled={settings.emailAlerts}
                  onToggle={() => toggleSetting("emailAlerts")}
              />
              <SettingToggle
                  label="Ré-entrainement auto"
                  description="Selon nouvelles données"
                  enabled={settings.autoRetrain}
                  onToggle={() => toggleSetting("autoRetrain")}
              />
            </div>

            {/* ADMIN */}
            <div className="rounded-2xl border border-border bg-card p-5">
              <Key className="mb-3 h-5 w-5" />
              <input
                  value={adminInfo.name}
                  onChange={(e) =>
                      setAdminInfo({ ...adminInfo, name: e.target.value })
                  }
                  className="mb-2 w-full rounded-lg border border-border px-3 py-2 text-sm"
              />
              <input
                  value={adminInfo.email}
                  onChange={(e) =>
                      setAdminInfo({ ...adminInfo, email: e.target.value })
                  }
                  className="w-full rounded-lg border border-border px-3 py-2 text-sm"
              />
            </div>
          </div>
        </div>
      </div>
  );
}
