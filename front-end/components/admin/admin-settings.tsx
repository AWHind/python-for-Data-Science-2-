"use client";

import { useState } from "react";
import {
  Brain,
  Database,
  Shield,
  Bell,
  Save,
  CheckCircle,
  Server,
  Key,
  Clock,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { useAuth } from "@/lib/auth-context";

/* ================= TOGGLE COMPONENT ================= */

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

/* ================= ADMIN SETTINGS ================= */

export function AdminSettings() {
  const { user } = useAuth();
  const [saved, setSaved] = useState(false);

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

  const toggleSetting = (key: keyof typeof settings) => {
    setSettings((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const handleSave = () => {
    console.log("SETTINGS SAVED", {
      settings,
      adminInfo,
    });

    setSaved(true);

    setTimeout(() => {
      setSaved(false);
    }, 2000);
  };

  return (
    <div className="flex flex-col gap-6">

      {/* HEADER */}

      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">
            CardioSense Admin Settings
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            AI system configuration and security settings
          </p>
        </div>

        <button
          onClick={handleSave}
          className={cn(
            "flex items-center gap-2 rounded-xl px-5 py-2.5 text-sm font-semibold",
            saved
              ? "bg-emerald-500 text-white"
              : "bg-primary text-primary-foreground"
          )}
        >
          {saved ? <CheckCircle className="h-4 w-4" /> : <Save className="h-4 w-4" />}
          {saved ? "Saved" : "Save Settings"}
        </button>
      </div>

      <div className="flex flex-col gap-6 xl:flex-row">

        {/* LEFT SIDE */}

        <div className="flex flex-1 flex-col gap-6">

          {/* AI SYSTEM MONITORING */}

          <div className="rounded-2xl border border-border bg-card p-5">

            <div className="flex items-center justify-between">

              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                <h3 className="text-base font-semibold">
                  AI System Monitoring
                </h3>
              </div>

              <span className="rounded-lg bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-600">
                Online
              </span>

            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">

              <div className="flex items-center gap-3 rounded-xl border border-border bg-background p-3">
                <Brain className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Active Model</p>
                  <p className="text-sm font-medium">Random Forest</p>
                </div>
              </div>

              <div className="flex items-center gap-3 rounded-xl border border-border bg-background p-3">
                <Database className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Predictions Today</p>
                  <p className="text-sm font-medium">124</p>
                </div>
              </div>

              <div className="flex items-center gap-3 rounded-xl border border-border bg-background p-3">
                <CheckCircle className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Model Accuracy</p>
                  <p className="text-sm font-medium">86%</p>
                </div>
              </div>

              <div className="flex items-center gap-3 rounded-xl border border-border bg-background p-3">
                <Server className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">System Health</p>
                  <p className="text-sm font-medium text-emerald-600">Healthy</p>
                </div>
              </div>

            </div>

          </div>

          {/* DATASET INFO */}

          <div className="rounded-2xl border border-border bg-card p-5">

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">

              <div className="flex items-center gap-3 rounded-xl border border-border bg-background p-3">
                <Server className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Dataset</p>
                  <p className="text-sm font-medium">UCI Arrhythmia</p>
                </div>
              </div>

              <div className="flex items-center gap-3 rounded-xl border border-border bg-background p-3">
                <Database className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Instances</p>
                  <p className="text-sm font-medium">452</p>
                </div>
              </div>

              <div className="flex items-center gap-3 rounded-xl border border-border bg-background p-3">
                <Brain className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Classes</p>
                  <p className="text-sm font-medium">16</p>
                </div>
              </div>

              <div className="flex items-center gap-3 rounded-xl border border-border bg-background p-3">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-xs text-muted-foreground">Last Update</p>
                  <p className="text-sm font-medium">10/02/2026</p>
                </div>
              </div>

            </div>

          </div>

        </div>

        {/* RIGHT SIDE */}

        <div className="flex w-full flex-col gap-6 xl:w-[380px]">

          {/* SECURITY */}

          <div className="rounded-2xl border border-border bg-card p-5">

            <Shield className="mb-3 h-5 w-5" />

            <SettingToggle
              label="Two Factor Authentication"
              description="Increase admin security"
              enabled={settings.twoFactor}
              onToggle={() => toggleSetting("twoFactor")}
            />

            <SettingToggle
              label="Audit Logs"
              description="Track admin actions"
              enabled={settings.auditLog}
              onToggle={() => toggleSetting("auditLog")}
            />

            <SettingToggle
              label="Automatic Backup"
              description="Daily database backup"
              enabled={settings.dataBackup}
              onToggle={() => toggleSetting("dataBackup")}
            />

          </div>

          {/* NOTIFICATIONS */}

          <div className="rounded-2xl border border-border bg-card p-5">

            <Bell className="mb-3 h-5 w-5" />

            <SettingToggle
              label="System Notifications"
              description="Receive important alerts"
              enabled={settings.notifications}
              onToggle={() => toggleSetting("notifications")}
            />

            <SettingToggle
              label="Email Alerts"
              description="Send notifications via email"
              enabled={settings.emailAlerts}
              onToggle={() => toggleSetting("emailAlerts")}
            />

            <SettingToggle
              label="Auto Retrain"
              description="Retrain model when new data arrives"
              enabled={settings.autoRetrain}
              onToggle={() => toggleSetting("autoRetrain")}
            />

          </div>

          {/* ADMIN PROFILE */}

          <div className="rounded-2xl border border-border bg-card p-5">

            <Key className="mb-3 h-5 w-5" />

            <input
              value={adminInfo.name}
              onChange={(e) =>
                setAdminInfo({
                  ...adminInfo,
                  name: e.target.value,
                })
              }
              className="mb-2 w-full rounded-lg border border-border px-3 py-2 text-sm"
            />

            <input
              value={adminInfo.email}
              onChange={(e) =>
                setAdminInfo({
                  ...adminInfo,
                  email: e.target.value,
                })
              }
              className="w-full rounded-lg border border-border px-3 py-2 text-sm"
            />

          </div>

        </div>

      </div>
    </div>
  );
}