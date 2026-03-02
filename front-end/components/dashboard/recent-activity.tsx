"use client";

import { Activity, FileHeart, Brain, Scan } from "lucide-react";

const activities = [
  {
    icon: Activity,
    iconColor: "text-primary",
    bgColor: "bg-primary/10",
    time: "14:20 - 15:00",
    title: "Analyse ECG",
    description: "Patient #1247 - Rythme sinusal",
  },
  {
    icon: FileHeart,
    iconColor: "text-amber-500",
    bgColor: "bg-amber-500/10",
    time: "11:30 - 12:15",
    title: "Classification",
    description: "Lot #38 - 12 enregistrements",
  },
  {
    icon: Brain,
    iconColor: "text-sky-500",
    bgColor: "bg-sky-500/10",
    time: "09:00 - 10:30",
    title: "Entrainement ML",
    description: "Random Forest - UCI Dataset",
  },
  {
    icon: Scan,
    iconColor: "text-emerald-500",
    bgColor: "bg-emerald-500/10",
    time: "08:00 - 08:45",
    title: "Revue donnees",
    description: "279 attributs - 452 patients",
  },
];

export function RecentActivity() {
  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <h3 className="text-base font-semibold text-foreground">
        Activite recente
      </h3>

      <div className="mt-4 flex items-center gap-3">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
          <Activity className="h-6 w-6 text-primary" />
        </div>
        <div>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-foreground">452</span>
            <span className="text-sm text-muted-foreground">/Patients</span>
          </div>
          <span className="text-xs text-muted-foreground">
            16 classes d{"'"}arythmie
          </span>
        </div>
      </div>

      <div className="mt-5 flex flex-col gap-3">
        {activities.map((activity) => (
          <div key={activity.title} className="flex items-start gap-3">
            <div
              className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${activity.bgColor}`}
            >
              <activity.icon className={`h-4 w-4 ${activity.iconColor}`} />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">
                  {activity.time}
                </span>
              </div>
              <p className="text-sm font-medium text-foreground">
                {activity.title}
              </p>
              <p className="truncate text-xs text-muted-foreground">
                {activity.description}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
