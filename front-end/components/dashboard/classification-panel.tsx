"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";

const arrhythmiaClasses = [
  { id: 1, name: "Normal", count: 245, percentage: 54.2, color: "bg-emerald-500" },
  { id: 2, name: "Arythmie ischem.", count: 44, percentage: 9.7, color: "bg-primary" },
  { id: 3, name: "PVC", count: 25, percentage: 5.5, color: "bg-amber-500" },
  { id: 4, name: "Bloc AV", count: 19, percentage: 4.2, color: "bg-sky-500" },
  { id: 5, name: "BBG", count: 22, percentage: 4.9, color: "bg-violet-500" },
  { id: 6, name: "BBD", count: 18, percentage: 4.0, color: "bg-orange-500" },
  { id: 7, name: "Autres", count: 79, percentage: 17.5, color: "bg-muted-foreground/50" },
];

export function ClassificationPanel() {
  const [hoveredId, setHoveredId] = useState<number | null>(null);

  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-foreground">
          Classification des Arythmies
        </h3>
        <span className="text-xs text-muted-foreground">UCI Dataset</span>
      </div>

      <div className="mt-4 flex h-3 w-full overflow-hidden rounded-full">
        {arrhythmiaClasses.map((cls) => (
          <div
            key={cls.id}
            className={cn(
              "transition-opacity",
              cls.color,
              hoveredId !== null && hoveredId !== cls.id
                ? "opacity-40"
                : "opacity-100"
            )}
            style={{ width: `${cls.percentage}%` }}
            onMouseEnter={() => setHoveredId(cls.id)}
            onMouseLeave={() => setHoveredId(null)}
          />
        ))}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2">
        {arrhythmiaClasses.map((cls) => (
          <div
            key={cls.id}
            className={cn(
              "flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors",
              hoveredId === cls.id ? "bg-muted" : ""
            )}
            onMouseEnter={() => setHoveredId(cls.id)}
            onMouseLeave={() => setHoveredId(null)}
          >
            <div className={cn("h-2.5 w-2.5 rounded-full", cls.color)} />
            <span className="text-xs text-foreground">{cls.name}</span>
            <span className="ml-auto text-xs text-muted-foreground">
              {cls.count}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
