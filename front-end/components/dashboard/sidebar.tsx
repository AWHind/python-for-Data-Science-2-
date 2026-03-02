"use client";

import {
  Heart,
  Home,
  Activity,
  FileBarChart,
  Brain,
  Users,
  Settings,
  ChevronLeft,
  ChevronRight,
  Shield,
  UserCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

export type ViewMode = "client" | "admin";

interface SidebarProps {
  activeView: ViewMode;
  onViewChange: (view: ViewMode) => void;
  activeNav: number;
  onNavChange: (index: number) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

const clientNavItems = [
  { icon: Home, label: "Accueil" },
  { icon: Activity, label: "Nouvelle Prediction" },
  { icon: FileBarChart, label: "Mes Resultats" },
  { icon: Heart, label: "Mon Profil Cardiaque" },
];

const adminNavItems = [
  { icon: Home, label: "Tableau de Bord" },
  { icon: Brain, label: "Modeles ML" },
  { icon: FileBarChart, label: "Analyse Statistique" },
  { icon: Users, label: "Gestion Patients" },
  { icon: Activity, label: "Performances" },
  { icon: Settings, label: "Parametres" },
];

export function DashboardSidebar({
  activeView,
  onViewChange,
  activeNav,
  onNavChange,
  collapsed,
  onToggleCollapse,
}: SidebarProps) {
  const navItems = activeView === "client" ? clientNavItems : adminNavItems;

  return (
    <aside
      className={cn(
        "relative flex h-full flex-col items-center gap-2 rounded-2xl bg-sidebar py-6 transition-all duration-300",
        collapsed ? "w-[72px]" : "w-[220px]"
      )}
    >
      {/* Logo */}
      <div
        className={cn(
          "mb-4 flex items-center gap-3 px-4",
          collapsed ? "justify-center" : "w-full"
        )}
      >
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary">
          <Heart className="h-5 w-5 text-primary-foreground" />
        </div>
        {!collapsed && (
          <span className="text-sm font-bold tracking-wide text-sidebar-foreground">
            CardioSense
          </span>
        )}
      </div>

      {/* View Switcher */}
      <div
        className={cn(
          "mb-4 flex gap-1 px-3",
          collapsed ? "flex-col" : "w-full"
        )}
      >
        <button
          onClick={() => onViewChange("client")}
          className={cn(
            "flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-medium transition-all",
            collapsed ? "h-10 w-10" : "flex-1",
            activeView === "client"
              ? "bg-primary text-primary-foreground"
              : "text-sidebar-foreground/60 hover:bg-sidebar-accent hover:text-sidebar-foreground"
          )}
          title="Interface Client"
        >
          <UserCircle className="h-4 w-4 shrink-0" />
          {!collapsed && <span>Client</span>}
        </button>
        <button
          onClick={() => onViewChange("admin")}
          className={cn(
            "flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-medium transition-all",
            collapsed ? "h-10 w-10" : "flex-1",
            activeView === "admin"
              ? "bg-primary text-primary-foreground"
              : "text-sidebar-foreground/60 hover:bg-sidebar-accent hover:text-sidebar-foreground"
          )}
          title="Interface Admin"
        >
          <Shield className="h-4 w-4 shrink-0" />
          {!collapsed && <span>Admin</span>}
        </button>
      </div>

      {/* Divider */}
      <div className="mx-4 mb-2 h-px w-8 bg-sidebar-accent" />

      {/* Nav Items */}
      <nav
        className={cn(
          "flex flex-1 flex-col gap-1",
          collapsed ? "items-center" : "w-full px-3"
        )}
      >
        {navItems.map((item, index) => (
          <button
            key={item.label}
            onClick={() => onNavChange(index)}
            className={cn(
              "flex items-center gap-3 rounded-xl transition-all duration-200",
              collapsed
                ? "h-11 w-11 justify-center"
                : "w-full px-3 py-2.5",
              index === activeNav
                ? "bg-sidebar-accent text-sidebar-primary"
                : "text-sidebar-foreground/60 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
            )}
            title={item.label}
          >
            <item.icon className="h-5 w-5 shrink-0" />
            {!collapsed && (
              <span className="text-sm font-medium">{item.label}</span>
            )}
            <span className="sr-only">{item.label}</span>
          </button>
        ))}
      </nav>

      {/* Collapse Toggle */}
      <button
        onClick={onToggleCollapse}
        className="mt-2 flex h-8 w-8 items-center justify-center rounded-lg text-sidebar-foreground/40 transition-colors hover:bg-sidebar-accent hover:text-sidebar-foreground"
      >
        {collapsed ? (
          <ChevronRight className="h-4 w-4" />
        ) : (
          <ChevronLeft className="h-4 w-4" />
        )}
      </button>
    </aside>
  );
}
