"use client";

import {
  Heart,
  Home,
  Brain,
  FileBarChart,
  Users,
  Activity,
  Settings,
  LogOut,
  ChevronLeft,
  ChevronRight,
  UserCheck,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/lib/auth-context";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

interface AdminSidebarProps {
  activeNav: number;
  onNavChange: (index: number) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

const navItems = [
  { icon: Home, label: "Tableau de Bord" },
  { icon: Brain, label: "Modeles ML" },
  { icon: FileBarChart, label: "Analyse Statistique" },
  { icon: Users, label: "Gestion Patients" },
  { icon: UserCheck, label: "Validations" },
  { icon: Activity, label: "Performances" },
  { icon: Settings, label: "Parametres" },
];

export function AdminSidebar({
  activeNav,
  onNavChange,
  collapsed,
  onToggleCollapse,
}: AdminSidebarProps) {
  const { user, logout } = useAuth();

  return (
    <aside
      className={cn(
        "relative flex h-full flex-col rounded-2xl bg-sidebar transition-all duration-300",
        collapsed ? "w-[72px]" : "w-[230px]"
      )}
    >
      {/* Logo */}
      <div
        className={cn(
          "flex items-center gap-3 px-4 pt-6 pb-4",
          collapsed ? "justify-center" : ""
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

      {/* User Info */}
      {!collapsed && user && (
        <div className="mx-3 mb-4 rounded-xl bg-sidebar-accent p-3">
          <div className="flex items-center gap-3">
            <Avatar className="h-8 w-8">
              <AvatarFallback className="bg-primary text-xs text-primary-foreground">
                {user.initials}
              </AvatarFallback>
            </Avatar>
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-sidebar-foreground">
                {user.name}
              </p>
              <p className="truncate text-xs text-sidebar-foreground/50">
                Administrateur
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Divider */}
      <div className="mx-4 mb-2 h-px bg-sidebar-accent" />

      {/* Nav */}
      <nav
        className={cn(
          "flex flex-1 flex-col gap-1 px-3 py-2",
          collapsed ? "items-center" : ""
        )}
      >
        {navItems.map((item, index) => (
          <button
            key={item.label}
            onClick={() => onNavChange(index)}
            className={cn(
              "flex items-center gap-3 rounded-xl transition-all duration-200",
              collapsed ? "h-11 w-11 justify-center" : "w-full px-3 py-2.5",
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

      {/* Bottom */}
      <div className="flex flex-col gap-2 px-3 pb-4">
        <div className="h-px bg-sidebar-accent" />
        <button
          onClick={logout}
          className={cn(
            "flex items-center gap-3 rounded-xl text-sidebar-foreground/60 transition-colors hover:bg-sidebar-accent hover:text-sidebar-foreground",
            collapsed ? "h-11 w-11 justify-center" : "w-full px-3 py-2.5"
          )}
          title="Deconnexion"
        >
          <LogOut className="h-5 w-5 shrink-0" />
          {!collapsed && (
            <span className="text-sm font-medium">Deconnexion</span>
          )}
          <span className="sr-only">Deconnexion</span>
        </button>

        <button
          onClick={onToggleCollapse}
          className="flex h-8 w-full items-center justify-center rounded-lg text-sidebar-foreground/40 transition-colors hover:bg-sidebar-accent hover:text-sidebar-foreground"
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </button>
      </div>
    </aside>
  );
}
