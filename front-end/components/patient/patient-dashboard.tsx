"use client";

import { useState } from "react";
import { PatientSidebar } from "./patient-sidebar";
import { PatientHome } from "./patient-home";
import { PatientPrediction } from "./patient-prediction";
import { PatientHistory } from "./patient-history";
import Chatbot from "@/components/chatbot";
import { useAuth } from "@/lib/auth-context";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Search,
  Bell,
  Calendar,
  Menu,
  X,
  Home,
  Activity,
  FileBarChart,
  LogOut,
  Heart,
} from "lucide-react";
import { cn } from "@/lib/utils";

export function PatientDashboard() {
  const { user, logout } = useAuth();
  const [activeNav, setActiveNav] = useState(0);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleNavigate = (page: number) => {
    setActiveNav(page);
    setMobileMenuOpen(false);
  };

  const mobileNavItems = [
    { icon: Home, label: "Accueil" },
    { icon: Activity, label: "Prediction" },
    { icon: FileBarChart, label: "Historique" },
  ];

  return (
    <div className="flex h-screen gap-4 p-4">
      {/* Desktop Sidebar */}
      <div className="hidden lg:block">
        <PatientSidebar
          activeNav={activeNav}
          onNavChange={setActiveNav}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        />
      </div>

      {/* Mobile Sidebar Overlay */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="absolute inset-0 bg-foreground/40"
            onClick={() => setMobileMenuOpen(false)}
          />
          <div className="relative z-10 flex h-full w-[260px] flex-col bg-sidebar p-4">
            <div className="flex items-center justify-between pb-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary">
                  <Heart className="h-5 w-5 text-primary-foreground" />
                </div>
                <span className="text-sm font-bold text-sidebar-foreground">CardioSense</span>
              </div>
              <button
                onClick={() => setMobileMenuOpen(false)}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-sidebar-foreground/60 hover:bg-sidebar-accent"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {user && (
              <div className="mb-4 rounded-xl bg-sidebar-accent p-3">
                <div className="flex items-center gap-3">
                  <Avatar className="h-8 w-8">
                    <AvatarFallback className="bg-primary text-xs text-primary-foreground">
                      {user.initials}
                    </AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="text-sm font-medium text-sidebar-foreground">{user.name}</p>
                    <p className="text-xs text-sidebar-foreground/50">Patient</p>
                  </div>
                </div>
              </div>
            )}

            <nav className="flex flex-1 flex-col gap-1">
              {mobileNavItems.map((item, index) => (
                <button
                  key={item.label}
                  onClick={() => handleNavigate(index)}
                  className={cn(
                    "flex items-center gap-3 rounded-xl px-3 py-2.5 transition-all",
                    index === activeNav
                      ? "bg-sidebar-accent text-sidebar-primary"
                      : "text-sidebar-foreground/60 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
                  )}
                >
                  <item.icon className="h-5 w-5" />
                  <span className="text-sm font-medium">{item.label}</span>
                </button>
              ))}
            </nav>

            <button
              onClick={logout}
              className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sidebar-foreground/60 hover:bg-sidebar-accent hover:text-sidebar-foreground"
            >
              <LogOut className="h-5 w-5" />
              <span className="text-sm font-medium">Deconnexion</span>
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex flex-1 flex-col gap-4 overflow-y-auto">
        {/* Header */}
        <header className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setMobileMenuOpen(true)}
              className="flex h-10 w-10 items-center justify-center rounded-xl border border-border bg-card text-muted-foreground lg:hidden"
            >
              <Menu className="h-5 w-5" />
              <span className="sr-only">Menu</span>
            </button>
            <div>
              <h1 className="font-serif text-2xl font-semibold tracking-tight text-foreground sm:text-3xl lg:text-4xl">
                Bonjour,{" "}
                <span className="text-balance">{user?.name || "Patient"}</span>
              </h1>
              <p className="mt-1 text-sm text-muted-foreground">
                Analyse et classification des arythmies cardiaques
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button className="hidden h-10 w-10 items-center justify-center rounded-xl border border-border bg-card text-muted-foreground transition-colors hover:text-foreground sm:flex">
              <Search className="h-4 w-4" />
              <span className="sr-only">Rechercher</span>
            </button>
            <button className="hidden h-10 w-10 items-center justify-center rounded-xl border border-border bg-card text-muted-foreground transition-colors hover:text-foreground sm:flex">
              <Calendar className="h-4 w-4" />
              <span className="sr-only">Calendrier</span>
            </button>
            <button className="relative flex h-10 w-10 items-center justify-center rounded-xl border border-border bg-card text-muted-foreground transition-colors hover:text-foreground">
              <Bell className="h-4 w-4" />
              <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-primary" />
              <span className="sr-only">Notifications</span>
            </button>
            <div className="hidden items-center gap-2 rounded-full border border-border bg-card py-1.5 pl-1.5 pr-4 sm:flex">
              <Avatar className="h-7 w-7">
                <AvatarFallback className="bg-primary text-xs text-primary-foreground">
                  {user?.initials || "PT"}
                </AvatarFallback>
              </Avatar>
              <span className="text-sm font-medium text-foreground">
                {user?.name || "Patient"}
              </span>
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1">
          {activeNav === 0 && <PatientHome onNavigate={handleNavigate} />}
          {activeNav === 1 && <PatientPrediction />}
          {activeNav === 2 && <PatientHistory />}
        </main>
      </div>

      {/* Chatbot */}
      <Chatbot />
    </div>
  );
}
