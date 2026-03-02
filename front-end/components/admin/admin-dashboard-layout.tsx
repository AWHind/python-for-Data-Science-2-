"use client";

import { useState } from "react";
import { AdminSidebar } from "./admin-sidebar";
import { AdminDashboard } from "./admin-dashboard";
import { AdminModels } from "./admin-models";
import { AdminStatistics } from "./admin-statistics";
import { AdminPatients } from "./admin-patients";
import { AdminValidations } from "./admin-validations";
import { AdminPerformance } from "./admin-performance";
import { AdminSettings } from "./admin-settings";
import { Chatbot } from "@/components/chatbot";
import { useAuth } from "@/lib/auth-context";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Search,
  Bell,
  Calendar,
  Shield,
  Menu,
  X,
  Home,
  Brain,
  FileBarChart,
  Users,
  UserCheck,
  Activity,
  Settings,
  LogOut,
  Heart,
} from "lucide-react";
import { cn } from "@/lib/utils";
import PredictionForm from "@/components/dashboard/prediction-form";// ✅ مهم

export function AdminDashboardLayout() {
  const { user, logout } = useAuth();
  const [activeNav, setActiveNav] = useState(0);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleNavigate = (page: number) => {
    setActiveNav(page);
    setMobileMenuOpen(false);
  };

  const mobileNavItems = [
    { icon: Home, label: "Tableau de Bord" },
    { icon: Brain, label: "Modeles ML" },
    { icon: FileBarChart, label: "Analyse Statistique" },
    { icon: Users, label: "Gestion Patients" },
    { icon: UserCheck, label: "Validations" },
    { icon: Activity, label: "Performances" },
    { icon: Settings, label: "Parametres" },
  ];

  const pageTitles = [
    "Tableau de Bord",
    "Modeles ML",
    "Analyse Statistique",
    "Gestion Patients",
    "Validations",
    "Performances",
    "Parametres",
  ];

  return (
      <div className="flex h-screen gap-4 p-4">
        {/* Desktop Sidebar */}
        <div className="hidden lg:block">
          <AdminSidebar
              activeNav={activeNav}
              onNavChange={setActiveNav}
              collapsed={sidebarCollapsed}
              onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
          />
        </div>

        {/* Main Content */}
        <div className="flex flex-1 flex-col gap-4 overflow-y-auto">
          {/* Header */}
          <header className="flex items-start justify-between">
            <div>
              <h1 className="text-2xl font-semibold">
                Bonjour, {user?.name || "Dr. Martin"}
              </h1>
              <p className="text-sm text-muted-foreground">
                Interface Administrateur - {pageTitles[activeNav]}
              </p>
            </div>
          </header>

          {/* Content */}
          <main className="flex-1">
            {activeNav === 0 && (
                <>
                  <AdminDashboard />

                  {/* ✅ Prediction Form مضاف هنا */}
                  <div className="mt-8">
                    <PredictionForm />
                  </div>
                </>
            )}

            {activeNav === 1 && <AdminModels />}
            {activeNav === 2 && <AdminStatistics />}
            {activeNav === 3 && <AdminPatients />}
            {activeNav === 4 && <AdminValidations />}
            {activeNav === 5 && <AdminPerformance />}
            {activeNav === 6 && <AdminSettings />}
          </main>
        </div>

        <Chatbot />
      </div>
  );
}