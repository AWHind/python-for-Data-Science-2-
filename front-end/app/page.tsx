"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth-context";
import { PublicHome } from "@/components/public-home";
import { LoginPage } from "@/components/login-page";
import { RegisterPage } from "@/components/register-page";
import { PatientDashboard } from "@/components/patient/patient-dashboard";
import { AdminDashboardLayout } from "@/components/admin/admin-dashboard-layout";

type PublicView = "home" | "login" | "register";

export default function Page() {
  const { user, isAuthenticated } = useAuth();
  const [publicView, setPublicView] = useState<PublicView>("home");

  // Authenticated users go directly to their dashboard
  if (isAuthenticated) {
    if (user?.role === "admin") {
      return <AdminDashboardLayout />;
    }
    return <PatientDashboard />;
  }

  // Public views
  if (publicView === "login") {
    return (
      <LoginPage
        onGoToRegister={() => setPublicView("register")}
        onGoToHome={() => setPublicView("home")}
      />
    );
  }

  if (publicView === "register") {
    return <RegisterPage onGoToLogin={() => setPublicView("login")} />;
  }

  // Default: Public Home
  return (
    <PublicHome
      onGoToLogin={() => setPublicView("login")}
      onGoToRegister={() => setPublicView("register")}
    />
  );
}
