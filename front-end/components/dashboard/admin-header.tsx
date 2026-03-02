"use client";

import { Search, Bell, Calendar, Shield } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

export function AdminHeader() {
  return (
    <header className="flex items-start justify-between">
      <div>
        <h1 className="font-serif text-3xl font-semibold tracking-tight text-foreground lg:text-4xl">
          Bonjour,{" "}
          <span className="text-balance">Dr. Martin</span>
        </h1>
        <p className="mt-1 flex items-center gap-1.5 text-sm text-muted-foreground">
          <Shield className="h-3.5 w-3.5 text-primary" />
          Interface Administrateur - Suivi des performances et analyses
        </p>
      </div>

      <div className="flex items-center gap-3">
        <button className="flex h-10 w-10 items-center justify-center rounded-xl border border-border bg-card text-muted-foreground transition-colors hover:text-foreground">
          <Search className="h-4 w-4" />
          <span className="sr-only">Rechercher</span>
        </button>
        <button className="flex h-10 w-10 items-center justify-center rounded-xl border border-border bg-card text-muted-foreground transition-colors hover:text-foreground">
          <Calendar className="h-4 w-4" />
          <span className="sr-only">Calendrier</span>
        </button>
        <button className="relative flex h-10 w-10 items-center justify-center rounded-xl border border-border bg-card text-muted-foreground transition-colors hover:text-foreground">
          <Bell className="h-4 w-4" />
          <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-primary" />
          <span className="sr-only">Notifications</span>
        </button>
        <div className="flex items-center gap-2 rounded-full border border-border bg-card py-1.5 pl-1.5 pr-4">
          <Avatar className="h-7 w-7">
            <AvatarFallback className="bg-foreground text-xs text-background">
              DM
            </AvatarFallback>
          </Avatar>
          <span className="text-sm font-medium text-foreground">
            Dr. Martin
          </span>
        </div>
      </div>
    </header>
  );
}
