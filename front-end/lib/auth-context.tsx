"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from "react";

export type UserRole = "patient" | "admin";
export type AccountStatus = "active" | "pending" | "rejected";

export interface User {
  id: string;
  name: string;
  role: UserRole;
  email: string;
  initials: string;
  status: AccountStatus;
}

export interface PendingRegistration {
  id: string;
  name: string;
  email: string;
  phone: string;
  age: string;
  sex: string;
  registeredAt: string;
  status: AccountStatus;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<{ success: boolean; pending?: boolean }>;
  logout: () => void;
  register: (data: {
    name: string;
    email: string;
    password: string;
    phone: string;
    age: string;
    sex: string;
  }) => Promise<{ success: boolean; message: string }>;
  pendingRegistrations: PendingRegistration[];
  approveRegistration: (id: string) => void;
  rejectRegistration: (id: string) => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

interface StoredUser {
  email: string;
  password: string;
  user: User;
}

const INITIAL_USERS: StoredUser[] = [
  {
    email: "patient@cardiosense.com",
    password: "patient123",
    user: {
      id: "u-001",
      name: "Ahmed Benali",
      role: "patient",
      email: "patient@cardiosense.com",
      initials: "AB",
      status: "active",
    },
  },
  {
    email: "admin@cardiosense.com",
    password: "admin123",
    user: {
      id: "u-002",
      name: "Dr. Martin",
      role: "admin",
      email: "admin@cardiosense.com",
      initials: "DM",
      status: "active",
    },
  },
];

const INITIAL_PENDING: PendingRegistration[] = [
  {
    id: "reg-001",
    name: "Samira Khaled",
    email: "samira.k@email.com",
    phone: "+213 555 123 456",
    age: "45",
    sex: "F",
    registeredAt: "08/02/2026 10:30",
    status: "pending",
  },
  {
    id: "reg-002",
    name: "Mohamed Amir",
    email: "m.amir@email.com",
    phone: "+213 555 789 012",
    age: "62",
    sex: "M",
    registeredAt: "09/02/2026 14:15",
    status: "pending",
  },
];

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [users, setUsers] = useState<StoredUser[]>(INITIAL_USERS);
  const [pendingRegistrations, setPendingRegistrations] =
    useState<PendingRegistration[]>(INITIAL_PENDING);

  const login = useCallback(
    async (
      email: string,
      password: string
    ): Promise<{ success: boolean; pending?: boolean }> => {
      await new Promise((r) => setTimeout(r, 800));
      const found = users.find(
        (u) => u.email === email && u.password === password
      );
      if (found) {
        if (found.user.status === "pending") {
          return { success: false, pending: true };
        }
        if (found.user.status === "rejected") {
          return { success: false };
        }
        setUser(found.user);
        return { success: true };
      }
      return { success: false };
    },
    [users]
  );

  const logout = useCallback(() => {
    setUser(null);
  }, []);

  const register = useCallback(
    async (data: {
      name: string;
      email: string;
      password: string;
      phone: string;
      age: string;
      sex: string;
    }): Promise<{ success: boolean; message: string }> => {
      await new Promise((r) => setTimeout(r, 1000));

      const exists = users.find((u) => u.email === data.email);
      if (exists) {
        return {
          success: false,
          message: "Un compte avec cet email existe deja.",
        };
      }

      const pendingExists = pendingRegistrations.find(
        (p) => p.email === data.email
      );
      if (pendingExists) {
        return {
          success: false,
          message: "Une demande d'inscription avec cet email est deja en attente.",
        };
      }

      const initials = data.name
        .split(" ")
        .map((n) => n[0])
        .join("")
        .toUpperCase()
        .slice(0, 2);

      const newId = `reg-${Date.now().toString(36)}`;

      const newRegistration: PendingRegistration = {
        id: newId,
        name: data.name,
        email: data.email,
        phone: data.phone,
        age: data.age,
        sex: data.sex,
        registeredAt: new Date().toLocaleString("fr-FR", {
          day: "2-digit",
          month: "2-digit",
          year: "numeric",
          hour: "2-digit",
          minute: "2-digit",
        }),
        status: "pending",
      };

      setPendingRegistrations((prev) => [newRegistration, ...prev]);

      const newUser: StoredUser = {
        email: data.email,
        password: data.password,
        user: {
          id: `u-${Date.now().toString(36)}`,
          name: data.name,
          role: "patient",
          email: data.email,
          initials,
          status: "pending",
        },
      };

      setUsers((prev) => [...prev, newUser]);

      return {
        success: true,
        message:
          "Votre demande d'inscription a ete envoyee. Vous recevrez une notification une fois votre compte valide par l'administrateur.",
      };
    },
    [users, pendingRegistrations]
  );

  const approveRegistration = useCallback((id: string) => {
    setPendingRegistrations((prev) =>
      prev.map((r) => (r.id === id ? { ...r, status: "active" as AccountStatus } : r))
    );
    // Find the registration to get the email
    setPendingRegistrations((prev) => {
      const reg = prev.find((r) => r.id === id);
      if (reg) {
        setUsers((prevUsers) =>
          prevUsers.map((u) =>
            u.email === reg.email
              ? { ...u, user: { ...u.user, status: "active" as AccountStatus } }
              : u
          )
        );
      }
      return prev;
    });
  }, []);

  const rejectRegistration = useCallback((id: string) => {
    setPendingRegistrations((prev) =>
      prev.map((r) => (r.id === id ? { ...r, status: "rejected" as AccountStatus } : r))
    );
    setPendingRegistrations((prev) => {
      const reg = prev.find((r) => r.id === id);
      if (reg) {
        setUsers((prevUsers) =>
          prevUsers.map((u) =>
            u.email === reg.email
              ? { ...u, user: { ...u.user, status: "rejected" as AccountStatus } }
              : u
          )
        );
      }
      return prev;
    });
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        login,
        logout,
        register,
        pendingRegistrations,
        approveRegistration,
        rejectRegistration,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
