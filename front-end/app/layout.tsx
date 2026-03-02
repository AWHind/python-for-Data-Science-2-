import React from "react"
import type { Metadata, Viewport } from "next";
import { Inter, Playfair_Display } from "next/font/google";
import { AuthProvider } from "@/lib/auth-context";

import "./globals.css";

const _inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const _playfair = Playfair_Display({
  subsets: ["latin"],
  variable: "--font-playfair",
});

export const metadata: Metadata = {
  title: "CardioSense - Analyse des Arythmies Cardiaques",
  description:
    "Application web intelligente pour l'analyse et la classification des arythmies cardiaques par intelligence artificielle - UCI Arrhythmia Dataset",
};

export const viewport: Viewport = {
  themeColor: "#d6336c",
  userScalable: true,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr">
      <body
        className={`${_inter.variable} ${_playfair.variable} font-sans antialiased`}
      >
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
