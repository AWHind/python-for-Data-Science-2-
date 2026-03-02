"use client";

import { useState, useRef, useEffect } from "react";
import {
  MessageCircle,
  X,
  Send,
  Bot,
  User,
  Heart,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface Message {
  id: string;
  role: "user" | "bot";
  content: string;
  timestamp: string;
}

const SUGGESTIONS = [
  "Qu'est-ce que CardioSense ?",
  "Comment fonctionne la prediction ?",
  "Quels modeles ML sont utilises ?",
  "Qu'est-ce qu'une arythmie cardiaque ?",
];

function getBotResponse(input: string): string {
  const lower = input.toLowerCase();

  if (lower.includes("cardiosense") || lower.includes("application") || lower.includes("c'est quoi")) {
    return "CardioSense est une application web intelligente dediee a l'analyse et la classification des arythmies cardiaques. Elle utilise des modeles de Machine Learning entraines sur le dataset UCI Arrhythmia (452 instances, 279 attributs, 16 classes) pour fournir une aide au diagnostic rapide et precise.";
  }

  if (lower.includes("prediction") || lower.includes("predire") || lower.includes("comment ca marche") || lower.includes("fonctionne")) {
    return "Pour obtenir une prediction, saisissez vos parametres ECG (age, frequence cardiaque, intervalles QRS, PR, QT, etc.) dans le formulaire dedie. Le systeme analyse ces donnees a l'aide d'algorithmes de Machine Learning et vous fournit la classe predite (normal ou type d'arythmie), le niveau de confiance et un rapport detaille.";
  }

  if (lower.includes("modele") || lower.includes("machine learning") || lower.includes("algorithme") || lower.includes("ml")) {
    return "CardioSense utilise plusieurs modeles de Machine Learning pour la classification : Random Forest (precision ~96.2%), SVM (Support Vector Machine), KNN (K-Nearest Neighbors), Decision Tree et Naive Bayes. Chaque modele est evalue avec des metriques comme l'accuracy, la precision, le rappel et la matrice de confusion.";
  }

  if (lower.includes("arythmie") || lower.includes("rythme") || lower.includes("cardiaque")) {
    return "Une arythmie cardiaque est un trouble du rythme du coeur. Le coeur peut battre trop vite (tachycardie), trop lentement (bradycardie) ou de maniere irreguliere. CardioSense permet de detecter et classifier 16 types differents d'arythmies a partir des donnees ECG, incluant le rythme sinusal normal, les arythmies ischemiques, les PVC, les blocs AV, et bien d'autres.";
  }

  if (lower.includes("dataset") || lower.includes("donnee") || lower.includes("uci")) {
    return "Le systeme est base sur le dataset UCI Arrhythmia, un jeu de donnees de reference en cardiologie. Il contient 452 instances (patients), 279 attributs (parametres ECG et cliniques) et 16 classes de classification allant du rythme normal aux differents types d'arythmies.";
  }

  if (lower.includes("precision") || lower.includes("accuracy") || lower.includes("performance") || lower.includes("fiable")) {
    return "Le modele principal (Random Forest) atteint une precision de 96.2% sur le dataset UCI Arrhythmia. Les performances sont evaluees avec une validation croisee a 10 folds, et des metriques detaillees incluant l'AUC-ROC (0.974), la precision, le rappel et le F1-score pour chaque classe.";
  }

  if (lower.includes("compte") || lower.includes("inscription") || lower.includes("inscrire") || lower.includes("creer")) {
    return "Pour creer un compte patient, cliquez sur 'Inscription' depuis la page d'accueil ou la page de connexion. Remplissez le formulaire avec vos informations (nom, email, telephone, age, sexe, mot de passe). Votre compte sera place en attente de validation par un administrateur pour garantir la securite du systeme.";
  }

  if (lower.includes("admin") || lower.includes("administrateur")) {
    return "L'interface administrateur offre un tableau de bord complet avec des indicateurs cles, la gestion des modeles ML, l'analyse statistique des donnees, la gestion des patients, la validation des inscriptions, le suivi des performances et les parametres du systeme.";
  }

  if (lower.includes("ecg") || lower.includes("electrocardiogramme")) {
    return "L'ECG (electrocardiogramme) est un examen qui enregistre l'activite electrique du coeur. Dans CardioSense, les parametres ECG cles incluent la duree QRS, les intervalles PR et QT, la frequence cardiaque et d'autres attributs utilises pour la classification des arythmies.";
  }

  if (lower.includes("securite") || lower.includes("confidentialite") || lower.includes("protection")) {
    return "CardioSense assure la securite des donnees medicales grace a un systeme d'authentification par roles (Patient/Administrateur), une validation des comptes par l'administrateur, et le respect des principes de confidentialite des donnees de sante.";
  }

  if (lower.includes("merci") || lower.includes("bravo") || lower.includes("super")) {
    return "Je vous en prie ! N'hesitez pas si vous avez d'autres questions sur CardioSense ou les arythmies cardiaques. Je suis la pour vous aider.";
  }

  if (lower.includes("bonjour") || lower.includes("salut") || lower.includes("hello") || lower.includes("bonsoir")) {
    return "Bonjour ! Je suis l'assistant virtuel de CardioSense. Je peux repondre a vos questions sur l'application, les arythmies cardiaques, les modeles de Machine Learning utilises et bien plus encore. Comment puis-je vous aider ?";
  }

  return "Merci pour votre question. Je peux vous renseigner sur CardioSense, les arythmies cardiaques, les modeles de Machine Learning utilises, le dataset UCI Arrhythmia, la creation de compte ou les fonctionnalites de l'application. N'hesitez pas a reformuler votre question ou a choisir parmi les suggestions proposees.";
}

export function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "bot",
      content:
        "Bonjour ! Je suis l'assistant virtuel de CardioSense. Je peux repondre a vos questions sur l'application, les arythmies cardiaques et l'utilisation du systeme. Comment puis-je vous aider ?",
      timestamp: new Date().toLocaleTimeString("fr-FR", {
        hour: "2-digit",
        minute: "2-digit",
      }),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const handleSend = (text?: string) => {
    const message = text || inputValue.trim();
    if (!message) return;

    const userMsg: Message = {
      id: `u-${Date.now()}`,
      role: "user",
      content: message,
      timestamp: new Date().toLocaleTimeString("fr-FR", {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue("");
    setIsTyping(true);

    setTimeout(() => {
      const response = getBotResponse(message);
      const botMsg: Message = {
        id: `b-${Date.now()}`,
        role: "bot",
        content: response,
        timestamp: new Date().toLocaleTimeString("fr-FR", {
          hour: "2-digit",
          minute: "2-digit",
        }),
      };
      setMessages((prev) => [...prev, botMsg]);
      setIsTyping(false);
    }, 800 + Math.random() * 700);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      {/* Floating Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "fixed bottom-6 right-6 z-50 flex h-14 w-14 items-center justify-center rounded-2xl shadow-lg transition-all duration-300 hover:scale-105",
          isOpen
            ? "bg-foreground text-background"
            : "bg-primary text-primary-foreground"
        )}
        aria-label={isOpen ? "Fermer le chatbot" : "Ouvrir le chatbot"}
      >
        {isOpen ? (
          <X className="h-6 w-6" />
        ) : (
          <MessageCircle className="h-6 w-6" />
        )}
      </button>

      {/* Notification dot when closed */}
      {!isOpen && (
        <span className="fixed bottom-[72px] right-6 z-50 flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500 text-[10px] font-bold text-primary-foreground shadow-md">
          1
        </span>
      )}

      {/* Chat Panel */}
      {isOpen && (
        <div className="fixed bottom-24 right-6 z-50 flex h-[520px] w-[380px] flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-2xl sm:w-[400px]">
          {/* Header */}
          <div className="flex items-center gap-3 border-b border-border bg-sidebar px-5 py-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary">
              <Heart className="h-5 w-5 text-primary-foreground" />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-sidebar-foreground">
                  CardioSense AI
                </h3>
                <Sparkles className="h-3.5 w-3.5 text-primary" />
              </div>
              <p className="text-xs text-sidebar-foreground/50">
                Assistant intelligent
              </p>
            </div>
            <span className="flex h-2.5 w-2.5 rounded-full bg-emerald-500" />
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-4">
            <div className="flex flex-col gap-4">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={cn(
                    "flex gap-2.5",
                    msg.role === "user" ? "flex-row-reverse" : "flex-row"
                  )}
                >
                  <div
                    className={cn(
                      "flex h-8 w-8 shrink-0 items-center justify-center rounded-xl",
                      msg.role === "user"
                        ? "bg-primary/10"
                        : "bg-sidebar"
                    )}
                  >
                    {msg.role === "user" ? (
                      <User className="h-4 w-4 text-primary" />
                    ) : (
                      <Bot className="h-4 w-4 text-sidebar-foreground" />
                    )}
                  </div>
                  <div
                    className={cn(
                      "max-w-[75%] rounded-2xl px-4 py-3",
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "border border-border bg-muted/50 text-foreground"
                    )}
                  >
                    <p className="text-sm leading-relaxed">{msg.content}</p>
                    <p
                      className={cn(
                        "mt-1.5 text-[10px]",
                        msg.role === "user"
                          ? "text-primary-foreground/60"
                          : "text-muted-foreground"
                      )}
                    >
                      {msg.timestamp}
                    </p>
                  </div>
                </div>
              ))}

              {isTyping && (
                <div className="flex gap-2.5">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-sidebar">
                    <Bot className="h-4 w-4 text-sidebar-foreground" />
                  </div>
                  <div className="rounded-2xl border border-border bg-muted/50 px-4 py-3">
                    <div className="flex items-center gap-1">
                      <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/40" style={{ animationDelay: "0ms" }} />
                      <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/40" style={{ animationDelay: "150ms" }} />
                      <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/40" style={{ animationDelay: "300ms" }} />
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Suggestions */}
          {messages.length <= 2 && (
            <div className="flex flex-wrap gap-2 border-t border-border px-4 py-3">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => handleSend(suggestion)}
                  className="rounded-full border border-border bg-background px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          )}

          {/* Input */}
          <div className="border-t border-border px-4 py-3">
            <div className="flex items-center gap-2 rounded-xl border border-border bg-background px-3 py-2">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Posez votre question..."
                className="flex-1 bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground/50"
              />
              <button
                onClick={() => handleSend()}
                disabled={!inputValue.trim() || isTyping}
                className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-40"
              >
                <Send className="h-4 w-4" />
                <span className="sr-only">Envoyer</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
