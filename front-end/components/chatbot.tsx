"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  MessageCircle,
  X,
  Send,
  Bot,
  User,
  Heart,
  Sparkles,
  Trash2,
} from "lucide-react";

type Message = {
  id: string;
  role: "user" | "bot";
  content: string;
  timestamp: string;
};

const STORAGE_KEY = "cardiosense_chat_history_v1";

const DEFAULT_SUGGESTIONS = [
  "What is arrhythmia?",
  "What are the symptoms of arrhythmia?",
  "How is arrhythmia detected?",
  "What treatments exist for arrhythmia?",
  "Can ECG detect arrhythmia?",
  "What is CardioSense?",
];

function nowFR() {
  return new Date().toLocaleTimeString("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function Chatbot() {

  const [isOpen, setIsOpen] = useState(false);

  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "bot",
      content:
        "Hello 👋 I am the CardioSense AI assistant. Ask me about arrhythmia or heart health.",
      timestamp: nowFR(),
    },
  ]);

  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [suggestions] = useState<string[]>(DEFAULT_SUGGESTIONS);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;

    const parsed = JSON.parse(raw);

    if (Array.isArray(parsed)) {
      setMessages(parsed);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const addMessage = (msg: Message) => {
    setMessages((prev) => [...prev, msg]);
  };

  const clearChat = () => {
    setMessages([
      {
        id: "welcome",
        role: "bot",
        content: "Hello! How can I help you?",
        timestamp: nowFR(),
      },
    ]);
    setInputValue("");
  };

  const sendToBackend = async (message: string) => {

    try {

      const controller = new AbortController();

      const timeout = setTimeout(() => {
        controller.abort();
      }, 8000);

      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!res.ok) {
        throw new Error("API error");
      }

      const data = await res.json();

      return data.response ?? data.reply ?? "No response from AI";

    } catch (error) {

      console.error("Chatbot error:", error);

      return "⚠️ The AI service is temporarily unavailable. Please try again.";

    }

  };

  const handleSend = async (text?: string) => {

    const message = (text ?? inputValue).trim();

    if (!message || isTyping) return;

    const userMsg: Message = {
      id: `u-${Date.now()}`,
      role: "user",
      content: message,
      timestamp: nowFR(),
    };

    addMessage(userMsg);
    setInputValue("");
    setIsTyping(true);

    try {

      const reply = await sendToBackend(message);

      const botMsg: Message = {
        id: `b-${Date.now()}`,
        role: "bot",
        content: reply,
        timestamp: nowFR(),
      };

      addMessage(botMsg);

    } catch {

      addMessage({
        id: `b-${Date.now()}`,
        role: "bot",
        content: "Server error. Check backend.",
        timestamp: nowFR(),
      });

    } finally {

      setIsTyping(false);

    }

  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      {/* Floating Button */}

      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed bottom-6 right-6 h-14 w-14 rounded-xl bg-red-600 text-white flex items-center justify-center shadow-lg hover:scale-105 transition"
      >
        {isOpen ? <X /> : <MessageCircle />}
      </button>

      {isOpen && (

        <div className="fixed bottom-24 right-6 w-[380px] h-[520px] bg-white rounded-xl shadow-xl flex flex-col border">

          {/* Header */}

          <div className="flex justify-between items-center p-4 border-b bg-red-600 text-white rounded-t-xl">

            <div className="flex gap-2 items-center">
              <Heart size={18} />
              <span className="font-semibold">CardioSense AI</span>
            </div>

            <button onClick={clearChat}>
              <Trash2 size={18} />
            </button>

          </div>

          {/* Messages */}

          <div className="flex-1 overflow-y-auto p-4 space-y-4">

            {messages.map((msg) => (

              <div
                key={msg.id}
                className={`flex ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >

                <div
                  className={`p-3 rounded-xl max-w-[75%] flex gap-2 items-start ${
                    msg.role === "user"
                      ? "bg-red-500 text-white"
                      : "bg-gray-100"
                  }`}
                >

                  {msg.role === "bot" && <Bot size={16} />}
                  {msg.role === "user" && <User size={16} />}

                  <p className="text-sm leading-relaxed whitespace-pre-line">
                    {msg.content}
                  </p>

                </div>

              </div>

            ))}

            {isTyping && (
              <div className="text-sm text-gray-400 flex items-center gap-2">
                <Bot size={14}/>
                CardioSense AI is typing...
              </div>
            )}

            <div ref={messagesEndRef} />

          </div>

          {/* Suggestions */}

          <div className="p-3 border-t flex flex-wrap gap-2">

            {suggestions.map((q) => (

              <button
                key={q}
                onClick={() => handleSend(q)}
                className="text-xs bg-gray-100 px-3 py-1 rounded-full flex items-center gap-1 hover:bg-gray-200 transition"
              >
                <Sparkles size={12} />
                {q}
              </button>

            ))}

          </div>

          {/* Input */}

          <div className="flex gap-2 p-3 border-t">

            <input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about arrhythmia..."
              className="flex-1 border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-red-500"
            />

            <button
              onClick={() => handleSend()}
              className="bg-red-600 text-white p-2 rounded hover:bg-red-700 transition"
            >
              <Send size={16} />
            </button>

          </div>

        </div>

      )}
    </>
  );
}