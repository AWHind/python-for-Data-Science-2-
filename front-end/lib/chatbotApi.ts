export type ChatMessage = {
  role: "user" | "bot";
  content: string;
};

export type ChatResponse = {
  reply: string;
  suggestions?: string[];
};

const DEFAULT_TIMEOUT_MS = 12_000;

function getApiBaseUrl() {
  return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
}

export async function chatRequest(
  message: string,
  history: ChatMessage[] = []
): Promise<ChatResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);

  try {
    const res = await fetch(`${getApiBaseUrl()}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({ message, history }),
    });

    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`Chat API failed: ${res.status} ${txt}`);
    }

    return (await res.json()) as ChatResponse;
  } finally {
    clearTimeout(timeout);
  }
}