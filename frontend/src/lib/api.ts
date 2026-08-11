// Streaming client for the HealthHarness backend's SSE /chat endpoint.
// Parses the raw "event: <type>\ndata: <json>\n\n" SSE wire format by hand
// (fetch + ReadableStream) since native EventSource doesn't support POST bodies.

export type RunEvent = {
  type:
    | "run_started"
    | "content_delta"
    | "tool_call_started"
    | "tool_call_result"
    | "run_completed"
    | "error";
  content?: string;
  tool_name?: string;
  tool_arguments?: Record<string, unknown>;
  tool_result?: string;
  error?: string;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function parseSSEBlock(block: string): RunEvent | null {
  let eventType = "message";
  const dataLines: string[] = [];

  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      eventType = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
    // lines starting with ":" are SSE comments/pings - ignored
  }

  if (dataLines.length === 0) return null;

  try {
    const parsed = JSON.parse(dataLines.join("\n"));
    return { type: eventType as RunEvent["type"], ...parsed };
  } catch {
    return null;
  }
}

export async function* streamChat(
  sessionId: string,
  message: string,
  signal?: AbortSignal
): AsyncGenerator<RunEvent> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
    signal,
  });

  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    let detail = text;
    try {
      detail = JSON.parse(text).detail ?? text;
    } catch {
      // not JSON, use raw text
    }
    throw new Error(`Chat request failed (${res.status}): ${detail}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // sse_starlette (the backend's SSE library) uses "\r\n" as its line
    // separator, so each event block actually ends with "\r\n\r\n" - not
    // "\n\n". Normalize to "\n" so the splitting below is separator-agnostic.
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    let sepIndex: number;
    while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
      const rawBlock = buffer.slice(0, sepIndex);
      buffer = buffer.slice(sepIndex + 2);
      const event = parseSSEBlock(rawBlock);
      if (event) yield event;
    }
  }
}

export async function resetSession(sessionId: string): Promise<void> {
  await fetch(`${API_URL}/sessions/${sessionId}/reset`, { method: "POST" });
}
