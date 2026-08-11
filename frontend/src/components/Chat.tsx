"use client";

import { useEffect, useRef, useState } from "react";
import { streamChat, resetSession } from "@/lib/api";
import type { ChatMessage } from "@/lib/types";
import Message from "./Message";

function newId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);
}

export default function Chat() {
  const [sessionId] = useState(() => newId());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend() {
    const text = input.trim();
    if (!text || isStreaming) return;

    const userMessage: ChatMessage = {
      id: newId(),
      role: "user",
      content: text,
      toolCalls: [],
    };
    const assistantId = newId();
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      toolCalls: [],
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput("");
    setIsStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    const updateAssistant = (updater: (m: ChatMessage) => ChatMessage) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? updater(m) : m))
      );
    };

    try {
      for await (const event of streamChat(sessionId, text, controller.signal)) {
        switch (event.type) {
          case "content_delta":
            updateAssistant((m) => ({
              ...m,
              content: m.content + (event.content ?? ""),
            }));
            break;
          case "tool_call_started":
            updateAssistant((m) => ({
              ...m,
              toolCalls: [
                ...m.toolCalls,
                {
                  name: event.tool_name ?? "unknown_tool",
                  arguments: event.tool_arguments ?? {},
                  status: "running",
                },
              ],
            }));
            break;
          case "tool_call_result":
            updateAssistant((m) => {
              const toolCalls = [...m.toolCalls];
              const idx = toolCalls.map((t) => t.status).lastIndexOf("running");
              if (idx !== -1) {
                toolCalls[idx] = {
                  ...toolCalls[idx],
                  result: event.tool_result,
                  status: "done",
                };
              }
              return { ...m, toolCalls };
            });
            break;
          case "error":
            updateAssistant((m) => ({
              ...m,
              error: event.error ?? "Something went wrong.",
              isStreaming: false,
            }));
            break;
          case "run_completed":
            updateAssistant((m) => ({ ...m, isStreaming: false }));
            break;
          default:
            break;
        }
      }
    } catch (err) {
      updateAssistant((m) => ({
        ...m,
        isStreaming: false,
        error: err instanceof Error ? err.message : "Connection error.",
      }));
    } finally {
      updateAssistant((m) => ({ ...m, isStreaming: false }));
      setIsStreaming(false);
      abortRef.current = null;
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  async function handleReset() {
    abortRef.current?.abort();
    await resetSession(sessionId);
    setMessages([]);
    setIsStreaming(false);
  }

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto">
      <header className="flex items-center justify-between px-4 py-3 border-b border-neutral-200">
        <h1 className="text-sm font-semibold text-neutral-800">
          HealthHarness — Agent Test Chat
        </h1>
        <button
          type="button"
          onClick={handleReset}
          className="text-xs text-neutral-500 hover:text-neutral-800 border border-neutral-200 rounded-md px-2.5 py-1"
        >
          New chat
        </button>
      </header>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {messages.length === 0 && (
          <div className="text-center text-sm text-neutral-400 mt-16">
            Ask something. Try &quot;what is 12 * 8?&quot; or &quot;what&apos;s today&apos;s date?&quot;
          </div>
        )}
        {messages.map((m) => (
          <Message key={m.id} message={m} />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-neutral-200 px-4 py-3">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message the agent…"
            rows={1}
            className="flex-1 resize-none rounded-xl border border-neutral-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-300 max-h-40"
          />
          <button
            type="button"
            onClick={handleSend}
            disabled={isStreaming || !input.trim()}
            className="rounded-xl bg-neutral-900 text-white text-sm px-4 py-2 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
