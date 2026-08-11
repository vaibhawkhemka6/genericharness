"use client";

import { useState } from "react";
import type { ToolCallInfo } from "@/lib/types";

export default function ToolCallBlock({ toolCall }: { toolCall: ToolCallInfo }) {
  const [expanded, setExpanded] = useState(false);
  const { name, arguments: args, result, status } = toolCall;

  return (
    <div className="my-1.5 rounded-lg border border-neutral-200 bg-neutral-50 text-xs">
      <button
        type="button"
        onClick={() => setExpanded((e) => !e)}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-neutral-600 hover:bg-neutral-100 rounded-lg"
      >
        <span aria-hidden>{status === "running" ? "⏳" : "🔧"}</span>
        <span className="font-mono">{name}</span>
        <span className="text-neutral-400">
          {status === "running" ? "running…" : "done"}
        </span>
        <span className="ml-auto text-neutral-400">{expanded ? "▾" : "▸"}</span>
      </button>
      {expanded && (
        <div className="border-t border-neutral-200 px-3 py-2 space-y-1.5">
          <div>
            <div className="text-neutral-400 mb-0.5">arguments</div>
            <pre className="whitespace-pre-wrap break-all font-mono text-neutral-700">
              {JSON.stringify(args, null, 2)}
            </pre>
          </div>
          {result !== undefined && (
            <div>
              <div className="text-neutral-400 mb-0.5">result</div>
              <pre className="whitespace-pre-wrap break-all font-mono text-neutral-700">
                {result}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
