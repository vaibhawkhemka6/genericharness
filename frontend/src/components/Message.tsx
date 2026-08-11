import type { ChatMessage } from "@/lib/types";
import ToolCallBlock from "./ToolCallBlock";

export default function Message({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[80%] ${isUser ? "items-end" : "items-start"} flex flex-col`}>
        {!isUser && message.toolCalls.length > 0 && (
          <div className="w-full mb-1">
            {message.toolCalls.map((tc, i) => (
              <ToolCallBlock key={i} toolCall={tc} />
            ))}
          </div>
        )}

        {(message.content || message.isStreaming) && (
          <div
            className={`rounded-2xl px-4 py-2.5 whitespace-pre-wrap break-words leading-relaxed ${
              isUser
                ? "bg-neutral-900 text-white rounded-br-sm"
                : "bg-neutral-100 text-neutral-900 rounded-bl-sm"
            }`}
          >
            {message.content}
            {message.isStreaming && (
              <span className="inline-block w-1.5 h-4 ml-0.5 -mb-0.5 bg-neutral-400 animate-pulse" />
            )}
          </div>
        )}

        {message.error && (
          <div className="mt-1 rounded-lg bg-red-50 text-red-700 text-sm px-3 py-2 border border-red-200">
            {message.error}
          </div>
        )}
      </div>
    </div>
  );
}
