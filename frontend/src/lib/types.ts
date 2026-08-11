export type ToolCallInfo = {
  name: string;
  arguments: Record<string, unknown>;
  result?: string;
  status: "running" | "done";
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolCalls: ToolCallInfo[];
  isStreaming?: boolean;
  error?: string;
};
