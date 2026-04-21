"use client";

import { FormEvent, useState } from "react";

import { sendChat, type ChatResponse } from "@/lib/api";

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: ChatResponse["sources"];
};

export default function ChatPage() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || loading) return;

    setError("");
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setQuery("");

    try {
      const result = await sendChat(trimmed, 5);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: result.answer,
          sources: result.sources,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send message");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="chat-wrap">
      <h2>Chat with Documents</h2>
      <div className="chat-box">
        {messages.length === 0 && <p className="muted">Ask a question to start.</p>}
        {messages.map((message, idx) => (
          <article key={`${message.role}-${idx}`} className={`message ${message.role}`}>
            <p className="message-role">{message.role === "user" ? "You" : "Assistant"}</p>
            <p>{message.content}</p>
            {message.sources && message.sources.length > 0 && (
              <div className="sources">
                <p className="message-role">Sources</p>
                {message.sources.map((source) => (
                  <div key={source.chunk_id} className="source-item">
                    <p>
                      <strong>{source.document_name}</strong> (chunk {source.chunk_id}, score{" "}
                      {source.score.toFixed(4)})
                    </p>
                    <p>{source.content}</p>
                  </div>
                ))}
              </div>
            )}
          </article>
        ))}
      </div>

      <form className="chat-form" onSubmit={onSubmit}>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Ask a question about uploaded files..."
          disabled={loading}
        />
        <button className="button" type="submit" disabled={loading || !query.trim()}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}
    </section>
  );
}
