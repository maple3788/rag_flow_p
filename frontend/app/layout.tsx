import "./globals.css";
import Link from "next/link";
import type { ReactNode } from "react";

export const metadata = {
  title: "RAG Flow",
  description: "Minimal RAG frontend",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <h1>RAG Flow</h1>
          <nav>
            <Link href="/datasets">Datasets</Link>
            <Link href="/chat">Chat</Link>
            <Link href="/workflow">Workflow</Link>
          </nav>
        </header>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
