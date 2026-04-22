import Link from "next/link";

export default function HomePage() {
  return (
    <section className="card">
      <h2>Minimal RAG Demo</h2>
      <p>Upload knowledge files, then ask questions over indexed chunks.</p>
      <div className="actions">
        <Link className="button secondary" href="/datasets">
          Manage Datasets
        </Link>
        <Link className="button" href="/upload">
          Go to Upload
        </Link>
        <Link className="button secondary" href="/chat">
          Go to Chat
        </Link>
      </div>
    </section>
  );
}
