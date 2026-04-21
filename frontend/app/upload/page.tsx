"use client";

import { DragEvent, useMemo, useState } from "react";

import { uploadFile } from "@/lib/api";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [message, setMessage] = useState<string>("");
  const [error, setError] = useState<string>("");

  const accepted = useMemo(() => ".pdf,.docx,.txt", []);

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) {
      setFile(dropped);
      setMessage("");
      setError("");
    }
  }

  async function handleUpload() {
    if (!file) return;
    setIsUploading(true);
    setError("");
    setMessage("");
    try {
      const result = await uploadFile(file);
      setMessage(
        `Indexed "${result.file_name}" with ${result.chunks_indexed} chunks (document #${result.document_id}).`
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <section className="card">
      <h2>Upload Documents</h2>
      <p>Accepted formats: PDF, DOCX, TXT.</p>

      <div
        className="dropzone"
        onDragOver={(event) => event.preventDefault()}
        onDrop={onDrop}
      >
        <p>{file ? `Selected: ${file.name}` : "Drag and drop file here"}</p>
        <label className="button secondary" htmlFor="file-input">
          Choose File
        </label>
        <input
          id="file-input"
          type="file"
          accept={accepted}
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          hidden
        />
      </div>

      <button className="button" disabled={!file || isUploading} onClick={handleUpload}>
        {isUploading ? "Uploading..." : "Upload & Index"}
      </button>

      {message && <p className="success">{message}</p>}
      {error && <p className="error">{error}</p>}
    </section>
  );
}
