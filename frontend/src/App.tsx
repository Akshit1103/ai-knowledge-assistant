import { useEffect, useMemo, useRef, useState } from 'react'

type Source = {
  document_name: string
  chunk_text: string
  chunk_id: string
  page_number?: number | null
  relevance_score: number
  retrieval_method: string
}

type EvaluationResult = {
  faithfulness_score: number
  relevance_score: number
  faithfulness_reasoning: string
  relevance_reasoning: string
}

type PipelineStep = {
  name: string
  duration_ms: number
  details: Record<string, unknown>
}

type QueryResponse = {
  answer: string
  sources: Source[]
  evaluation?: EvaluationResult | null
  pipeline_steps: PipelineStep[]
  conversation_id: string
  confidence: number
  is_fallback: boolean
}

type DocumentInfo = {
  id: string
  name: string
  size_bytes: number
  num_chunks: number
  uploaded_at: string
}

type DocumentListResponse = {
  documents: DocumentInfo[]
  total_chunks: number
}

type ChatItem =
  | { id: string; role: 'user'; content: string }
  | { id: string; role: 'assistant'; content: string; response?: QueryResponse }

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000'

function fmtPct(v: number) {
  if (!Number.isFinite(v)) return '—'
  return `${Math.round(v * 100)}%`
}

function fmtBytes(bytes: number) {
  if (!Number.isFinite(bytes)) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let b = Math.max(0, bytes)
  let i = 0
  while (b >= 1024 && i < units.length - 1) {
    b /= 1024
    i += 1
  }
  return `${b.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) throw new Error(await res.text())
  return (await res.json()) as T
}

async function apiPostJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(await res.text())
  return (await res.json()) as T
}

async function apiDelete(path: string): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(await res.text())
}

async function apiUpload(file: File) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return (await res.json()) as { message: string; name: string; num_chunks: number }
}

function Card({
  title,
  children,
  right,
}: {
  title: string
  right?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="text-sm font-semibold text-white/90">{title}</div>
        {right}
      </div>
      {children}
    </div>
  )
}

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs text-white/80">
      {children}
    </span>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [chat, setChat] = useState<ChatItem[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [includeEval, setIncludeEval] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [docs, setDocs] = useState<DocumentInfo[]>([])
  const [totalChunks, setTotalChunks] = useState(0)
  const [uploading, setUploading] = useState(false)
  const fileRef = useRef<HTMLInputElement | null>(null)

  const lastAssistant = useMemo(() => {
    for (let i = chat.length - 1; i >= 0; i -= 1) {
      const it = chat[i]
      if (it.role === 'assistant') return it
    }
    return null
  }, [chat])

  async function refreshDocs() {
    const data = await apiGet<DocumentListResponse>('/api/documents')
    setDocs(data.documents ?? [])
    setTotalChunks(data.total_chunks ?? 0)
  }

  useEffect(() => {
    refreshDocs().catch((e) => setError(String(e?.message ?? e)))
  }, [])

  async function onUploadPicked(f: File | null) {
    if (!f) return
    setError(null)
    setUploading(true)
    try {
      const result = await apiUpload(f)
      await refreshDocs()
      setChat((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: `Indexed **${result.name}** into **${result.num_chunks}** chunks.`,
        },
      ])
    } catch (e) {
      setError(String((e as Error)?.message ?? e))
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  async function runQuery() {
    const q = query.trim()
    if (!q || loading) return
    setError(null)
    setLoading(true)

    const userItem: ChatItem = { id: crypto.randomUUID(), role: 'user', content: q }
    setChat((prev) => [...prev, userItem])
    setQuery('')

    try {
      const resp = await apiPostJson<QueryResponse>('/api/query', {
        query: q,
        conversation_id: conversationId,
        include_evaluation: includeEval,
      })
      setConversationId(resp.conversation_id)
      setChat((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: resp.answer,
          response: resp,
        },
      ])
    } catch (e) {
      setError(String((e as Error)?.message ?? e))
    } finally {
      setLoading(false)
    }
  }

  async function newConversation() {
    setConversationId(null)
    setChat([])
    setError(null)
  }

  async function clearConversation() {
    if (!conversationId) return newConversation()
    setError(null)
    try {
      await apiDelete(`/api/conversations/${encodeURIComponent(conversationId)}`)
      await newConversation()
    } catch (e) {
      setError(String((e as Error)?.message ?? e))
    }
  }

  async function deleteDoc(name: string) {
    if (!confirm(`Delete document "${name}" from the index?`)) return
    setError(null)
    try {
      await apiDelete(`/api/documents/${encodeURIComponent(name)}`)
      await refreshDocs()
    } catch (e) {
      setError(String((e as Error)?.message ?? e))
    }
  }

  return (
    <div className="min-h-dvh bg-[#070A12] text-white">
      <div className="mx-auto grid max-w-6xl grid-cols-1 gap-4 px-4 py-6 md:grid-cols-[320px_1fr]">
        <aside className="space-y-4">
          <div className="rounded-2xl border border-white/10 bg-gradient-to-b from-white/10 to-white/5 p-4">
            <div className="text-lg font-semibold tracking-tight">AI Knowledge Assistant</div>
            <div className="mt-1 text-sm text-white/70">
              Hybrid retrieval + reranking + memory + evaluation + guardrails
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <Badge>Docs: {docs.length}</Badge>
              <Badge>Chunks: {totalChunks}</Badge>
              {conversationId ? <Badge>Session: {conversationId.slice(0, 8)}…</Badge> : <Badge>No session</Badge>}
            </div>
          </div>

          <Card
            title="Documents"
            right={
              <button
                className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-xs text-white/80 hover:bg-white/10"
                onClick={() => refreshDocs().catch((e) => setError(String(e?.message ?? e)))}
                type="button"
              >
                Refresh
              </button>
            }
          >
            <div className="flex items-center gap-2">
              <input
                ref={fileRef}
                type="file"
                accept=".pdf,.txt,.md,.markdown"
                onChange={(e) => onUploadPicked(e.target.files?.[0] ?? null)}
                className="block w-full cursor-pointer rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm file:mr-3 file:rounded-md file:border-0 file:bg-white/10 file:px-3 file:py-1.5 file:text-xs file:text-white/90 hover:bg-white/10"
                disabled={uploading}
              />
            </div>
            <div className="mt-3 space-y-2">
              {docs.length === 0 ? (
                <div className="text-sm text-white/60">No documents indexed yet. Upload a PDF/TXT/MD.</div>
              ) : (
                docs.map((d) => (
                  <div
                    key={d.id}
                    className="flex items-start justify-between gap-3 rounded-lg border border-white/10 bg-white/5 px-3 py-2"
                  >
                    <div className="min-w-0">
                      <div className="truncate text-sm font-medium text-white/90">{d.name}</div>
                      <div className="mt-0.5 text-xs text-white/60">
                        {d.num_chunks} chunks · {fmtBytes(d.size_bytes)}
                      </div>
                    </div>
                    <button
                      className="shrink-0 rounded-md border border-white/10 bg-white/5 px-2 py-1 text-xs text-white/80 hover:bg-red-500/20 hover:text-red-200"
                      onClick={() => deleteDoc(d.name)}
                      type="button"
                    >
                      Delete
                    </button>
                  </div>
                ))
              )}
            </div>
          </Card>

          <Card title="Conversation">
            <div className="flex flex-wrap items-center gap-2">
              <button
                className="rounded-md border border-white/10 bg-white/5 px-3 py-2 text-sm hover:bg-white/10"
                onClick={() => newConversation().catch(() => {})}
                type="button"
              >
                New
              </button>
              <button
                className="rounded-md border border-white/10 bg-white/5 px-3 py-2 text-sm hover:bg-white/10"
                onClick={() => clearConversation().catch(() => {})}
                type="button"
              >
                Clear
              </button>
              <label className="ml-auto inline-flex cursor-pointer items-center gap-2 rounded-md border border-white/10 bg-white/5 px-3 py-2 text-sm hover:bg-white/10">
                <input
                  type="checkbox"
                  checked={includeEval}
                  onChange={(e) => setIncludeEval(e.target.checked)}
                  className="accent-white"
                />
                Include evaluation
              </label>
            </div>
          </Card>
        </aside>

        <main className="space-y-4">
          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
              <div className="flex-1">
                <div className="text-sm font-semibold text-white/90">Ask a question</div>
                <div className="text-xs text-white/60">
                  Tip: upload `documents/sample_ai_overview.md` and ask about “hybrid retrieval” or “guardrails”.
                </div>
              </div>
              <div className="text-xs text-white/60">
                API base: <span className="font-mono text-white/80">{API_BASE}</span>
              </div>
            </div>

            <div className="mt-4 flex gap-2">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    runQuery().catch(() => {})
                  }
                }}
                placeholder="Ask a question about your documents…"
                className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm outline-none placeholder:text-white/40 focus:border-white/20"
                disabled={loading}
              />
              <button
                className="rounded-xl bg-white px-4 py-3 text-sm font-semibold text-black hover:bg-white/90 disabled:opacity-50"
                onClick={() => runQuery().catch(() => {})}
                type="button"
                disabled={loading || query.trim().length === 0}
              >
                {loading ? 'Thinking…' : 'Ask'}
              </button>
            </div>

            {error ? (
              <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
                {error}
              </div>
            ) : null}
          </div>

          <div className="space-y-3">
            {chat.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-white/15 bg-white/5 p-10 text-center text-sm text-white/60">
                Ask your first question to start the conversation.
              </div>
            ) : (
              chat.map((m) => (
                <div
                  key={m.id}
                  className={`rounded-2xl border p-4 ${
                    m.role === 'user'
                      ? 'border-white/10 bg-white/5'
                      : 'border-emerald-400/15 bg-emerald-400/5'
                  }`}
                >
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <div className="text-xs font-semibold uppercase tracking-wide text-white/60">
                      {m.role === 'user' ? 'You' : 'Assistant'}
                    </div>
                    {m.role === 'assistant' && m.response ? (
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge>Confidence: {fmtPct(m.response.confidence)}</Badge>
                        {m.response.is_fallback ? <Badge>Fallback</Badge> : <Badge>Grounded</Badge>}
                        <Badge>Sources: {m.response.sources.length}</Badge>
                      </div>
                    ) : null}
                  </div>

                  <div className="whitespace-pre-wrap text-sm leading-relaxed text-white/90">{m.content}</div>

                  {m.role === 'assistant' && m.response ? (
                    <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-2">
                      <Card
                        title="Sources"
                        right={<span className="text-xs text-white/60">{m.response.sources.length} chunks</span>}
                      >
                        <div className="space-y-2">
                          {m.response.sources.length === 0 ? (
                            <div className="text-sm text-white/60">No sources returned.</div>
                          ) : (
                            m.response.sources.map((s, i) => (
                              <div
                                key={s.chunk_id}
                                className="rounded-lg border border-white/10 bg-white/5 p-3"
                              >
                                <div className="flex flex-wrap items-center gap-2">
                                  <div className="text-xs font-semibold text-white/80">Source {i + 1}</div>
                                  <Badge>{s.retrieval_method}</Badge>
                                  <Badge>Rel: {fmtPct(s.relevance_score)}</Badge>
                                  <span className="ml-auto truncate text-xs text-white/60">{s.document_name}</span>
                                </div>
                                <div className="mt-2 line-clamp-6 whitespace-pre-wrap text-xs leading-relaxed text-white/80">
                                  {s.chunk_text}
                                </div>
                              </div>
                            ))
                          )}
                        </div>
                      </Card>

                      <div className="space-y-3">
                        <Card title="Evaluation">
                          {m.response.evaluation ? (
                            <div className="space-y-2 text-sm">
                              <div className="flex flex-wrap gap-2">
                                <Badge>Faithfulness: {fmtPct(m.response.evaluation.faithfulness_score)}</Badge>
                                <Badge>Relevance: {fmtPct(m.response.evaluation.relevance_score)}</Badge>
                              </div>
                              <div className="text-xs text-white/70">
                                <div className="font-semibold text-white/80">Faithfulness reasoning</div>
                                <div className="mt-1 whitespace-pre-wrap">{m.response.evaluation.faithfulness_reasoning}</div>
                              </div>
                              <div className="text-xs text-white/70">
                                <div className="font-semibold text-white/80">Relevance reasoning</div>
                                <div className="mt-1 whitespace-pre-wrap">{m.response.evaluation.relevance_reasoning}</div>
                              </div>
                            </div>
                          ) : (
                            <div className="text-sm text-white/60">
                              Evaluation not included (or no context retrieved).
                            </div>
                          )}
                        </Card>

                        <Card title="Pipeline steps">
                          <div className="space-y-2">
                            {m.response.pipeline_steps.map((s) => (
                              <div
                                key={s.name}
                                className="flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-white/5 px-3 py-2"
                              >
                                <div className="text-sm text-white/85">{s.name}</div>
                                <div className="text-xs text-white/60">{s.duration_ms} ms</div>
                              </div>
                            ))}
                          </div>
                        </Card>
                      </div>
                    </div>
                  ) : null}
                </div>
              ))
            )}
          </div>

          {lastAssistant?.role === 'assistant' && lastAssistant.response ? (
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-xs text-white/60">
              Latest session id: <span className="font-mono text-white/80">{lastAssistant.response.conversation_id}</span>
            </div>
          ) : null}
        </main>
      </div>
    </div>
  )
}
