# app.py
import os, json, time, chromadb, networkx as nx
from chromadb.config import Settings
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import streamlit as st
from dotenv import load_dotenv

# ---------- CONFIG ----------
load_dotenv()  # reads .env if present
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

PERSIST_DIR = "./chroma_data"
os.makedirs(PERSIST_DIR, exist_ok=True)

DEFAULT_PATH_CHUNKS  = "fina_company_profile_chunks.jsonl"
DEFAULT_PATH_TRIPLES = "fina_knowledge_graph.jsonl"
DEFAULT_PATH_FAQ     = "fina_faq_prompts.jsonl"
DEFAULT_PATH_SCHEMA  = "fina_schema_readme.json"

EMBED_MODEL = "text-embedding-004"   # 768-dim
CHAT_MODEL  = "gemini-2.5-flash"       # or "gemini-1.5-flash"

COLL_CHUNKS = "fina_chunks"
COLL_FAQ    = "fina_faq"

# ---------- HELPERS ----------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def embed_texts(texts: List[str]) -> List[List[float]]:
    vecs = []
    for t in texts:
        r = genai.embed_content(model=EMBED_MODEL, content=t)
        vecs.append(r["embedding"])
    return vecs

def to_primitive_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple, set)):
            out[f"{k}_csv"] = "|".join(str(x) for x in v)
            out[f"{k}_count"] = len(v)
        elif isinstance(v, dict):
            out[f"{k}_json"] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out

@st.cache_resource(show_spinner=False)
def get_chroma():
    return chromadb.Client(Settings(persist_directory=PERSIST_DIR, anonymized_telemetry=False))

@st.cache_resource(show_spinner=False)
def load_graph(path_triples: str) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    if not os.path.exists(path_triples):
        return G
    for t in read_jsonl(path_triples):
        head, rel, tail = t["head"], t["relation"], t["tail"]
        G.add_edge(head, tail, relation=rel, prov=t.get("provenance", {}).get("chunk_id"))
    return G

def graph_expand(G: nx.MultiDiGraph, entity: str, max_edges: int = 20) -> List[str]:
    facts = []
    if entity in G:
        for _, tail, d in G.out_edges(entity, data=True):
            facts.append(f"{entity} —{d.get('relation')}→ {tail} (prov: {d.get('prov')})")
            if len(facts) >= max_edges:
                break
    return facts

def _first_entity_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    ecsv = meta.get("entities_csv")
    if not ecsv:
        return None
    parts = [p.strip() for p in ecsv.split("|") if p.strip()]
    return parts[0] if parts else None

def build_where_from_sections(sections: List[str]) -> Optional[Dict[str, Any]]:
    if not sections:
        return None
    return {"$or": [ {"section": {"$eq": s}} for s in sections ]}

# ---------- INDEXING ----------
def reset_collections(chroma_client, names: List[str]):
    for name in names:
        try:
            chroma_client.delete_collection(name)
            st.success(f"Deleted collection '{name}'.")
        except Exception as e:
            st.info(f"Collection '{name}' not found or already deleted.")

def index_chunks(chroma_client, path_chunks: str, batch: int = 64):
    recs = read_jsonl(path_chunks)
    coll = chroma_client.get_or_create_collection(COLL_CHUNKS)
    for i in range(0, len(recs), batch):
        batch_recs = recs[i:i+batch]
        ids   = [f"{r['doc_id']}::{r['chunk_id']}" for r in batch_recs]
        docs  = [r["text"] for r in batch_recs]
        metas = [to_primitive_meta({
            "doc_id": r.get("doc_id"),
            "chunk_id": r.get("chunk_id"),
            "section": r.get("section"),
            "title": r.get("title"),
            "entities": r.get("entities", []),
            "keywords": r.get("keywords", []),
        }) for r in batch_recs]
        embs  = embed_texts(docs)
        coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return len(recs)

def index_faq(chroma_client, path_faq: str, batch: int = 64):
    faqs = read_jsonl(path_faq)
    coll = chroma_client.get_or_create_collection(COLL_FAQ)
    for i in range(0, len(faqs), batch):
        b = faqs[i:i+batch]
        ids   = [f"{q['doc_id']}::faq::{i+j}" for j, q in enumerate(b)]
        docs  = [q["question"] for q in b]
        metas = [to_primitive_meta({
            "answer": q["answer"],
            "source_chunk_ids": q.get("source_chunk_ids", []),
        }) for q in b]
        embs  = embed_texts(docs)
        coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return len(faqs)

# ---------- RETRIEVAL ----------
def search_chunks(chroma_client, query: str, k: int = 8, where: Optional[Dict[str, Any]] = None):
    coll = chroma_client.get_collection(COLL_CHUNKS)
    qemb = embed_texts([query])  # ensure 768-dim
    kwargs: Dict[str, Any] = {"query_embeddings": qemb, "n_results": k, "include": ["metadatas","documents","distances"]}
    if isinstance(where, dict) and len(where) > 0:
        kwargs["where"] = where
    res = coll.query(**kwargs)
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i],
            "score": res["distances"][0][i] if "distances" in res else None,
        })
    return hits

def compose_context(chroma_client, query: str, G: Optional[nx.MultiDiGraph], sections_filter: List[str], k: int = 12, max_sections: int = 4):
    where = build_where_from_sections(sections_filter)
    top_hits = search_chunks(chroma_client, query, k=k, where=where)
    seen, picked = set(), []
    for h in top_hits:
        sec = h["meta"].get("section")
        if sec not in seen:
            picked.append(h); seen.add(sec)
        if len(picked) >= max_sections:
            break

    graph_facts = []
    if G and picked:
        first_ent = _first_entity_from_meta(picked[0]["meta"])
        if first_ent:
            graph_facts = graph_expand(G, first_ent, max_edges=12)

    ctx_blocks = []
    for h in picked:
        ctx_blocks.append(f"[CHUNK {h['meta']['chunk_id']} | {h['meta']['title']}] {h['text']}")
    if graph_facts:
        ctx_blocks.append("[GRAPH FACTS]\n" + "\n".join(f"- {f}" for f in graph_facts))

    return {"context_text": "\n\n".join(ctx_blocks), "used_chunks": picked, "graph_facts": graph_facts}

def answer_with_gemini(chroma_client, query: str, schema_path: str, G: Optional[nx.MultiDiGraph], sections_filter: List[str]):
    schema_note = ""
    if schema_path and os.path.exists(schema_path):
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_note = f.read()[:800]
        except Exception:
            pass

    pack = compose_context(chroma_client, query, G=G, sections_filter=sections_filter, k=12, max_sections=4)
    context_text, used = pack["context_text"], pack["used_chunks"]

    sys = (
        "You are a factual assistant for FINA LLC. Use ONLY the provided context. "
        "Cite sources as [CHUNK id]. If information is missing, say so plainly."
    )
    prompt = (
        f"{sys}\n\n"
        f"SCHEMA HINT (optional):\n{schema_note}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"USER QUESTION: {query}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer concisely and specifically.\n"
        "- Use bullet points for lists.\n"
        "- End with a 'Sources:' line listing CHUNK ids used.\n"
    )

    model = genai.GenerativeModel(CHAT_MODEL)
    resp = model.generate_content(prompt)
    srcs = ", ".join(sorted({h["meta"]["chunk_id"] for h in used}))
    return resp.text.strip(), used, srcs

# ---------- UI ----------
st.set_page_config(page_title="FINA – Agentic RAG", page_icon="💬", layout="wide")
st.title("FINA – Agentic RAG Chat 💬")

with st.sidebar:
    st.header("Settings")
    persist_info = st.empty()

    st.subheader("Data paths")
    path_chunks  = st.text_input("Chunks JSONL",  value=DEFAULT_PATH_CHUNKS)
    path_triples = st.text_input("Triples JSONL", value=DEFAULT_PATH_TRIPLES)
    path_faq     = st.text_input("FAQ JSONL",     value=DEFAULT_PATH_FAQ)
    path_schema  = st.text_input("Schema JSON",   value=DEFAULT_PATH_SCHEMA)

    st.subheader("Sections filter")
    selected_sections = st.multiselect(
        "Only search in:",
        options=["company_overview","products","services","projects","r_and_d"],
        default=[]
    )

    st.divider()
    st.subheader("Admin")
    colA, colB = st.columns(2)
    with colA:
        do_reset = st.button("Reset collections (wipe & rebuild)", type="secondary")
    with colB:
        do_index = st.button("Index / Reindex now", type="primary")

# init resources
chroma_client = get_chroma()
G = load_graph(path_triples)

# show collection stats
try:
    chunks_coll = chroma_client.get_or_create_collection(COLL_CHUNKS)
    faq_coll = chroma_client.get_or_create_collection(COLL_FAQ)
    persist_info.info(f"Collections -> {COLL_CHUNKS}: {chunks_coll.count()} | {COLL_FAQ}: {faq_coll.count()}")
except Exception:
    persist_info.warning("Chroma not initialized yet.")

# admin actions
if do_reset:
    reset_collections(chroma_client, [COLL_CHUNKS, COLL_FAQ])
    st.toast("Collections deleted. Click 'Index / Reindex now' to rebuild.", icon="🧹")

if do_index:
    with st.spinner("Indexing chunks..."):
        n_chunks = index_chunks(chroma_client, path_chunks)
    with st.spinner("Indexing FAQ..."):
        n_faq = index_faq(chroma_client, path_faq)
    G = load_graph(path_triples)
    st.success(f"Indexed {n_chunks} chunks and {n_faq} FAQs. Graph nodes: {G.number_of_nodes()}")

# chat area
st.subheader("Ask a question")
q = st.text_input("Your question", placeholder="e.g., What is FINA IRP and where is it implemented?")
show_chunks = st.checkbox("Show retrieved chunks", value=True)

if st.button("Ask"):
    if not q.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer, used, srcs = answer_with_gemini(
                    chroma_client=chroma_client,
                    query=q.strip(),
                    schema_path=path_schema,
                    G=G,
                    sections_filter=selected_sections
                )
                st.markdown(answer)
                st.markdown(f"**Sources:** {srcs}")

                if show_chunks and used:
                    st.divider()
                    st.caption("Retrieved chunks")
                    for h in used:
                        with st.expander(f"[{h['meta'].get('chunk_id')}] {h['meta'].get('title')}  ·  section={h['meta'].get('section')}  ·  score={h.get('score')}"):
                            st.write(h["text"])
                            st.json(h["meta"])
            except Exception as e:
                st.error(f"Error: {repr(e)}")
