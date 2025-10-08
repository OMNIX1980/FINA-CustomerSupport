import os, json, shutil, chromadb, networkx as nx
from chromadb.config import Settings
import google.generativeai as genai
from typing import List, Dict, Any, Optional

# ----------------------------
# CONFIG
# ----------------------------
PATH_CHUNKS = r"C:\Users\luka\Downloads\fina_company_profile_chunks.jsonl"
PATH_TRIPLES = r"C:\Users\luka\Downloads\fina_knowledge_graph.jsonl"
PATH_FAQ    = r"C:\Users\luka\Downloads\fina_faq_prompts.jsonl"
PATH_SCHEMA = r"C:\Users\luka\Downloads\fina_schema_readme.json"

EMBED_MODEL = "text-embedding-004"   # 768-dim
CHAT_MODEL  = "gemini-2.5-flash"       # or "gemini-1.5-flash"

COLL_CHUNKS = "fina_chunks"
COLL_FAQ    = "fina_faq"

# Reset strategy (set True if you want to wipe collections before every run)
RESET_COLLECTIONS_FIRST = False
NUKE_PERSIST_DIR = False
PERSIST_DIR = "./chroma_data"

# ----------------------------
# GEMINI & CHROMA SETUP
# ----------------------------
if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError('GOOGLE_API_KEY not set. In PowerShell: setx GOOGLE_API_KEY "YOUR_KEY" and restart terminal.')
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

if NUKE_PERSIST_DIR and os.path.isdir(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

chroma = chromadb.Client(Settings(
    persist_directory=PERSIST_DIR,
    anonymized_telemetry=False
))

# ----------------------------
# UTILITIES
# ----------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed with Gemini (768-dim)."""
    vecs = []
    for t in texts:
        r = genai.embed_content(model=EMBED_MODEL, content=t)
        vecs.append(r["embedding"])
    return vecs

def to_primitive_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    """Make metadata Chroma-safe (no lists/dicts)."""
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

def reset_collections(names: List[str]):
    for name in names:
        try:
            chroma.delete_collection(name)
            print(f"[reset] Deleted collection '{name}'.")
        except Exception:
            pass

# ----------------------------
# 1) INDEX CHUNKS
# ----------------------------
def index_chunks(path=PATH_CHUNKS, collection_name=COLL_CHUNKS, batch=64):
    recs = read_jsonl(path)
    coll = chroma.get_or_create_collection(collection_name)

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

    print(f"[chunks] Indexed {len(recs)} records into collection '{collection_name}'.")
    return chroma.get_collection(collection_name)

# ----------------------------
# 2) KNOWLEDGE GRAPH
# ----------------------------
def load_graph(path=PATH_TRIPLES) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for t in read_jsonl(path):
        head, rel, tail = t["head"], t["relation"], t["tail"]
        G.add_edge(head, tail, relation=rel, prov=t.get("provenance", {}).get("chunk_id"))
    print(f"[graph] Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def graph_expand(G: nx.MultiDiGraph, entity: str, max_edges: int = 30) -> List[str]:
    facts = []
    if entity in G:
        for _, tail, d in G.out_edges(entity, data=True):
            facts.append(f"{entity} —{d.get('relation')}→ {tail} (prov: {d.get('prov')})")
            if len(facts) >= max_edges:
                break
    return facts

# ----------------------------
# 3) INDEX FAQ
# ----------------------------
def index_faq(path=PATH_FAQ, collection_name=COLL_FAQ, batch=50):
    faqs = read_jsonl(path)
    coll = chroma.get_or_create_collection(collection_name)
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
    print(f"[faq] Indexed {len(faqs)} QAs into '{collection_name}'.")
    return chroma.get_collection(collection_name)

def faq_search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    faq_coll = chroma.get_collection(COLL_FAQ)
    qemb = embed_texts([query])  # ensure same 768-dim model for queries
    res = faq_coll.query(query_embeddings=qemb, n_results=k, include=["metadatas","documents","distances"])
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "question": res["documents"][0][i],
            "meta": res["metadatas"][0][i],
            "score": res["distances"][0][i] if "distances" in res else None
        })
    return hits

# ----------------------------
# 4) RETRIEVAL + CONTEXT
# ----------------------------
def search_chunks(query: str, k: int = 8, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    coll = chroma.get_collection(COLL_CHUNKS)
    qemb = embed_texts([query])  # <<-- KEY FIX: use Gemini for queries too (768-dim)
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

def _first_entity_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    ecsv = meta.get("entities_csv")
    if not ecsv:
        return None
    parts = [p.strip() for p in ecsv.split("|") if p.strip()]
    return parts[0] if parts else None

def compose_context(query: str, G: Optional[nx.MultiDiGraph] = None, max_sections: int = 4) -> Dict[str, Any]:
    top_hits = search_chunks(query, k=12)
    seen_sections, picked = set(), []
    for h in top_hits:
        sec = h["meta"].get("section")
        if sec not in seen_sections:
            picked.append(h); seen_sections.add(sec)
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

# ----------------------------
# 5) GEMINI ANSWER
# ----------------------------
def answer_with_gemini(query: str, schema_path: str = PATH_SCHEMA, G: Optional[nx.MultiDiGraph]=None) -> str:
    schema_note = ""
    if os.path.exists(schema_path):
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_note = f.read()[:800]
        except Exception:
            pass

    pack = compose_context(query, G=G, max_sections=4)
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
    resp  = model.generate_content(prompt)
    srcs = ", ".join(sorted({h['meta']['chunk_id'] for h in used}))
    return f"{resp.text.strip()}\n\nSources: {srcs}"

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    if RESET_COLLECTIONS_FIRST:
        reset_collections([COLL_CHUNKS, COLL_FAQ])

    index_chunks()
    G = load_graph()
    index_faq()

    print("\n=== TEST 1: Product definition ===")
    print(answer_with_gemini("What is FINA IRP and who is it for?", G=G))

    print("\n=== TEST 2: Countries implemented ===")
    print(answer_with_gemini("List countries where FINA IRP is implemented.", G=G))

    print("\n=== TEST 3: Services overview ===")
    print(answer_with_gemini("Summarize FINA's core services in 4 bullets.", G=G))
