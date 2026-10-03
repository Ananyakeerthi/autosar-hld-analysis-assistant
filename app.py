"""
AUTOSAR HLD Document Analysis Assistant — Streamlit UI
--------------------------------------------------------
Wraps the notebook pipeline (Case_Study_1_AUTOSAR_HLD_Assistant.ipynb) into a
single-page web app. Every function here is a direct port of the notebook's
logic — same prompts, same rules, same behavior. Only the interface changed.

Run with:
    streamlit run app.py

Requirements (same as the notebook):
    pip install streamlit pymupdf sentence-transformers faiss-cpu numpy requests

Ollama must be running locally with llama3.2 pulled:
    ollama pull llama3.2
    ollama serve   (usually starts automatically after install)
"""

import os
import json
import glob
import difflib
import tempfile
from datetime import datetime

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AUTOSAR HLD Assistant",
    page_icon="📘",
    layout="wide",
)

EXPORT_DIR = "hld_analysis_exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")


# ----------------------------------------------------------------------------
# Core pipeline functions (ported 1:1 from the notebook)
# ----------------------------------------------------------------------------

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_pages(pdf_path):
    """Extract text from each page of the PDF, keeping page numbers for citation."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def chunk_pages(pages, chunk_size=800, overlap=100):
    """Split page text into overlapping chunks; retain page number for each chunk."""
    chunks = []
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({"page": p["page"], "text": chunk_text})
            start += chunk_size - overlap
    return chunks


def build_index(chunks):
    embedding_model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


def retrieve(query, chunks, index, top_k=4):
    embedding_model = get_embedding_model()
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]


def call_llm(prompt, temperature=0.1):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["response"]


def build_prompt(query, results):
    context = "\n\n".join([f"[Page {r['page']}]\n{r['text']}" for r in results])
    return f"""You are an AUTOSAR HLD Document Analysis Assistant.

Answer the user's question using ONLY the provided document context.

Important rules:
1. Do not add information from your general knowledge.
2. Do not invent components, interfaces, or relationships.
3. Distinguish between:
   - Software components
   - Software layers
   - Interfaces
   - Services/functions
4. If the question refers to a diagram, view, table, or architecture section,
report the entities and labels explicitly shown in that source section.
Preserve the terminology used in the document.
Do not reinterpret or rename them.
5. Cite the page number immediately after each relevant statement.
6. If the context does not contain enough information, say:
   "The provided context does not contain enough information to answer this."

Document context:
{context}

Question:
{query}

Answer:"""


def ask(query, chunks, index, top_k=4):
    results = retrieve(query, chunks, index, top_k=top_k)
    prompt = build_prompt(query, results)
    answer = call_llm(prompt)
    return answer, results


def extract_architecture(query, chunks, index, top_k=4, force_pages=(79,)):
    results = retrieve(query, chunks, index, top_k=top_k)

    # Force-include known high-value architecture pages, same as the original notebook.
    for page_num in force_pages:
        for c in chunks:
            if c["page"] == page_num and c not in results:
                results.append(c)

    context = "\n\n".join([f"[Page {r['page']}]\n{r['text']}" for r in results])
    prompt = f"""You are an AUTOSAR architecture document analysis assistant.

Use ONLY the supplied document context.

Extract architecture information explicitly present in the context.

Respond with ONLY the four sections below — nothing before them, nothing
after them, and do not repeat these instructions or the word "RULES" in
your answer.

COMPONENTS:
- <component name> — Page <number>
(or, if none are explicitly present: None found)

INTERFACES:
- <interface name> — Page <number>
(or, if none are explicitly present: None found)

LAYERS:
- <layer name> — Page <number>
(or, if none are explicitly present: None found)

OTHER ENTITIES:
- <entity name> — Page <number>
(or, if none are explicitly present: None found)

Formatting rules you must silently follow (do not print this list):
- Do not use general AUTOSAR knowledge, only the context below.
- Do not invent or infer entities.
- Do not treat section headings as components or interfaces.
- Only include an item when the context explicitly supports it.
- Keep the terminology used in the document.
- A service/function should NOT be classified as a component.
- A layer should NOT be classified as a component.
- Never write the literal word "Name" — always write the actual entity name.
- Do not repeat the same finding twice within a section.

DOCUMENT CONTEXT:
{context}

QUESTION:
{query}

Now write ONLY the four sections (COMPONENTS, INTERFACES, LAYERS, OTHER ENTITIES):
"""
    answer = call_llm(prompt)
    return answer, results


def extract_dependencies(query, chunks, index, top_k=6):
    results = retrieve(query, chunks, index, top_k=top_k)
    context = "\n\n".join([f"[Page {r['page']}]\n{r['text']}" for r in results])
    prompt = f"""
You are an AUTOSAR HLD document analysis assistant.

Use ONLY the supplied document context.

Identify dependencies or relationships that are EXPLICITLY
described in the document.

Return the answer in exactly this format:

DEPENDENCIES:

1. SOURCE:
   - Name: <source entity>
   - Depends on / interacts with: <target entity>
   - Relationship: <brief description>
   - Page: <page number>

RULES:
- Use only information explicitly supported by the context.
- Do not use general AUTOSAR knowledge.
- Do not invent relationships.
- Do not assume that two entities are connected just because
  they appear on the same page.
- Preserve terminology from the document.
- Include the page number supporting each relationship.
- If no explicit dependency is found, write:
  "No explicit dependencies found in the provided context."

DOCUMENT CONTEXT:
{context}

QUESTION:
{query}

DEPENDENCY ANALYSIS:
"""
    answer = call_llm(prompt)
    return answer, results


def generate_component_report(component_name, chunks, index, top_k=6):
    query = f"""
    Find all information in the document related to the component:
    {component_name}

    Include its description, interfaces, layers, interactions,
    dependencies, and other explicitly mentioned information.
    """
    results = retrieve(query, chunks, index, top_k=top_k)
    context = "\n\n".join([f"[Page {r['page']}]\n{r['text']}" for r in results])
    prompt = f"""
You are an AUTOSAR HLD Document Analysis Assistant.

Generate a structured report for the requested architecture entity.

Use ONLY the supplied document context.
Do not use general AUTOSAR knowledge.
Do not invent missing information.

Return exactly this format:

COMPONENT REPORT
================

COMPONENT:
<name>

DESCRIPTION:
<information explicitly stated in the document>

INTERFACES:
- <interface> — Page X

LAYERS:
- <layer> — Page X

INTERACTIONS:
- <entity> — <relationship> — Page X

DEPENDENCIES:
- <entity> — <dependency/relationship> — Page X

OTHER INFORMATION:
- <relevant information> — Page X

DOCUMENT REFERENCES:
- Page X

RULES:
1. Include only information supported by the supplied context.
2. Every factual statement must have a page citation.
3. Do not invent interfaces, dependencies, or relationships.
4. If a section has no supported information, write "None found".
5. Keep the terminology from the document.
6. Do not confuse a service or function with a component.
7. Do not treat section headings as architecture entities.

DOCUMENT CONTEXT:
{context}

REQUESTED COMPONENT:
{component_name}

COMPONENT REPORT:
"""
    answer = call_llm(prompt)
    return answer, results


def check_completeness(query, chunks, index, top_k=8):
    results = retrieve(query, chunks, index, top_k=top_k)
    context = "\n\n".join([f"[Page {r['page']}]\n{r['text']}" for r in results])
    prompt = f"""You are an AUTOSAR HLD Document Analysis Assistant.

Analyze ONLY the supplied document context.

Your task is to identify:
1. Missing information
2. Potential inconsistencies or mismatches
3. Areas where the document does not provide enough information

Do NOT use general AUTOSAR knowledge.
Do NOT assume something is missing simply because it is not
present in the retrieved context.

Return exactly this format:

COMPLETENESS / MISMATCH CHECK
=============================

MISSING INFORMATION:
- <what is missing, in plain words> — Page X

POTENTIAL MISMATCHES:
- <what conflicts or seems inconsistent, in plain words> — Page X

INCOMPLETE / UNCLEAR AREAS:
- <what is unclear or underspecified, in plain words> — Page X

WELL-SPECIFIED AREAS:
- <what is clearly and fully specified, in plain words> — Page X

OVERALL ASSESSMENT:
<brief assessment>

RULES:
- Every bullet MUST describe the actual finding in a few words —
  never write just "Finding — Page X" with no description.
- A single page number must NOT appear in both MISSING INFORMATION
  (or POTENTIAL MISMATCHES / INCOMPLETE) and WELL-SPECIFIED AREAS.
  If a page has both strong and weak parts, pick the single category
  that best describes its overall state and mention the nuance in
  the description text instead of listing it twice.
- Only report findings supported by the supplied context.
- Do not invent mismatches.
- Do not treat absence from the retrieved context as proof that
  something is absent from the entire document.
- Clearly distinguish "not found in the provided context" from
  "missing from the document".
- If no potential mismatch is found, write "None identified".
- If no clear missing information is found, write "None identified".
- Keep terminology from the document.

DOCUMENT CONTEXT:
{context}

QUESTION:
{query}

COMPLETENESS / MISMATCH ANALYSIS:
"""
    answer = call_llm(prompt)
    return answer, results


def extract_and_chunk(pdf_path, chunk_size=800, overlap=100):
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    return pages, chunks


def diff_pages(pages_a, pages_b, similarity_threshold=0.85):
    pages_a_map = {p["page"]: p["text"] for p in pages_a}
    pages_b_map = {p["page"]: p["text"] for p in pages_b}
    all_page_numbers = sorted(set(pages_a_map.keys()) | set(pages_b_map.keys()))

    diff_results = []
    for page_num in all_page_numbers:
        text_a = pages_a_map.get(page_num)
        text_b = pages_b_map.get(page_num)

        if text_a is None and text_b is not None:
            diff_results.append({"page": page_num, "status": "ADDED", "text_b": text_b})
        elif text_a is not None and text_b is None:
            diff_results.append({"page": page_num, "status": "REMOVED", "text_a": text_a})
        else:
            ratio = difflib.SequenceMatcher(None, text_a, text_b).ratio()
            if ratio >= similarity_threshold:
                diff_results.append({"page": page_num, "status": "UNCHANGED", "similarity": round(ratio, 3)})
            else:
                diff_results.append({
                    "page": page_num, "status": "CHANGED", "similarity": round(ratio, 3),
                    "text_a": text_a, "text_b": text_b,
                })
    return diff_results


def summarize_diff_for_llm(diff_results, max_changed_pages=8, snippet_chars=500):
    added = [d for d in diff_results if d["status"] == "ADDED"]
    removed = [d for d in diff_results if d["status"] == "REMOVED"]
    changed = sorted(
        [d for d in diff_results if d["status"] == "CHANGED"],
        key=lambda d: d["similarity"],
    )[:max_changed_pages]

    context_parts = []
    for d in added:
        context_parts.append(f"[ADDED - Page {d['page']} in Version B]\n{d['text_b'][:snippet_chars]}")
    for d in removed:
        context_parts.append(f"[REMOVED - Page {d['page']} from Version A]\n{d['text_a'][:snippet_chars]}")
    for d in changed:
        context_parts.append(
            f"[CHANGED - Page {d['page']} - similarity {d['similarity']}]\n"
            f"VERSION A:\n{d['text_a'][:snippet_chars]}\n\n"
            f"VERSION B:\n{d['text_b'][:snippet_chars]}"
        )
    return "\n\n".join(context_parts)


def compare_revisions(pdf_path_a, pdf_path_b, chunk_size=800, overlap=100,
                       similarity_threshold=0.85, max_changed_pages=8, progress_cb=None):
    if progress_cb: progress_cb("Extracting and chunking Version A...")
    pages_a, chunks_a = extract_and_chunk(pdf_path_a, chunk_size, overlap)

    if progress_cb: progress_cb(f"Version A: {len(pages_a)} pages, {len(chunks_a)} chunks")
    pages_b, chunks_b = extract_and_chunk(pdf_path_b, chunk_size, overlap)

    if progress_cb: progress_cb(f"Version B: {len(pages_b)} pages, {len(chunks_b)} chunks")
    if progress_cb: progress_cb("Computing deterministic page-level diff...")
    diff_results = diff_pages(pages_a, pages_b, similarity_threshold)

    added_count = sum(1 for d in diff_results if d["status"] == "ADDED")
    removed_count = sum(1 for d in diff_results if d["status"] == "REMOVED")
    changed_count = sum(1 for d in diff_results if d["status"] == "CHANGED")
    unchanged_count = sum(1 for d in diff_results if d["status"] == "UNCHANGED")

    if progress_cb:
        progress_cb(f"Diff summary: {added_count} added, {removed_count} removed, "
                     f"{changed_count} changed, {unchanged_count} unchanged pages")

    context = summarize_diff_for_llm(diff_results, max_changed_pages)

    prompt = f"""
You are an AUTOSAR HLD Document Analysis Assistant performing a revision comparison.

Use ONLY the page-level diff data below, which was computed deterministically
by comparing Version A (older) and Version B (newer) of the same document.

Do NOT use general AUTOSAR knowledge.
Do NOT invent additions, removals, or changes that are not present in the diff data.

Return exactly this format:

REVISION COMPARISON REPORT
===========================

ADDED:
- Finding — Page X (Version B)

REMOVED:
- Finding — Page X (Version A)

CHANGED:
- Finding — Page X (summarize what changed between Version A and Version B)

POTENTIAL INCONSISTENCIES:
- Finding — Page X (e.g. a component/interface described differently, or
  contradicted, between the two versions)

OVERALL ASSESSMENT:
<brief assessment of the scope and nature of the revision>

RULES:
- Only report findings supported by the supplied diff data.
- Every finding must include its page number.
- If a category has no findings, write "None identified".
- Keep terminology from the document.

DIFF DATA:
{context}

Remember: respond using EXACTLY the section headers ADDED / REMOVED / CHANGED /
POTENTIAL INCONSISTENCIES / OVERALL ASSESSMENT shown above. Do not use any other
headers or free-form prose sections.

REVISION COMPARISON ANALYSIS:
"""
    answer = call_llm(prompt)

    return {
        "answer": answer,
        "diff_results": diff_results,
        "summary_counts": {
            "added": added_count, "removed": removed_count,
            "changed": changed_count, "unchanged": unchanged_count,
        },
    }


def export_result(label, content, sources=None, metadata=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{label}_{timestamp}"

    md_path = os.path.join(EXPORT_DIR, f"{base_filename}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {label.replace('_', ' ').title()}\n\n")
        f.write(f"*Generated: {timestamp}*\n\n")
        if metadata:
            f.write("## Metadata\n")
            for k, v in metadata.items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n")
        f.write("## Output\n\n")
        f.write(content)
        f.write("\n")

    json_path = os.path.join(EXPORT_DIR, f"{base_filename}.json")
    export_data = {
        "label": label, "timestamp": timestamp, "metadata": metadata or {},
        "content": content, "sources": sources or [],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    return md_path, json_path


def save_uploaded_pdf(uploaded_file):
    """Streamlit gives us an in-memory file; fitz needs a real path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return True, models
    except Exception:
        return False, []


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "pages" not in st.session_state:
    st.session_state.pages = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("📘 AUTOSAR HLD Assistant")
st.sidebar.caption("RAG pipeline for AUTOSAR HLD documents — Q&A, extraction, and revision comparison.")

ollama_ok, ollama_models = check_ollama()
if ollama_ok:
    st.sidebar.success(f"Ollama connected ({', '.join(ollama_models) or 'no models found'})")
else:
    st.sidebar.error("Ollama not reachable at localhost:11434 — start it before running any analysis.")

page = st.sidebar.radio(
    "Go to",
    [
        "1. Upload & Index",
        "2. Ask a Question",
        "3. Architecture Extraction",
        "4. Dependency Mapping",
        "5. Component Report",
        "6. Completeness Check",
        "7. Revision Comparison",
        "8. Exported Files",
    ],
)


# ----------------------------------------------------------------------------
# 1. Upload & Index
# ----------------------------------------------------------------------------
if page == "1. Upload & Index":
    st.header("Upload & Index a Document")
    st.write("Upload the AUTOSAR HLD PDF you want to analyze (Q&A, extraction, completeness). "
             "This builds the chunk index used by every tab except Revision Comparison, "
             "which works on two separate files directly.")

    uploaded = st.file_uploader("AUTOSAR HLD PDF", type="pdf")

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk size", value=800, step=100)
    with col2:
        overlap = st.number_input("Overlap", value=100, step=25)

    if uploaded is not None and st.button("Extract, chunk & index", type="primary"):
        with st.spinner("Extracting pages..."):
            path = save_uploaded_pdf(uploaded)
            pages = extract_pages(path)
        st.write(f"Extracted **{len(pages)}** pages.")

        with st.spinner("Chunking..."):
            chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        st.write(f"Created **{len(chunks)}** chunks.")

        with st.spinner("Building embeddings + FAISS index (this can take a minute)..."):
            index, embeddings = build_index(chunks)
        st.success(f"Indexed {index.ntotal} chunk embeddings.")

        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.doc_name = uploaded.name

    if st.session_state.chunks is not None:
        st.info(f"Currently indexed: **{st.session_state.doc_name}** "
                f"({len(st.session_state.pages)} pages, {len(st.session_state.chunks)} chunks)")


# ----------------------------------------------------------------------------
# Shared guard for tabs 2-6
# ----------------------------------------------------------------------------
def require_index():
    if st.session_state.chunks is None or st.session_state.index is None:
        st.warning("No document indexed yet. Go to **1. Upload & Index** first.")
        st.stop()


# ----------------------------------------------------------------------------
# 2. Ask a Question
# ----------------------------------------------------------------------------
if page == "2. Ask a Question":
    st.header("Ask a Question")
    require_index()

    query = st.text_area("Your question", "Summarize the main software components described in this document.")
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, 4)

    if st.button("Ask", type="primary"):
        with st.spinner("Retrieving and generating answer..."):
            answer, results = ask(query, st.session_state.chunks, st.session_state.index, top_k=top_k)
        st.session_state["last_qa"] = (answer, results, query, top_k)

    if "last_qa" in st.session_state:
        answer, results, saved_query, saved_top_k = st.session_state["last_qa"]
        st.markdown(answer)
        with st.expander("Show retrieved source chunks"):
            for r in results:
                st.markdown(f"**Page {r['page']}**")
                st.text(r["text"][:500])

        if st.button("Export this answer"):
            md, js = export_result("qa_answer", answer, sources=[r["page"] for r in results],
                                    metadata={"query": saved_query, "top_k": saved_top_k})
            st.success(f"Exported: {md}")


# ----------------------------------------------------------------------------
# 3. Architecture Extraction
# ----------------------------------------------------------------------------
if page == "3. Architecture Extraction":
    st.header("Architecture Extraction")
    require_index()

    query = st.text_area(
        "Extraction query",
        "Identify the architecture components, interfaces, layers and other entities "
        "explicitly shown or described in the document.",
    )
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, 4, key="arch_topk")

    if st.button("Extract architecture", type="primary"):
        with st.spinner("Extracting..."):
            answer, results = extract_architecture(query, st.session_state.chunks, st.session_state.index, top_k=top_k)
        st.session_state["last_architecture"] = (answer, results, query)
        st.markdown(answer)

    if "last_architecture" in st.session_state and st.button("Export architecture extraction"):
        answer, results, query = st.session_state["last_architecture"]
        md, js = export_result("architecture_extraction", answer, sources=[r["page"] for r in results],
                                metadata={"query": query})
        st.success(f"Exported: {md}")


# ----------------------------------------------------------------------------
# 4. Dependency Mapping
# ----------------------------------------------------------------------------
if page == "4. Dependency Mapping":
    st.header("Dependency Mapping")
    require_index()

    query = st.text_area(
        "Dependency query",
        "Identify the explicit dependencies and interactions between software components, "
        "interfaces, RTE, BSW, and other architecture entities.",
    )
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 12, 6, key="dep_topk")

    if st.button("Extract dependencies", type="primary"):
        with st.spinner("Extracting..."):
            answer, results = extract_dependencies(query, st.session_state.chunks, st.session_state.index, top_k=top_k)
        st.session_state["last_dependencies"] = (answer, results, query)
        st.markdown(answer)

    if "last_dependencies" in st.session_state and st.button("Export dependency mapping"):
        answer, results, query = st.session_state["last_dependencies"]
        md, js = export_result("dependency_mapping", answer, sources=[r["page"] for r in results],
                                metadata={"query": query})
        st.success(f"Exported: {md}")


# ----------------------------------------------------------------------------
# 5. Component Report
# ----------------------------------------------------------------------------
if page == "5. Component Report":
    st.header("Component Report")
    require_index()

    component_name = st.text_input("Component name (e.g. RTE, ECU Abstraction Layer)", "RTE")
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 12, 6, key="comp_topk")

    if st.button("Generate report", type="primary"):
        with st.spinner("Generating..."):
            answer, results = generate_component_report(component_name, st.session_state.chunks,
                                                          st.session_state.index, top_k=top_k)
        st.session_state["last_component"] = (answer, results, component_name)
        st.markdown(answer)

    if "last_component" in st.session_state and st.button("Export component report"):
        answer, results, component_name = st.session_state["last_component"]
        md, js = export_result(f"component_report_{component_name.lower().replace(' ', '_')}", answer,
                                sources=[r["page"] for r in results], metadata={"component": component_name})
        st.success(f"Exported: {md}")


# ----------------------------------------------------------------------------
# 6. Completeness Check
# ----------------------------------------------------------------------------
if page == "6. Completeness Check":
    st.header("Completeness / Mismatch Check")
    require_index()

    query = st.text_area(
        "Query",
        "Check the AUTOSAR architecture information for missing, incomplete, "
        "or potentially inconsistent information.",
    )
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 15, 8, key="comp_check_topk")

    if st.button("Run completeness check", type="primary"):
        with st.spinner("Analyzing..."):
            answer, results = check_completeness(query, st.session_state.chunks, st.session_state.index, top_k=top_k)
        st.session_state["last_completeness"] = (answer, results, query)
        st.markdown(answer)

    if "last_completeness" in st.session_state and st.button("Export completeness check"):
        answer, results, query = st.session_state["last_completeness"]
        md, js = export_result("completeness_check", answer, sources=[r["page"] for r in results],
                                metadata={"query": query})
        st.success(f"Exported: {md}")


# ----------------------------------------------------------------------------
# 7. Revision Comparison
# ----------------------------------------------------------------------------
if page == "7. Revision Comparison":
    st.header("Revision Comparison")
    st.write("Upload two versions of the same document. Version A = older/baseline, Version B = newer/updated.")

    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Version A (older)", type="pdf", key="rev_a")
    with col2:
        file_b = st.file_uploader("Version B (newer)", type="pdf", key="rev_b")

    max_changed_pages = st.slider("Max changed pages sent to the LLM", 4, 40, 30)

    if file_a is not None and file_b is not None and st.button("Compare revisions", type="primary"):
        path_a = save_uploaded_pdf(file_a)
        path_b = save_uploaded_pdf(file_b)

        status = st.empty()
        result = compare_revisions(
            path_a, path_b, max_changed_pages=max_changed_pages,
            progress_cb=lambda msg: status.write(msg),
        )
        st.session_state["last_revision"] = (result, file_a.name, file_b.name)

        counts = result["summary_counts"]
        st.write(f"**Diff summary:** {counts['added']} added, {counts['removed']} removed, "
                 f"{counts['changed']} changed, {counts['unchanged']} unchanged pages")
        st.markdown(result["answer"])

    if "last_revision" in st.session_state and st.button("Export revision comparison"):
        result, name_a, name_b = st.session_state["last_revision"]
        md, js = export_result("revision_comparison", result["answer"],
                                metadata={"pdf_a": name_a, "pdf_b": name_b,
                                          "diff_summary": result["summary_counts"]})
        st.success(f"Exported: {md}")


# ----------------------------------------------------------------------------
# 8. Exported Files
# ----------------------------------------------------------------------------
if page == "8. Exported Files":
    st.header("Exported Files")
    st.write(f"All exports are saved to `{os.path.abspath(EXPORT_DIR)}`")

    files = sorted(glob.glob(os.path.join(EXPORT_DIR, "*.md")), reverse=True)
    if not files:
        st.info("No exports yet. Run an analysis and click its Export button.")
    else:
        for f in files:
            with st.expander(os.path.basename(f)):
                with open(f, "r", encoding="utf-8") as fh:
                    content = fh.read()
                st.markdown(content)
                st.download_button("Download this file", content, file_name=os.path.basename(f))

    st.divider()
    st.subheader("Combine all latest exports into one report")
    if st.button("Build combined report"):
        order = ["architecture_extraction", "dependency_mapping", "component_report",
                 "completeness_check", "revision_comparison", "qa_answer"]
        all_files = glob.glob(os.path.join(EXPORT_DIR, "*.md"))
        latest_by_label = {}
        for f in all_files:
            filename = os.path.basename(f)
            for label in order:
                if filename.startswith(label):
                    if label not in latest_by_label or filename > os.path.basename(latest_by_label[label]):
                        latest_by_label[label] = f

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = os.path.join(EXPORT_DIR, f"COMBINED_REPORT_{timestamp}.md")
        with open(combined_path, "w", encoding="utf-8") as out:
            out.write("# AUTOSAR HLD Document Analysis — Combined Report\n\n")
            out.write(f"*Generated: {timestamp}*\n\n---\n\n")
            for label in order:
                if label in latest_by_label:
                    with open(latest_by_label[label], "r", encoding="utf-8") as f:
                        out.write(f.read())
                    out.write("\n\n---\n\n")
        st.success(f"Combined report saved: {combined_path}")
