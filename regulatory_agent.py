import os
from pathlib import Path
from typing import List, Dict, Tuple, Literal
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain.schema import SystemMessage, HumanMessage



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o-mini")
CHROMA_DIR  = os.getenv("CHROMA_DIR",  "./chroma_db")


class RAGAgent:
    def __init__(
        self,
        pdf_filenames: List[str],
        embed_model: str = EMBED_MODEL,
        openai_api_key: str = OPENAI_API_KEY,
        db_path: str = CHROMA_DIR,
    ):
        self.pdf_filenames = pdf_filenames
        self.embeddings = OpenAIEmbeddings(model=embed_model, openai_api_key=openai_api_key)
        self.db_settings = Settings(persist_directory=db_path, anonymized_telemetry=False)
        self.db_path = db_path
        self._build_vector_store()

    def _init_vector_store(self):
        # Try to open an existing collection
        try:
            self.vectordb = Chroma(
                persist_directory=self.db_path,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                client_settings=self.db_settings,
            )
            # If it has docs, we’re done
            if self.vectordb._collection.count() > 0:
                return
        except Exception:
            pass
        # Otherwise build from PDFs once
        self._build_vector_store()


    def _build_vector_store(self):
        docs = []
        for filename in self.pdf_filenames:
            full_path = Path("docs") / filename
            if not full_path.exists():
                raise FileNotFoundError(f"Missing PDF: {full_path.resolve()}")
            loader = PyPDFLoader(str(full_path))
            for d in loader.load():
                # normalize metadata
                page = d.metadata.get("page")
                if isinstance(page, int):
                    d.metadata["page"] = page + 1  # 1-based
                d.metadata["source"] = os.path.basename(d.metadata.get("source", str(full_path)))
                docs.append(d)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
        chunks = splitter.split_documents(docs)
        for i, c in enumerate(chunks):
            c.metadata["chunk_id"] = i

        self.vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path,
            client_settings=self.db_settings,
        )

    def retrieve(self, query: str, k: int = 6) -> List[Dict]:
        retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": max(20, 3*k), "lambda_mult": 0.5},
        )
        res = retriever.invoke(query)
        return [{"content": r.page_content, "metadata": r.metadata} for r in res]

    def retrieve_group_balanced(
        self, query: str, groups: List[str], k_per: int = 3, fetch_k: int = 80
    ) -> List[Dict]:
        """
        Try to include chunks from each 'group' (entity) by matching tokens in the source filename.
        groups: e.g., ["FDA", "WHO"] or any capitalized terms the router extracted
        """
        retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": fetch_k, "fetch_k": fetch_k, "lambda_mult": 0.5},
        )
        results = retriever.invoke(query)
        picked = []
        tokens = [g.lower() for g in groups if g and g.strip()]
        for tok in tokens:
            grp = [r for r in results if tok in str(r.metadata.get("source", "")).lower()]
            picked.extend(grp[:k_per])
        if not picked:
            picked = results[:2 * k_per]
        return [{"content": r.page_content, "metadata": r.metadata} for r in picked]


class AnswerFormat(str, Enum):
    YES_NO = "yes_no"
    COMPARE = "compare"
    LIST_BY_ASPECTS = "list_by_aspects"
    LIST = "list"
    OPEN_QA = "open_qa"

class RouteDecision(BaseModel):
    format: AnswerFormat = Field(..., description="Chosen output format.")
    entities: List[str] = Field(default_factory=list, description="Entities/orgs to compare or focus on (e.g., ['FDA','WHO']).")
    aspects: List[str] = Field(default_factory=list, description="Named aspects if the user asked about multiple aspects (e.g., ['Verification','Validation']).")

class FormatRouter:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def decide(self, query: str) -> RouteDecision:
        system = SystemMessage(content=
            "You map a user's question to a generic answer format.\n"
            "Available formats:\n"
            "1) yes_no — for questions asking if something is true/allowed/considered.\n"
            "2) compare — for questions asking to compare/contrast entities; also extract entity names.\n"
            "3) list_by_aspects — for questions asking to list items by multiple named aspects (e.g., 'verification and validation'). Extract aspect names.\n"
            "4) list — for questions asking to list items but not tied to multiple named aspects (e.g., 'documentation needed').\n"
            "5) open_qa — fallback short paragraph if none of the above clearly applies.\n"
            "Return JSON with fields: format, entities (list), aspects (list).\n"
            "Entities/aspects must be short, capitalized where appropriate, and come from the question text when possible."
        )
        user = HumanMessage(content=f"Question:\n{query}\n\nReturn JSON now.")
        structured = self.llm.with_structured_output(RouteDecision)
        return structured.invoke([system, user])


class QA_YesNo(BaseModel):
    verdict: Literal["yes", "no", "insufficient"]
    justification: str
    used_indices: List[int]

class Section(BaseModel):
    title: str
    bullets: List[str]

class QA_Compare(BaseModel):
    similarities: List[str]
    differences: List[str]
    used_indices: List[int]

class QA_ListByAspects(BaseModel):
    sections: List[Section]
    used_indices: List[int]

class QA_List(BaseModel):
    items: List[str]
    used_indices: List[int]

class QA_OpenQA(BaseModel):
    answer: str
    used_indices: List[int]


def build_excerpts_block(contexts: List[Dict]) -> Tuple[str, List[Tuple[int, str, int, str]]]:
    # stable numbering: source, page, chunk_id
    contexts_sorted = sorted(
        contexts,
        key=lambda c: (
            c["metadata"].get("source", ""),
            c["metadata"].get("page", 10**9),
            c["metadata"].get("chunk_id", 10**9),
        ),
    )
    numbered: List[Tuple[int, str, int, str]] = []
    for i, c in enumerate(contexts_sorted, start=1):
        meta = c["metadata"]
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        numbered.append((i, src, page, c["content"]))
    block = "\n\n".join(
        f"[{i}] {src} | p. {page}\n{content}" for i, src, page, content in numbered
    )
    return block, numbered

def dedupe_and_validate_used(used_indices: List[int], k_max: int) -> List[int]:
    used: List[int] = []
    for idx in used_indices:
        if isinstance(idx, int) and 1 <= idx <= k_max and idx not in used:
            used.append(idx)
    return used

def references_block(used: List[int], numbered: List[Tuple[int, str, int, str]]) -> str:
    seen = set()
    lines: List[str] = []
    for i in used:
        _, src, page, _ = numbered[i - 1]
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {src} | p. {page}")
    return "\n".join(lines)

def groups_present(used: List[int], numbered: List[Tuple[int, str, int, str]], entities: List[str]) -> Dict[str, bool]:
    present = {}
    ents = [e.lower() for e in entities if e.strip()]
    for ent in ents:
        has = any(ent in numbered[i - 1][1].lower() for i in used)
        present[ent] = has
    return present

class ResponseAgent:
    def __init__(self, llm_model: str = CHAT_MODEL, temperature: float = 0.0, openai_api_key: str = OPENAI_API_KEY):
        self.llm = ChatOpenAI(model_name=llm_model, temperature=temperature, openai_api_key=openai_api_key)

    def _base_system(self) -> str:
        return (
            "You are a strict, citations-only analyst.\n"
            "Use ONLY the provided EXCERPTS.\n"
            "Do NOT include numeric citation markers in the prose.\n"
            "If excerpts are insufficient, return the schema with empty lists or 'insufficient' verdict as appropriate.\n"
            "Always fill used_indices with the 1-based EXCERPT indices you relied on.\n"
        )

    def _answer_yesno(self, query: str, excerpts_block: str, numbered: List[Tuple[int, str, int, str]]) -> str:
        structured = self.llm.with_structured_output(QA_YesNo)
        sys = SystemMessage(content=self._base_system() +
            "Task: Answer as Yes/No with one-paragraph justification. If not enough information, set verdict='insufficient'.")
        user = HumanMessage(content=(
            f"Question:\n{query}\n\nEXCERPTS:\n{excerpts_block}\n\n"
            "Return JSON: verdict('yes'|'no'|'insufficient'), justification, used_indices (1-based)."
        ))
        res: QA_YesNo = structured.invoke([sys, user])
        used = dedupe_and_validate_used(res.used_indices, len(numbered))
        if res.verdict == "insufficient" or not used:
            return "Insufficient information to answer based on the provided excerpts."
        refs = references_block(used, numbered)
        return f"{'Yes' if res.verdict=='yes' else 'No'}\n\n{res.justification.strip()}\n\nReferences:\n{refs}"

    def _answer_compare(self, query: str, excerpts_block: str, numbered: List[Tuple[int, str, int, str]], entities: List[str]) -> str:
        structured = self.llm.with_structured_output(QA_Compare)
        sys = SystemMessage(content=self._base_system() +
            "Task: Compare the named entities by listing bullet points for similarities and differences. Only use supported points.")
        user = HumanMessage(content=(
            f"Question:\n{query}\n\nEXCERPTS:\n{excerpts_block}\n\n"
            "Return JSON: similarities(list[str]), differences(list[str]), used_indices (1-based)."
        ))
        res: QA_Compare = structured.invoke([sys, user])
        used = dedupe_and_validate_used(res.used_indices, len(numbered))
        # if entities were provided (e.g., from router), require at least one source per entity
        if entities:
            presence = groups_present(used, numbered, entities)
            if not all(presence.values()) or not used:
                return "Insufficient information to produce a balanced comparison from the provided excerpts."
        if not used or (not res.similarities and not res.differences):
            return "Insufficient information to answer based on the provided excerpts."
        refs = references_block(used, numbered)
        out = []
        if res.similarities:
            out.append("**Similarities**")
            out += [f"- {s}" for s in res.similarities]
        if res.differences:
            out.append("\n**Differences**")
            out += [f"- {d}" for d in res.differences]
        return "\n".join(out) + f"\n\nReferences:\n{refs}"

    def _answer_list_by_aspects(self, query: str, excerpts_block: str, numbered: List[Tuple[int, str, int, str]], aspects: List[str]) -> str:
        structured = self.llm.with_structured_output(QA_ListByAspects)
        sys = SystemMessage(content=self._base_system() +
            "Task: Produce multiple sections, one per aspect, each with concise bullets. Only include items supported by the excerpts.")
        aspects_str = ", ".join(aspects) if aspects else "the aspects mentioned in the question"
        user = HumanMessage(content=(
            f"Question:\n{query}\n\nAspects to cover: {aspects_str}\n\n"
            f"EXCERPTS:\n{excerpts_block}\n\n"
            "Return JSON: sections(list of {title, bullets}), used_indices (1-based)."
        ))
        res: QA_ListByAspects = structured.invoke([sys, user])
        used = dedupe_and_validate_used(res.used_indices, len(numbered))
        if not used or not res.sections:
            return "Insufficient information to answer based on the provided excerpts."
        refs = references_block(used, numbered)
        lines = []
        for sec in res.sections:
            if sec.bullets:
                lines.append(f"**{sec.title}**")
                lines += [f"- {b}" for b in sec.bullets]
                lines.append("")  # blank line between sections
        return "\n".join(lines).strip() + f"\n\nReferences:\n{refs}"

    def _answer_list(self, query: str, excerpts_block: str, numbered: List[Tuple[int, str, int, str]]) -> str:
        structured = self.llm.with_structured_output(QA_List)
        sys = SystemMessage(content=self._base_system() +
            "Task: Produce a concise bullet list of items supported by the excerpts.")
        user = HumanMessage(content=(
            f"Question:\n{query}\n\nEXCERPTS:\n{excerpts_block}\n\n"
            "Return JSON: items(list[str]), used_indices (1-based)."
        ))
        res: QA_List = structured.invoke([sys, user])
        used = dedupe_and_validate_used(res.used_indices, len(numbered))
        if not used or not res.items:
            return "Insufficient information to answer based on the provided excerpts."
        refs = references_block(used, numbered)
        bullets = "\n".join(f"- {it}" for it in res.items)
        return f"{bullets}\n\nReferences:\n{refs}"

    def _answer_openqa(self, query: str, excerpts_block: str, numbered: List[Tuple[int, str, int, str]]) -> str:
        structured = self.llm.with_structured_output(QA_OpenQA)
        sys = SystemMessage(content=self._base_system() + "Task: Provide a short paragraph answer.")
        user = HumanMessage(content=(
            f"Question:\n{query}\n\nEXCERPTS:\n{excerpts_block}\n\n"
            "Return JSON: answer(string), used_indices (1-based)."
        ))
        res: QA_OpenQA = structured.invoke([sys, user])
        used = dedupe_and_validate_used(res.used_indices, len(numbered))
        if not used or not res.answer.strip():
            return "Insufficient information to answer based on the provided excerpts."
        refs = references_block(used, numbered)
        return f"{res.answer.strip()}\n\nReferences:\n{refs}"

    def answer(self, query: str, contexts: List[Dict], route: RouteDecision) -> str:
        if not contexts:
            return "Insufficient information to answer based on the provided excerpts."
        excerpts_block, numbered = build_excerpts_block(contexts)

        if route.format == AnswerFormat.YES_NO:
            return self._answer_yesno(query, excerpts_block, numbered)
        elif route.format == AnswerFormat.COMPARE:
            return self._answer_compare(query, excerpts_block, numbered, route.entities)
        elif route.format == AnswerFormat.LIST_BY_ASPECTS:
            return self._answer_list_by_aspects(query, excerpts_block, numbered, route.aspects)
        elif route.format == AnswerFormat.LIST:
            return self._answer_list(query, excerpts_block, numbered)
        else:
            return self._answer_openqa(query, excerpts_block, numbered)


class OrchestratorAgent:
    def __init__(self, rag: RAGAgent, responder: ResponseAgent, router: FormatRouter):
        self.rag = rag
        self.responder = responder
        self.router = router

    def handle(self, query: str) -> str:
        # 1) Decide output format, entities, aspects (generic)
        route: RouteDecision = self.router.decide(query)

        # 2) Retrieval strategy
        contexts: List[Dict]
        if route.format == AnswerFormat.COMPARE and route.entities:
            # try to supply balanced contexts per entity
            contexts = self.rag.retrieve_group_balanced(query, groups=route.entities, k_per=4)
        else:
            contexts = self.rag.retrieve(query, k=8 if route.format in (AnswerFormat.COMPARE, AnswerFormat.LIST_BY_ASPECTS) else 6)

        # 3) Answer in the chosen format
        return self.responder.answer(query, contexts, route)


def get_response(user_input: str) -> str:
    pdfs = [
        # Add or remove PDFs as needed. The filename becomes the "source" name shown to users.
        "FDA_Policy_Device_Software_Functions.pdf",
        "WHO_Medical_Device_Regulations.pdf",
        "FDA_Design_Control_Guidance.pdf",
    ]
    rag = RAGAgent(pdf_filenames=pdfs)
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.0, openai_api_key=OPENAI_API_KEY)
    router = FormatRouter(llm)
    responder = ResponseAgent()
    orchestrator = OrchestratorAgent(rag, responder, router)
    return orchestrator.handle(user_input)




if __name__ == "__main__":
    queries = [
        "Is our AI-powered MRI analysis tool considered a medical device software?",
        "What are the design control requirements for verification and validation?",
        "Compare FDA and WHO approaches to risk management for medical devices",
        "What documentation is needed for a mobile app that monitors heart rate?",
    ]
    for q in queries:
        print("Q:", q)
        print("A:", get_response(q))
        print("---\n")
