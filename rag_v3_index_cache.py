import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "rag-chatbot"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PDF_PATH = "islr.pdf"
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)

# ---------- Simple Helpers ----------
@traceable(name="load_pdf")
def load_pdf(path): 
    return PyPDFLoader(path).load()

@traceable(name="split_docs")
def split_docs(docs):
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)

@traceable(name="build_vs")
def build_vs(splits):
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    return FAISS.from_documents(splits, emb)

# ---------- SIMPLE Cache (3 lines) ----------
def get_cache_key(pdf_path):
    # Hash PDF content + settings
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()[:16]  # Short unique ID

@traceable(name="cache_check")
def load_or_build(pdf_path):
    key = get_cache_key(pdf_path)
    cache_dir = INDEX_ROOT / key
    
    if cache_dir.exists():
        print(f"✅ CACHE HIT: {cache_dir}")
        emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        return FAISS.load_local(str(cache_dir), emb, allow_dangerous_deserialization=True)
    else:
        print(f"🔨 BUILDING cache: {cache_dir}")
        docs = load_pdf(pdf_path)
        splits = split_docs(docs)
        vs = build_vs(splits)
        cache_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(cache_dir))
        return vs

# ---------- RAG Pipeline ----------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from context. Say 'don't know' if not found."),
    ("human", "Question: {question}\n\nContext: {context}")
])

def format_docs(docs): 
    return "\n\n".join(d.page_content for d in docs)

@traceable(name="rag_query")
def query_rag(question):
    # Get cached vectorstore → RAG chain
    vectorstore = load_or_build(PDF_PATH)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }) | prompt | llm | StrOutputParser()
    )
    return chain.invoke(question)

# ---------- Run ----------
if __name__ == "__main__":
    print("🚀 RAG v3 (Gemini + Cache) ready!")
    while True:
        try:
            q = input("\nQ: ").strip()
            if q.lower() in ['exit', 'quit']: break
            if not q: continue
            ans = query_rag(q)
            print("\nA:", ans)
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
