from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment
load_dotenv()


# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY missing in .env")

print("📄 Loading PDF...")
# 1. Load + split PDF
docs = PyPDFLoader("islr.pdf").load()
splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
print(f"✅ Split into {len(splits)} chunks")

print("🔗 Creating embeddings...")
# 2. Embed + retrieve
vectorstore = FAISS.from_documents(
    splits, 
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
)
retriever = vectorstore.as_retriever()

print("🤖 Setting up LLM...")
# 3. LLM + RAG chain
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(
    "Answer from context: {context}\n\nQuestion: {question}"
)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG ready! Ask questions about islr.pdf")
print("-" * 50)

# 4. Chat loop
while True:
    try:
        q = input("\nQ: ").strip()
        if not q:
            print("Enter question or Ctrl+C to exit")
            continue
        if q.lower() in ['exit', 'quit']:
            break
            
        result = chain.invoke(q)
        print("A:", result)
        
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        break
