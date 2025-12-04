from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
chain = PromptTemplate.from_template("{question}") | model | StrOutputParser()

result = chain.invoke({"question": "What is the capital of India?"})
print(result)

