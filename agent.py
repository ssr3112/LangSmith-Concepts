from dotenv import load_dotenv
import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "final-agent"

@tool
def get_weather(city: str) -> str:
    """Get current weather"""
    if not WEATHERSTACK_API_KEY:
        return f"Weather tool needs API key for {city}"
    url = f'https://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={city}'
    try:
        data = requests.get(url).json()
        temp = data['current']['temperature']
        return f"{temp}°C in {city}"
    except:
        return f"No weather data for {city}"

# ONLY 1 TOOL - Weather
tools = [get_weather]

# Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)

# LangGraph agent (NO complex imports!)
agent = create_react_agent(llm, tools)

print("🌤️ Weather Agent ready!")
print("Try: 'Gurgaon temperature?', 'Delhi weather?'")
while True:
    q = input("\nAsk: ").strip()
    if q.lower() in ['exit', 'quit']: break
    try:
        result = agent.invoke({"messages": [("human", q)]})
        print("\n✅", result['messages'][-1].content)
    except Exception as e:
        print(f"❌ {e}")
