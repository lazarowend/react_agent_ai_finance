import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.agents.agent_toolkits import create_python_agent


load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("A chave GOOGLE_API_KEY não foi encontrada. Verifique o arquivo .env.")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1,
)

prompt = '''
    Como assistente financeiro pessoal, que responderá as perguntas dando dicas financeiras e de investimentos. Responda Tudo em português brasileiro
    Perguntas: {q}
'''
prompt_template = PromptTemplate.from_template(prompt)

python_repl = PythonREPL()
python_repl_tool = Tool(
    name='Python REPL',
    description='Um shell python. Use isso para executar códigos python. Execute apenas código Python seguros. Se você precisar obter o retorno do código, use a função "print()". Use para realizar cálculos financeiros necessários para responder as perguntas',
    func=python_repl
)

search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name='DuckDuckGo',
    description='Útil para encontra informações e dicas de enconomia e opções de investimento. Você sempre deve pesquisar na internet para obter as melhores dicas usando esta ferramenta, Responda diretamente. Sua resposta deve informar que há elementos pesquisados na internet',
    func=search
)
