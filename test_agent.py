from agno.agent import Agent
from agno.tools.tavily import TavilyTools
from agno.models.openai import OpenAIChat
from agno.tools.financial_datasets import 


web_agent = Agent(
    name="web_agent",
    role="搜索互联网获取信息",
    model=OpenAIChat(api_key='yourAIzaSyDmA_tI1pMgm-eOdsGIIoRtkORVQLISq2k', base_url='https://openai.hamuna.club/v1', id='gemini-1.5-flash'),
    instructions='Always include sources',
    show_tool_calls=True,
    markdown=True
)

