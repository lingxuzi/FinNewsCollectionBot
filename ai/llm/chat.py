from agno.agent import Agent
from agno.models.openai import OpenAILike, OpenAIChat
from agno.tools.tavily import TavilyTools
from agno.tools.duckduckgo import DuckDuckGoTools

from ai.agents.tools.financial_statement import FinancialStatementToolKit


class LLMChatAgent:
    def __init__(self, config):
        self.config = config
        tools = self.__build_tools(config['tools'])
        self._agent = Agent(
            name=config['name'],
            model= OpenAILike(config['llm']['model'],  api_key=config['llm']['api_key'], base_url=config['llm']['base_url'], temperature=config['llm']['temperature'], max_tokens=config['llm']['max_tokens'], timeout=config['llm']['timeout']),
            tools=tools,
            system_message_role=config['system_message_role'],
            markdown=config['markdown'],
            description=config['description'],
            instructions=config['instructions'],
            goal=config['goal'],
            show_tool_calls=True
        )

    def __build_tools(self, config):
        tools = []
        for tool, tool_config in config.items():
            if tool == 'duckduck':
                tools.append(DuckDuckGoTools(verify_ssl=False))
            elif tool == 'tavily':
                tools.append(TavilyTools(tool_config['api_key']))
            elif tool == 'financial':
                tools.append(FinancialStatementToolKit(tool_config))
        return tools
    
    def chat(self, message):
        return self._agent.run(message, stream=self.config['stream'])
        