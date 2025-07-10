from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.tavily import TavilyTools


class LLMChatAgent:
    def __init__(self, config):
        tools = self.__build_tools(config['tools'])
        self._agent = Agent(
            name=config['name'],
            model= OpenAIChat(config['llm']['model_name'], api_key=config['llm']['api_key'], base_url=config['llm']['base_url'], temperature=config['llm']['temperature'], max_tokens=config['llm']['max_tokens'], timeout=config['llm']['timeout']),
            tools=tools,
            markdown=config['markdown'],
            description=config['description'],
            instructions=config['instructions']
        )

    def __build_tools(self, config):
        tools = []
        for tool, tool_config in config.items():
            if tool == 'tavily':
                tools.append(TavilyTools(tool_config['api_key']))
        return tools
    
    def chat(self, message):
        return self._agent.chat(message)
        