from ai.llm.chat import LLMChatAgent
import yaml

if __name__ == '__main__':
    with open('config/agents/stock_analytic.yml', 'r') as f:
        config = yaml.safe_load(f)
    agent = LLMChatAgent(config)
    resp = agent.chat('分析下华能信托')
    print(resp.content)
