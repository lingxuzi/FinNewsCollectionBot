import openai
import os
from tavily import TavilyClient
import asyncio

class LLMChatManager:
    def __init__(self, api_key=None, base_url=None, model="gpt-4-1106-preview", tavily_api_key=None, custom_api_path=None):
        """
        初始化 LLM 对话管理类。

        :param api_key: OpenAI 或兼容服务的 API 密钥。如果未提供，会尝试从环境变量 OPENAI_API_KEY 获取。
        :param base_url: 自定义 API 基础 URL，用于兼容非 OpenAI 服务。
        :param model: 要使用的 LLM 模型，默认为 gpt-4-1106-preview。
        :param tavily_api_key: Tavily 的 API 密钥。如果未提供，会尝试从环境变量 TAVILY_API_KEY 获取。
        :param custom_api_path: 自定义 API 路径，用于扩展 API 调用。
        """
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Please provide it or set the OPENAI_API_KEY environment variable.")
        openai.api_key = self.api_key
        if base_url:
            openai.api_base = base_url
        self.model = model
        self.conversation_history = []
        self.tavily_api_key = tavily_api_key if tavily_api_key else os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError("Tavily API key is required. Please provide it or set the TAVILY_API_KEY environment variable.")
        self.custom_api_path = custom_api_path
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        # 定义函数调用描述
        self.functions = [
            {
                "name": "tavily_search",
                "description": "使用 Tavily 进行网页搜索",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "需要在网页上搜索的查询内容"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

    def add_user_message(self, message):
        """
        向对话历史中添加用户消息。

        :param message: 用户输入的消息内容。
        """
        self.conversation_history.append({"role": "user", "content": message})

    def tavily_search(self, query):
        """
        使用 Tavily 进行网页搜索。

        :param query: 需要搜索的查询内容。
        :return: 搜索结果的摘要。
        """
        try:
            search_results = self.tavily_client.search(query=query, search_depth="advanced", max_results=3)
            return "\n".join([result["content"] for result in search_results["results"]])
        except Exception as e:
            print(f"An error occurred during Tavily search: {e}")
            return "搜索过程中出现错误，请稍后重试。"

    async def get_llm_response(self, temperature=0.7, max_tokens=500, stream=False):
        """
        异步调用 LLM 获取回复，并将回复添加到对话历史中。

        :param temperature: 模型的温度参数，控制输出的随机性，取值范围 0 到 2。
        :param max_tokens: 生成回复的最大令牌数。
        :param stream: 是否开启流式输出，默认为 False。
        :return: LLM 生成的回复内容或流式生成器。
        """
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens,
                functions=self.functions,
                function_call="auto",
                stream=stream
            )
            if stream:
                async def stream_generator():
                    full_content = ""
                    async for chunk in response:
                        delta = chunk['choices'][0]['delta']
                        if "function_call" in delta:
                            function_name = delta["function_call"]["name"]
                            function_args = eval(delta["function_call"]["parameters"])
                            if function_name == "tavily_search":
                                function_response = self.tavily_search(
                                    query=function_args.get("query")
                                )
                                self.conversation_history.append({"role": "function", "name": function_name, "content": function_response})
                                async for inner_chunk in await self.get_llm_response(temperature=temperature, max_tokens=max_tokens, stream=stream):
                                    yield inner_chunk
                        elif "content" in delta:
                            content = delta["content"]
                            full_content += content
                            yield content
                    self.conversation_history.append({"role": "assistant", "content": full_content})
                return stream_generator()
            else:
                response_message = response['choices'][0]['message']
                if response_message.get("function_call"):
                    function_name = response_message["function_call"]["name"]
                    function_args = eval(response_message["function_call"]["parameters"])
                    if function_name == "tavily_search":
                        function_response = self.tavily_search(
                            query=function_args.get("query")
                        )
                        self.conversation_history.append(response_message)
                        self.conversation_history.append(
                            {
                                "role": "function",
                                "name": function_name,
                                "content": function_response
                            }
                        )
                        return await self.get_llm_response(temperature=temperature, max_tokens=max_tokens, stream=stream)
                assistant_message = response_message['content']
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                return assistant_message
        except Exception as e:
            print(f"An error occurred while getting LLM response: {e}")
            return None

    def clear_conversation(self):
        """
        清空对话历史。
        """
        self.conversation_history = []

    def get_conversation_history(self):
        """
        获取当前的对话历史。

        :return: 包含对话历史的列表。
        """
        return self.conversation_history

# Generate LLM Chat Manager
