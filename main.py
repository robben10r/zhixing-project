import os
import redis
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.redis import RedisSaver

load_dotenv()

# 确保读取到了环境变量，否则后面会报错
if not os.getenv("DB_URI"):
    raise ValueError("请在 .env 文件中配置 DB_URI")

model = ChatOpenAI(
    model=os.getenv("QWEN_MODEL_NLP", "qwen-max"),
    base_url=os.getenv("QWEN_BASE_URL"),
    api_key=os.getenv("QWEN_API_KEY"),
    temperature=0.3,
)


# ================= 1. 定义 Graph =================
class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    return {"messages": [model.invoke(state.get("messages"))]}


builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)


# ================= 2. Redis 配置与运行 =================
# 建立连接
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
    db=0,
    password=os.getenv("REDIS_PASSWORD"), # 如果有密码请取消注释
    #decode_responses=False # 注意：LangGraph 序列化通常处理二进制，这里建议保持 False 或根据具体库版本调整
)

# 初始化 Checkpointer
checkpointer = RedisSaver(redis_client=redis_client)

# 编译
graph = builder.compile(checkpointer=checkpointer)

# ================= 3. 执行测试 =================
config = {"configurable": {"thread_id": "redis_user_999"}}

print(f"--- Redis 对话 ---")
result = graph.invoke({"messages": [("user", "Redis 准备好了吗？")]}, config)
print(f"Bot: {result['messages'][-1].content}")

# 验证 Key 是否存在
print(f"Key 数量: {len(redis_client.keys('checkpoint:*'))}")