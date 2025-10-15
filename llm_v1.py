import requests

# Ollama本地API
API_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "deepseek-r1:8b"

# 初始化消息上下文
messages = [
    {"role": "system", "content": "你是一个友好的助手。"}
]

def chat(prompt: str):
    """发送消息给Ollama模型并返回回复"""
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False
    }

    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    data = response.json()

    # 从返回中取出assistant的内容
    reply = data["choices"][0]["message"]["content"]
    print(f"🤖: {reply}\n")

    # 保存上下文
    messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    print("=== Ollama Chat - deepseek-r1:8b ===")
    print("输入内容开始对话，输入 exit 退出。\n")

    while True:
        user_input = input("你: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("再见！")
            break
        try:
            chat(user_input)
        except Exception as e:
            print("❌ 出错:", e)
