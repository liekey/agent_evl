import openai
import os
from typing import List, Dict, Optional
import json
from datetime import datetime

class OllamaChat:
    """使用OpenAI客户端与Ollama模型交互的聊天类"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434/v1",
        model: str = "deepseek-r1:8b",
        system_prompt: str = "你是一个友好的助手。",
        max_history: int = 20
    ):
        """
        初始化Ollama聊天客户端
        
        Args:
            base_url: Ollama API的基础URL
            model: 使用的模型名称
            system_prompt: 系统提示词
            max_history: 最大历史消息数量
        """
        self.client = openai.OpenAI(
            base_url=base_url,
            # api_key="ollama"  # Ollama不需要真实的API key
        )
        self.model = model
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        
    def add_message(self, role: str, content: str) -> None:
        """添加消息到历史记录"""
        self.messages.append({"role": role, "content": content})
        
        # 保持历史记录在合理范围内
        if len(self.messages) > self.max_history:
            # 保留系统消息和最近的消息
            self.messages = [self.messages[0]] + self.messages[-(self.max_history-1):]
    
    def chat(self, prompt: str, stream: bool = False) -> Optional[str]:
        """
        发送消息给Ollama模型并返回回复
        
        Args:
            prompt: 用户输入的提示词
            stream: 是否使用流式响应
            
        Returns:
            模型的回复内容，如果出错则返回None
        """
        try:
            # 添加用户消息
            self.add_message("user", prompt)
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=stream,
                temperature=0.7,
                max_tokens=2048
            )
            
            if stream:
                # 流式响应处理
                full_response = ""
                print("🤖: ", end="", flush=True)
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_response += content
                print("\n")
                assistant_reply = full_response
            else:
                # 非流式响应
                assistant_reply = response.choices[0].message.content
                print(f"🤖: {assistant_reply}\n")
            
            # 添加助手回复到历史记录
            self.add_message("assistant", assistant_reply)
            return assistant_reply
            
        except openai.APIError as e:
            print(f"❌ API错误: {e}")
            return None
        except Exception as e:
            print(f"❌ 出错: {e}")
            return None
    
    def save_conversation(self, filename: str = None) -> None:
        """保存对话历史到文件"""
        if filename is None:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
            print(f"💾 对话已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存对话失败: {e}")
    
    def clear_history(self) -> None:
        """清除对话历史（保留系统消息）"""
        system_message = self.messages[0] if self.messages else {"role": "system", "content": "你是一个友好的助手。"}
        self.messages = [system_message]
        print("🗑️  对话历史已清除")
    
    def show_history(self) -> None:
        """显示对话历史"""
        print("\n=== 对话历史 ===")
        for msg in self.messages:
            role_emoji = "🤖" if msg["role"] == "assistant" else "👤" if msg["role"] == "user" else "⚙️"
            print(f"{role_emoji} {msg['role']}: {msg['content']}")
        print("================\n")

def main():
    """主函数 - 交互式聊天界面"""
    print("=== Ollama Chat - OpenAI Client Version ===")
    print("输入内容开始对话，输入 'exit' 退出，输入 'save' 保存对话，输入 'clear' 清除历史，输入 'history' 查看历史\n")
    
    # 创建聊天实例
    chatbot = OllamaChat(
        base_url="http://localhost:11434/v1",
        model="deepseek-r1:8b"
    )
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == 'exit':
                print("再见！")
                break
            elif user_input.lower() == 'save':
                chatbot.save_conversation()
                continue
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                continue
            elif user_input.lower() == 'history':
                chatbot.show_history()
                continue
            
            # 正常对话
            chatbot.chat(user_input)
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"❌ 出错: {e}")

if __name__ == "__main__":
    main()