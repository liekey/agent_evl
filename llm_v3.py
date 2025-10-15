import openai
import os
from typing import List, Dict, Optional, Callable, Any
import json
from datetime import datetime
import subprocess
import webbrowser
import pydoc
import re

class Tool:
    """工具类，用于定义可由AI调用的工具"""
    
    def __init__(self, name: str, description: str, function: Callable, parameters: Dict[str, Any]):
        """
        初始化工具
        
        Args:
            name: 工具名称
            description: 工具描述
            function: 工具对应的Python函数
            parameters: 工具参数的JSON Schema定义
        """
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters
    
    def to_dict(self) -> Dict[str, Any]:
        """将工具转换为OpenAI API格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

# 工具函数定义
def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 安全地计算表达式
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"

def search_files(pattern: str, directory: str = ".") -> str:
    """在指定目录中搜索文件"""
    try:
        # Windows下使用dir命令
        result = subprocess.run(
            ["dir", "/s", "/b", pattern], 
            capture_output=True, 
            text=True,
            shell=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"搜索错误: {result.stderr}"
    except Exception as e:
        return f"搜索失败: {str(e)}"

def open_url(url: str) -> str:
    """打开URL"""
    try:
        webbrowser.open(url)
        return f"已打开URL: {url}"
    except Exception as e:
        return f"无法打开URL: {str(e)}"

def get_help(topic: str = "") -> str:
    """获取Python帮助信息"""
    try:
        if topic:
            help_text = pydoc.render_doc(topic, renderer=pydoc.text)
            return help_text[:1000] + "..." if len(help_text) > 1000 else help_text
        else:
            return "请指定要查询的主题，例如: help('list')"
    except Exception as e:
        return f"获取帮助信息失败: {str(e)}"

""" 
简单的工具调用测试，明天用deepseek的api做测试
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1:8b",
    "messages": [
      {"role": "user", "content": "帮我调用get_weather函数获取北京天气"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
  }'] } } } "required": ["city"]string", "description": "城市名称"}
"""

class OllamaChatWithTools:
    """使用OpenAI客户端与Ollama模型交互的聊天类，支持工具调用"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434/v1",
        model: str = "deepseek-r1:8b",
        system_prompt: str = """你是一个友好的助手，可以使用工具来帮助用户。

当你需要使用工具时，请使用以下JSON格式：
{"tool": "工具名称", "arguments": {"参数名": "参数值"}}

例如：
{"tool": "get_current_time", "arguments": {}}
{"tool": "calculate", "arguments": {"expression": "2 + 3"}}

请确保在回复中包含完整的JSON格式工具调用，然后等待用户确认后再执行工具。
""",
        max_history: int = 20,
        enable_tools: bool = True,
        auto_execute: bool = True  # 是否自动执行工具
    ):
        """
        初始化Ollama聊天客户端
        
        Args:
            base_url: Ollama API的基础URL
            model: 使用的模型名称
            system_prompt: 系统提示词
            max_history: 最大历史消息数量
            enable_tools: 是否启用工具调用功能
            auto_execute: 是否自动执行工具
        """
        self.client = openai.OpenAI(
            base_url=base_url,
            # api_key="ollama"  # Ollama不需要真实的API key
        )
        self.model = model
        self.max_history = max_history
        self.enable_tools = enable_tools
        self.auto_execute = auto_execute
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 初始化工具
        self.tools = {}
        if enable_tools:
            self._initialize_tools()
    
    def _initialize_tools(self):
        """初始化可用工具"""
        tools = [
            Tool(
                name="get_current_time",
                description="获取当前日期和时间",
                function=get_current_time,
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            Tool(
                name="calculate",
                description="计算数学表达式",
                function=calculate,
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "要计算的数学表达式，例如: 2 + 3 * 4"
                        }
                    },
                    "required": ["expression"]
                }
            ),
            Tool(
                name="search_files",
                description="在指定目录中搜索文件",
                function=search_files,
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "文件名模式，例如: *.py"
                        },
                        "directory": {
                            "type": "string",
                            "description": "要搜索的目录，默认为当前目录",
                            "default": "."
                        }
                    },
                    "required": ["pattern"]
                }
            ),
            Tool(
                name="open_url",
                description="在浏览器中打开URL",
                function=open_url,
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "要打开的URL"
                        }
                    },
                    "required": ["url"]
                }
            ),
            Tool(
                name="get_help",
                description="获取Python帮助信息",
                function=get_help,
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "要查询的主题，例如: list, dict",
                            "default": ""
                        }
                    },
                    "required": []
                }
            )
        ]
        
        for tool in tools:
            self.tools[tool.name] = tool
    
    def add_message(self, role: str, content: str, tool_calls: List[Dict] = None, tool_call_id: str = None) -> None:
        """添加消息到历史记录"""
        message = {"role": role, "content": content}
        
        if tool_calls:
            message["tool_calls"] = tool_calls
        
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
            
        self.messages.append(message)
        
        # 保持历史记录在合理范围内
        if len(self.messages) > self.max_history:
            # 保留系统消息和最近的消息
            self.messages = [self.messages[0]] + self.messages[-(self.max_history-1):]
    
    def _execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行工具调用"""
        if tool_name not in self.tools:
            return f"错误: 未知工具 '{tool_name}'"
        
        try:
            tool = self.tools[tool_name]
            result = tool.function(**arguments)
            return result
        except Exception as e:
            return f"执行工具 '{tool_name}' 时出错: {str(e)}"
    
    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """从文本中解析工具调用"""
        tool_calls = []
        
        # 尝试匹配JSON格式的工具调用
        json_pattern = r'\{\s*"tool":\s*"([^"]+)"\s*,\s*"arguments":\s*\{([^}]+)\}\s*\}'
        matches = re.findall(json_pattern, text)
        
        for tool_name, args_str in matches:
            try:
                # 尝试解析参数
                args = {}
                for pair in args_str.split(','):
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        key = key.strip().strip('"')
                        value = value.strip().strip('"')
                        args[key] = value
                
                tool_calls.append({
                    "tool": tool_name,
                    "arguments": args
                })
            except Exception:
                continue
        
        return tool_calls
    
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
            
            # 准备API请求参数
            api_params = {
                "model": self.model,
                "messages": self.messages,
                "stream": stream,
                "temperature": 0.7,
                "max_tokens": 2048
            }
            
            # 调用API
            response = self.client.chat.completions.create(**api_params)
            
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
                
                # 添加助手回复到历史记录
                self.add_message("assistant", assistant_reply)
            else:
                # 获取助手回复
                message = response.choices[0].message
                assistant_reply = message.content or ""
                print(f"🤖: {assistant_reply}\n")
                
                # 添加助手回复到历史记录
                self.add_message("assistant", assistant_reply)
                
                # 如果启用工具，尝试从文本中解析工具调用
                if self.enable_tools:
                    tool_calls = self._parse_tool_calls(assistant_reply)
                    
                    if tool_calls:
                        print("🔧 检测到工具调用:")
                        for tool_call in tool_calls:
                            tool_name = tool_call["tool"]
                            arguments = tool_call["arguments"]
                            print(f"  - 工具: {tool_name}")
                            print(f"  - 参数: {arguments}")
                            
                            # 执行工具
                            if self.auto_execute:
                                result = self._execute_tool_call(tool_name, arguments)
                                print(f"  - 结果: {result}")
                                
                                # 添加工具结果到历史记录
                                self.add_message("system", f"工具 {tool_name} 的执行结果: {result}")
                                
                                # 再次调用API，获取包含工具结果的回复
                                second_response = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=self.messages,
                                    temperature=0.7,
                                    max_tokens=2048
                                )
                                
                                second_reply = second_response.choices[0].message.content or ""
                                print(f"🤖: {second_reply}\n")
                                
                                # 添加最终回复到历史记录
                                self.add_message("assistant", second_reply)
                                
                                return second_reply
                            else:
                                print("  - 等待用户确认执行...")
            
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
            role_emoji = "🤖" if msg["role"] == "assistant" else "👤" if msg["role"] == "user" else "⚙️" if msg["role"] == "system" else "🔧"
            content = msg.get("content", "")
            if msg["role"] == "tool":
                content = f"[工具结果] {content}"
            print(f"{role_emoji} {msg['role']}: {content}")
        print("================\n")
    
    def list_tools(self) -> None:
        """列出所有可用工具"""
        if not self.enable_tools:
            print("❌ 工具调用功能未启用")
            return
            
        print("\n=== 可用工具 ===")
        for tool_name, tool in self.tools.items():
            print(f"🔧 {tool_name}: {tool.description}")
        print("================\n")
    
    def toggle_tools(self) -> None:
        """切换工具调用功能"""
        self.enable_tools = not self.enable_tools
        status = "启用" if self.enable_tools else "禁用"
        print(f"✅ 工具调用功能已{status}")
    
    def toggle_auto_execute(self) -> None:
        """切换自动执行工具功能"""
        self.auto_execute = not self.auto_execute
        status = "启用" if self.auto_execute else "禁用"
        print(f"✅ 自动执行工具功能已{status}")
    
    def execute_pending_tools(self) -> None:
        """执行待处理的工具调用"""
        if not self.messages:
            print("❌ 没有待处理的工具调用")
            return
        
        # 获取最后一条助手消息
        last_message = self.messages[-1]
        if last_message["role"] != "assistant":
            print("❌ 没有待处理的工具调用")
            return
        
        # 解析工具调用
        tool_calls = self._parse_tool_calls(last_message["content"])
        
        if not tool_calls:
            print("❌ 没有检测到工具调用")
            return
        
        print("🔧 执行待处理的工具调用:")
        for tool_call in tool_calls:
            tool_name = tool_call["tool"]
            arguments = tool_call["arguments"]
            print(f"  - 工具: {tool_name}")
            print(f"  - 参数: {arguments}")
            
            # 执行工具
            result = self._execute_tool_call(tool_name, arguments)
            print(f"  - 结果: {result}")
            
            # 添加工具结果到历史记录
            self.add_message("system", f"工具 {tool_name} 的执行结果: {result}")
        
        # 再次调用API，获取包含工具结果的回复
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=0.7,
                max_tokens=2048
            )
            
            assistant_reply = response.choices[0].message.content or ""
            print(f"🤖: {assistant_reply}\n")
            
            # 添加最终回复到历史记录
            self.add_message("assistant", assistant_reply)
        except Exception as e:
            print(f"❌ 获取回复失败: {e}")

def main():
    """主函数 - 交互式聊天界面"""
    print("=== Ollama Chat with Tools - OpenAI Client Version ===")
    print("输入内容开始对话，输入 'exit' 退出，输入 'save' 保存对话，输入 'clear' 清除历史，输入 'history' 查看历史")
    print("输入 'tools' 查看可用工具，输入 'toggle' 切换工具调用功能，输入 'auto' 切换自动执行工具")
    print("输入 'execute' 执行待处理的工具调用\n")
    
    # 创建聊天实例
    chatbot = OllamaChatWithTools(
        base_url="http://localhost:11434/v1",
        model="deepseek-r1:8b",
        enable_tools=True,
        auto_execute=True  # 默认自动执行工具
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
            elif user_input.lower() == 'tools':
                chatbot.list_tools()
                continue
            elif user_input.lower() == 'toggle':
                chatbot.toggle_tools()
                continue
            elif user_input.lower() == 'auto':
                chatbot.toggle_auto_execute()
                continue
            elif user_input.lower() == 'execute':
                chatbot.execute_pending_tools()
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