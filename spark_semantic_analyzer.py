# -*- coding=utf-8 -*-
# @Time: 2025/10/22 11:20
# @Author: 邱楠
# @File: spark_semantic_analyzer.py
# @Software: PyCharm

import json
import pandas as pd
import websocket
import json as json_module
import base64
import ssl
import hashlib
import hmac
from urllib.parse import urlencode, urlparse
from datetime import datetime
from time import mktime
from wsgiref.handlers import format_date_time
import time
import re


class SparkSemanticAnalyzer:
    def __init__(self, APPID, APIKey, APISecret):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Spark_url = "wss://spark-api.xf-yun.com/v1.1/chat"
        self.domain = "lite"
        self.message_history = []

    class Ws_Param(object):
        def __init__(self, APPID, APIKey, APISecret, gpt_url):
            self.APPID = APPID
            self.APIKey = APIKey
            self.APISecret = APISecret
            self.host = urlparse(gpt_url).netloc
            self.path = urlparse(gpt_url).path
            self.gpt_url = gpt_url

        def create_url(self):
            # 生成RFC1123格式的时间戳
            now = datetime.now()
            date = format_date_time(mktime(now.timetuple()))

            # 拼接字符串
            signature_origin = "host: " + self.host + "\n"
            signature_origin += "date: " + date + "\n"
            signature_origin += "GET " + self.path + " HTTP/1.1"

            # 进行hmac-sha256进行加密
            signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                     digestmod=hashlib.sha256).digest()

            signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

            authorization_origin = f'api_key="{self.APIKey}", ' \
                                   f'algorithm="hmac-sha256", ' \
                                   f'headers="host date request-line", ' \
                                   f'signature="{signature_sha_base64}"'

            authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

            # 将请求的鉴权参数组合为字典
            v = {
                "authorization": authorization,
                "date": date,
                "host": self.host
            }
            # 拼接鉴权参数，生成url
            url = self.gpt_url + '?' + urlencode(v)
            return url

    def optimize_question(self, original_question: str, max_retries=3) -> str:
        """
        优化用户问题，使其更完整和专业

        Args:
            original_question: 原始用户问题
            max_retries: 最大重试次数

        Returns:
            优化后的问题
        """
        prompt = self._build_optimization_prompt(original_question)

        for attempt in range(max_retries):
            try:
                result = self._send_request(prompt)
                if result:
                    optimized_question = self._parse_response(result, original_question)
                    if optimized_question and optimized_question != original_question:
                        print(f"🎯 语义分析优化: '{original_question}' -> '{optimized_question}'")
                        return optimized_question

                time.sleep(1)  # 重试前等待

            except Exception as e:
                print(f"❌ 第{attempt + 1}次语义分析尝试失败: {e}")
                time.sleep(1)

        # 如果所有尝试都失败，返回原问题
        print("⚠️  语义分析失败，使用原始问题")
        return original_question

    def _build_optimization_prompt(self, question: str) -> str:
        """构建优化提示词"""
        prompt = f"""
        你是一个专业的汽车技术文档问答助手业。请将用户的问题优化为更完整、专业、便于在技术文档数据库中检索的形式。
        
        优化要求：
        1. 保持原意不变，不要添加不存在的信息
        2. 补充必要的技术术语，使其更专业
        3. 使其更加明确和具体，避免模糊表述
        4. 如果问题不完整，基于汽车技术文档的常见问题模式进行补充
        5. 输出只需要优化后的问题，不要任何解释
        
        优化示例：
        - 输入："油耗怎么样" → 输出："车辆的油耗表现和节能技术特点是什么？"
        - 输入："怎么开这个功能" → 输出："如何开启和使用智能驾驶辅助系统？"
        - 输入："坏了怎么办" → 输出："车辆出现故障时的诊断和解决方法是什么？"
        
        现在请优化这个用户问题：
        "{question}"
        
        优化后的问题：
        """
        return prompt

    def _parse_response(self, response: str, original_question: str) -> str:
        """解析大模型响应"""
        if not response:
            return original_question

        # 清理响应文本
        response = response.strip()

        # 移除可能的引号
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        elif response.startswith("'") and response.endswith("'"):
            response = response[1:-1]

        # 检查响应是否有效
        if len(response) < 3 or response == original_question:
            return original_question

        return response

    def _send_request(self, prompt):
        """发送请求到星火API并获取响应"""
        response_content = []
        wsParam = self.Ws_Param(self.APPID, self.APIKey, self.APISecret, self.Spark_url)

        def on_message(ws, message):
            data = json.loads(message)
            if data['header']['code'] != 0:
                # print(f'请求错误: {data}')
                ws.close()
            else:
                content = data["payload"]["choices"]["text"][0]["content"]
                response_content.append(content)
                if data["payload"]["choices"]["status"] == 2:
                    ws.close()

        def on_error(ws, error):
            # print("### error:", error)
            return

        def on_close(ws, close_status_code, close_msg):  # 修改这里接受3个参数
            # print(f"### closed ### 状态码: {close_status_code}, 消息: {close_msg}")
            return

        def on_open(ws):
            data = json.dumps({
                "header": {"app_id": self.APPID},
                "parameter": {
                    "chat": {
                        "domain": self.domain,
                        "temperature": 0.7,
                        "max_tokens": 2048
                    }
                },
                "payload": {
                    "message": {
                        "text": [{"role": "user", "content": prompt}]
                    }
                }
            })
            ws.send(data)

        ws = websocket.WebSocketApp(
            wsParam.create_url(),
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        return ''.join(response_content)