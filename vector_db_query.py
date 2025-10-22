# -*- coding=utf-8 -*-
# @Time: 2025/10/21 16:53
# @Author: 邱楠
# @File: vector_db_query.py
# @Software: PyCharm


import os
from excel_to_vector_db import ExcelToVectorDB
from spark_semantic_analyzer import SparkSemanticAnalyzer
import argparse
from typing import List, Dict

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class VectorDBQuery:
    def __init__(self, db_path="./vector_db", model_name='BAAI/bge-small-zh',
                 collection_name="qa_embeddings_bge", use_semantic_analysis=True,
                 spark_appid="17fd554e", spark_apikey="f6e09b7ea35d556c60c16bc8b06822ae",
                 spark_apisecret="YzQwOWQ4M2U3NzM2ODYzYzE3ODI0M2M0"):
        """
        初始化向量数据库查询系统

        Args:
            db_path: 数据库路径
            model_name: 模型名称
            collection_name: 要查询的集合名称
            use_semantic_analysis: 是否启用语义分析
            spark_appid: 星火APPID
            spark_apikey: 星火APIKey
            spark_apisecret: 星火APISecret
        """
        print("正在加载向量数据库查询系统...")
        self.vector_db = ExcelToVectorDB(
            model_name=model_name,
            db_path=db_path,
            collection_name=collection_name
        )

        # 初始化语义分析器
        self.use_semantic_analysis = use_semantic_analysis
        self.semantic_analyzer = None

        if use_semantic_analysis and spark_appid and spark_apikey and spark_apisecret:
            try:
                self.semantic_analyzer = SparkSemanticAnalyzer(
                    APPID=spark_appid,
                    APIKey=spark_apikey,
                    APISecret=spark_apisecret
                )
                print("✓ 语义分析器初始化完成!")
            except Exception as e:
                print(f"❌ 语义分析器初始化失败: {e}")
                self.use_semantic_analysis = False
        else:
            if use_semantic_analysis:
                print("⚠️  语义分析已启用但缺少API密钥，将禁用该功能")
                self.use_semantic_analysis = False

        print(f"✓ 查询系统初始化完成! 目标集合: {collection_name}")
        print(f"🔧 语义分析功能: {'已启用' if self.use_semantic_analysis else '已禁用'}")

    def optimize_question_with_semantic(self, question: str) -> str:
        """使用语义分析优化问题"""
        if not self.use_semantic_analysis or not self.semantic_analyzer:
            return question

        try:
            return self.semantic_analyzer.optimize_question(question)
        except Exception as e:
            print(f"❌ 语义分析出错: {e}")
            return question

    def list_available_collections(self):
        """列出所有可用的集合"""
        return self.vector_db.list_collections()

    def query_question(self, question: str, top_k: int = 5, min_score: float = 0.5,
                       use_semantic: bool = None) -> List[Dict]:
        """
        查询相似问题

        Args:
            question: 用户问题
            top_k: 返回最相似的前K个结果
            min_score: 最小相似度阈值
            use_semantic: 是否使用语义分析（None时使用默认设置）

        Returns:
            相似问题列表
        """
        # 确定是否使用语义分析
        if use_semantic is None:
            use_semantic = self.use_semantic_analysis

        original_question = question

        # 语义分析优化
        if use_semantic:
            question = self.optimize_question_with_semantic(question)

        print(f"\n🔍 查询问题: {original_question}")
        if use_semantic and question != original_question:
            print(f"🎯 优化后问题: {question}")
        print(f"📊 参数: top_k={top_k}, min_score={min_score}")
        print("-" * 80)

        results = self.vector_db.search_similar_questions(
            query=question,
            top_k=top_k,
            min_score=min_score
        )

        return results

    def format_results(self, results: List[Dict]) -> str:
        """
        格式化查询结果
        """
        if not results:
            return "❌ 未找到相关答案"

        formatted_output = f"✅ 找到 {len(results)} 个相关结果:\n\n"

        for i, result in enumerate(results, 1):
            formatted_output += f"【结果 {i}】相似度: {result.get('score', 0):.4f}\n"
            formatted_output += f"📝 问题: {result.get('question', '')}\n"
            formatted_output += f"💡 答案: {result.get('standard_answer', '')}\n"
            formatted_output += f"🏷️ 分类: {result.get('header', '')}\n"
            formatted_output += f"📁 来源: {result.get('file_source', '')}\n"

            raw_content = result.get('raw_content', '')
            formatted_output += "📃 原文: \n" + "*" * 60 + "\n" + f"{raw_content}\n" + "*" * 60 + "\n"

            # 安全地获取图片URL
            image_url = result.get('image_url', '无')
            formatted_output += f"🔗 图片: {image_url}\n"

            formatted_output += "-" * 60 + "\n"

        return formatted_output

    def interactive_query(self):
        """
        交互式查询模式
        """
        print("\n🎯 进入交互式查询模式")
        print("输入 'quit' 或 'exit' 退出程序")
        print("=" * 60)

        while True:
            try:
                # 获取用户输入
                question = input("\n请输入问题: ").strip()

                if question.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break

                if not question:
                    print("⚠️  问题不能为空，请重新输入")
                    continue

                # 执行查询
                results = self.query_question(question, top_k=5, min_score=0.3)

                # 显示结果
                print(self.format_results(results))

            except KeyboardInterrupt:
                print("\n\n👋 用户中断，再见！")
                break
            except Exception as e:
                print(f"❌ 查询过程中出错: {e}")
