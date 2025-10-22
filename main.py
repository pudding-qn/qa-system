# -*- coding=utf-8 -*-
# @Time: 2025/10/22 11:28
# @Author: 邱楠
# @File: main.py
# @Software: PyCharm


from vector_db_query import VectorDBQuery
import os
import argparse

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def main():
    parser = argparse.ArgumentParser(description='向量数据库查询系统')
    parser.add_argument('--question', '-q', type=str, help='要查询的问题')
    parser.add_argument('--top_k', '-k', type=int, default=5, help='返回结果数量')
    parser.add_argument('--min_score', '-s', type=float, default=0.3, help='最小相似度阈值')
    parser.add_argument('--interactive', '-i', action='store_true', help='进入交互模式')
    # parser.add_argument('--semantic', action='store_true', help='启用语义分析')
    # parser.add_argument('--spark_appid', type=str, help='星火APPID')
    # parser.add_argument('--spark_apikey', type=str, help='星火APIKey')
    # parser.add_argument('--spark_apisecret', type=str, help='星火APISecret')

    args = parser.parse_args()

    # 初始化查询系统
    query_system = VectorDBQuery(
        use_semantic_analysis=True,
        spark_appid="17fd554e",
        spark_apikey="f6e09b7ea35d556c60c16bc8b06822ae",
        spark_apisecret="YzQwOWQ4M2U3NzM2ODYzYzE3ODI0M2M0"
    )

    # 检查数据库状态
    count = query_system.vector_db.get_collection_stats()
    print(f"📊 向量数据库中的问答对数量: {count}")

    if args.interactive:
        # 交互模式
        query_system.interactive_query()
    elif args.question:
        # 单次查询模式
        results = query_system.query_question(
            question=args.question,
            top_k=args.top_k,
            min_score=args.min_score,
            use_semantic=True
        )
        print(query_system.format_results(results))
    else:
        # 默认进入交互模式
        query_system.interactive_query()


if __name__ == "__main__":
    main()