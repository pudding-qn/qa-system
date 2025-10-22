# -*- coding=utf-8 -*-
# @Time: 2025/10/21 14:10
# @Author: 邱楠
# @File: excel_to_vector_db.py.py
# @Software: PyCharm


import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import chromadb
from chromadb.config import Settings
import glob
import json
from tqdm import tqdm
import re

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class ExcelToVectorDB:
    def __init__(self, model_name='BAAI/bge-small-zh', db_path="./vector_db", collection_name="qa_embeddings_bge"):
        """
        初始化向量数据库系统

        Args:
            model_name: 模型名称
            db_path: 数据库路径
            collection_name: 集合名称
        """
        print(f"正在加载模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        self.collection_name = collection_name

        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(path=db_path)

        # 创建或获取集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "问答对向量数据库_bge"}
        )
        print(f"✓ 向量数据库初始化完成! 使用集合: {collection_name}")

    def list_collections(self):
        """列出所有可用的集合"""
        collections = self.client.list_collections()
        collection_names = [col.name for col in collections]
        print(f"📁 数据库中的集合: {collection_names}")
        return collection_names

    def read_excel_files(self, excel_folder, file_pattern="*.xlsx"):
        """
        读取多个Excel文件中的问答对
        """
        print(f"正在读取Excel文件从: {excel_folder}")

        # 查找所有Excel文件
        excel_files = glob.glob(os.path.join(excel_folder, file_pattern))
        excel_files.extend(glob.glob(os.path.join(excel_folder, "*.xls")))

        if not excel_files:
            raise ValueError(f"在 {excel_folder} 中未找到Excel文件")

        print(f"找到 {len(excel_files)} 个Excel文件")

        all_qa_pairs = []

        for file_path in excel_files:
            print(f"正在处理: {os.path.basename(file_path)}")

            try:
                # 读取Excel文件
                excel_data = pd.read_excel(file_path)

                # 处理每一行数据
                for idx, row in excel_data.iterrows():
                    chunk_id = str(row["chunk_id"]).strip()
                    header = str(row["header"]).strip()
                    raw_content = str(row["content"]).strip()
                    standard_question = str(row["标准问题"]).strip()
                    extended_question = str(row["发散问题"]).strip()
                    standard_answer = str(row["客服答案"]).strip()
                    image_url = str(row["图片"]).strip()

                    qa_pair = {
                        "chunk_id": chunk_id,
                        "header": header,
                        "raw_content": raw_content,
                        "standard_question": standard_question,
                        "extended_question": extended_question,
                        "standard_answer": standard_answer,
                        "image_url": image_url,
                        "file_source": os.path.basename(file_path),
                        "row_index": idx
                    }
                    all_qa_pairs.append(qa_pair)

                print(
                    f"  ✓ 从 {os.path.basename(file_path)} 提取了 {len([p for p in all_qa_pairs if p['file_source'] == os.path.basename(file_path)])} 个问答对")

            except Exception as e:
                print(f"❌ 处理文件 {file_path} 时出错: {e}")
                continue

        print(f"✓ 总共提取了 {len(all_qa_pairs)} 个问答对")
        return all_qa_pairs

    def process_and_store_embeddings(self, excel_folder, batch_size=100):
        """
        处理问答对并存储到向量数据库
        """
        # 读取问答对
        qa_pairs = self.read_excel_files(excel_folder)

        if not qa_pairs:
            print("❌ 没有找到可用的问答对")
            return

        print("正在生成嵌入向量并存储到数据库...")
        seen_questions = set()

        for i in tqdm(range(0, len(qa_pairs), batch_size)):
            batch = qa_pairs[i:i + batch_size]

            # 准备问题和对应的元数据
            questions = []
            metadatas = []

            for item in batch:
                # 添加标准问题
                standard_question = item["standard_question"]
                if item["standard_question"] not in seen_questions:
                    seen_questions.add(standard_question)
                    questions.append(item["standard_question"])
                    metadatas.append({
                        "chunk_id": item["chunk_id"],
                        "header": item["header"],
                        "raw_content": item["raw_content"],
                        "standard_answer": item["standard_answer"],
                        "image_url": item["image_url"],
                        "file_source": item["file_source"],
                        "row_index": item["row_index"],
                        "question_type": "standard"  # 标记为标准问题
                    })

                # 添加发散问题（如果存在且有效）
                if item["extended_question"] and item["extended_question"] != 'nan':
                    questions.append(item["extended_question"])
                    metadatas.append({
                        "chunk_id": item["chunk_id"],
                        "header": item["header"],
                        "raw_content": item["raw_content"],
                        "standard_answer": item["standard_answer"],
                        "image_url": item["image_url"],
                        "file_source": item["file_source"],
                        "row_index": item["row_index"],
                        "question_type": "extended"  # 标记为发散问题
                    })

            if not questions:
                continue

            # 生成嵌入向量
            embeddings = self.model.encode(questions)
            embeddings = normalize(embeddings)

            # 准备数据 - 确保所有字段长度一致
            ids = [f"qa_{i + j}" for j in range(len(questions))]
            documents = questions  # 文档就是问题本身
            # print(documents)

            # 存储到向量数据库
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        print(f"✓ 成功存储 {len(qa_pairs)} 个问答对到向量数据库")

    def search_similar_questions(self, query, top_k=5, min_score=0.5):
        """
        搜索相似问题
        """
        # 生成查询向量
        query_embedding = self.model.encode([query])[0]
        query_embedding = normalize([query_embedding])[0]

        # 在向量数据库中搜索
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        formatted_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                score = 1 - distance  # 转换为相似度分数

                if score >= min_score:
                    formatted_results.append({
                        'question': results['documents'][0][i],
                        'chunk_id': results['metadatas'][0][i]['chunk_id'],
                        'header': results['metadatas'][0][i]['header'],
                        'raw_content': results['metadatas'][0][i]['raw_content'],
                        'standard_answer': results['metadatas'][0][i]['standard_answer'],
                        'image_url': results['metadatas'][0][i]['image_url'],
                        'file_source': results['metadatas'][0][i]['file_source'],
                        'score': float(score)
                    })

        return formatted_results

    def get_collection_stats(self):
        """获取集合统计信息"""
        return self.collection.count()

    def export_qa_data(self, output_file="qa_data_export.json"):
        """导出问答数据"""
        try:
            all_data = self.collection.get()

            if not all_data['ids']:
                print("❌ 数据库中没有数据")
                return

            qa_list = []
            for i in range(len(all_data['ids'])):
                # 安全地访问元数据
                metadata = all_data['metadatas'][i] if i < len(all_data['metadatas']) else {}

                qa_list.append({
                    "id": all_data['ids'][i],
                    "chunk_id": metadata.get('chunk_id', ''),
                    "header": metadata.get('header', ''),
                    "raw_content": metadata.get('raw_content', ''),
                    "question": all_data['documents'][i] if i < len(all_data['documents']) else '',
                    "standard_answer": metadata.get('standard_answer', ''),
                    "file_source": metadata.get('file_source', ''),
                    "question_type": metadata.get('question_type', 'unknown'),
                    "image_url": metadata.get('image_url', '')
                })

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_list, f, ensure_ascii=False, indent=2)

            print(f"✓ 问答数据已导出到 {output_file}，共 {len(qa_list)} 条记录")

        except Exception as e:
            print(f"❌ 导出数据时出错: {e}")


def main():
    # 初始化系统
    vector_db = ExcelToVectorDB()

    # 指定Excel文件所在的文件夹
    excel_folder = "./excel_files"  # 修改为你的Excel文件夹路径

    # 处理Excel文件并存储到向量数据库
    vector_db.process_and_store_embeddings(excel_folder)

    # 显示统计信息
    count = vector_db.get_collection_stats()
    print(f"\n📊 向量数据库中的问答对数量: {count}")

    # 测试搜索功能
    test_queries = [
        "智尊版L有什么特点？",
        "是否有特定的温度或湿度要求？",
        "低油耗技术会有什么影响吗对环境？"
    ]

    print("\n🔍 测试搜索功能:")
    print("=" * 60)

    for query in test_queries:
        print(f"\n查询: {query}")
        results = vector_db.search_similar_questions(query, top_k=3)

        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. 相似度: {result['score']:.4f}")
                print(f"     问题: {result['question']}")
                print(f"     来源: {result['file_source'], result['chunk_id'],result['header']}")
                print(f"     原始数据: {result['raw_content']}")
                print(f"     答案: {result['standard_answer']}")
        else:
            print("  ❌ 未找到相关答案")

        print("-" * 60)

    # 导出数据（可选）
    vector_db.export_qa_data()


if __name__ == "__main__":
    main()