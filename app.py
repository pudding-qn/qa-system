# -*- coding=utf-8 -*-
# @Time: 2025/10/22 13:02
# @Author: 邱楠
# @File: app.py
# @Software: PyCharm

import streamlit as st
import os
from vector_db_query import VectorDBQuery
import pandas as pd
from typing import List, Dict
import time
import io

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 页面配置
st.set_page_config(
    page_title="智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #1f77b4;
    }
    .similarity-score {
        background-color: #1f77b4;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .question-text {
        font-size: 1.1rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .answer-text {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .metadata {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .example-question {
        background-color: #e6f3ff;
        border: 1px solid #b3d9ff;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.2rem 0;
        cursor: pointer;
    }
    .example-question:hover {
        background-color: #d1e7ff;
    }
</style>
""", unsafe_allow_html=True)


# 使用缓存加载系统，避免重复初始化
@st.cache_resource(show_spinner="正在加载智能问答系统...")
def load_qa_system():
    """加载QA系统"""
    try:
        system = VectorDBQuery(
            use_semantic_analysis=True,
            spark_appid="17fd554e",
            spark_apikey="f6e09b7ea35d556c60c16bc8b06822ae",
            spark_apisecret="YzQwOWQ4M2U3NzM2ODYzYzE3ODI0M2M0"
        )
        return system
    except Exception as e:
        st.error(f"系统初始化失败: {str(e)}")
        return None


def execute_query(query_system, question, top_k, min_score, use_semantic):
    """执行查询并返回结果"""
    with st.spinner("🔍 正在搜索相关答案..."):
        start_time = time.time()
        try:
            results = query_system.query_question(
                question=question,
                top_k=top_k,
                min_score=min_score,
                use_semantic=use_semantic
            )
            query_time = time.time() - start_time
            return results, query_time, None
        except Exception as e:
            return None, 0, str(e)


def convert_to_csv(df):
    """将DataFrame转换为CSV格式，解决中文乱码问题"""
    try:
        # 方法1: 使用utf-8-sig编码（推荐，兼容Excel）
        output = io.BytesIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        return output.getvalue()
    except Exception as e:
        st.error(f"CSV转换错误: {e}")
        # 方法2: 如果utf-8-sig失败，尝试gbk编码
        try:
            output = io.BytesIO()
            df.to_csv(output, index=False, encoding='gbk')
            return output.getvalue()
        except:
            # 方法3: 最后尝试普通utf-8
            output = io.BytesIO()
            df.to_csv(output, index=False, encoding='utf-8')
            return output.getvalue()


def display_results(results: List[Dict], query_time: float, query_question: str):
    """显示查询结果"""
    if not results:
        st.warning("⚠️ 未找到与您问题相关的答案，请尝试调整查询条件或重新表述问题。")
        return

    # 显示查询的问题和统计信息
    st.success(f"🔍 查询问题: **{query_question}**")
    st.success(f"✅ 找到 **{len(results)}** 个相关结果 (查询耗时: {query_time:.2f}秒)")

    # 显示结果
    for i, result in enumerate(results, 1):
        with st.container():
            st.markdown(f"""
            <div class="result-card">
                <div class="similarity-score">相似度: {result.get('score', 0):.4f}</div>
                <div class="question-text">📝 问题: {result.get('question', '')}</div>
                <div class="answer-text">💡 答案: {result.get('standard_answer', '')}</div>
                <div class="metadata">
                    🏷️ 分类: {result.get('header', '')} | 
                    📁 来源: {result.get('file_source', '')} |
                    🔗 图片: {result.get('image_url', '无')}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 显示原始数据表格 - 使用新的列顺序
    with st.expander("📋 查看详细数据"):
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            # 使用新的列顺序：file_source, raw_content, question, standard_answer, score, header
            display_columns = ['file_source', 'raw_content', 'question', 'standard_answer', 'score', 'header']

            # 只显示存在的列
            available_columns = [col for col in display_columns if col in results_df.columns]

            # 重新排列DataFrame的列顺序
            results_df = results_df[available_columns]

            # 格式化显示
            styled_df = results_df.style.format({
                'score': '{:.4f}'  # 格式化相似度分数为4位小数
            })

            st.dataframe(styled_df, use_container_width=True)

            # 添加多种格式的下载按钮
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                # CSV下载（解决乱码问题）
                csv_data = convert_to_csv(results_df)
                st.download_button(
                    label="📥 下载CSV (推荐)",
                    data=csv_data,
                    file_name=f"查询结果_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="使用UTF-8-BOM编码，兼容Excel中文显示"
                )

            with col2:
                # Excel下载（避免编码问题）
                try:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        results_df.to_excel(writer, index=False, sheet_name='查询结果')
                    excel_data = excel_buffer.getvalue()

                    st.download_button(
                        label="📊 下载Excel",
                        data=excel_data,
                        file_name=f"查询结果_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Excel格式，无编码问题"
                    )
                except Exception as e:
                    st.error(f"Excel导出失败: {e}")

            with col3:
                # JSON下载
                json_data = results_df.to_json(force_ascii=False, orient='records', indent=2)
                st.download_button(
                    label="📄 下载JSON",
                    data=json_data,
                    file_name=f"查询结果_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="JSON格式，保持中文原样"
                )

            # 编码说明
            st.info("💡 **下载说明**: 如果CSV在Excel中显示乱码，请选择Excel格式下载，或使用文本编辑器打开CSV文件")


def main():
    """主函数"""
    # 初始化session state
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'example_clicked' not in st.session_state:
        st.session_state.example_clicked = False
    if 'example_question' not in st.session_state:
        st.session_state.example_question = ""
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'current_query_time' not in st.session_state:
        st.session_state.current_query_time = 0
    if 'current_query_question' not in st.session_state:
        st.session_state.current_query_question = ""

    # 加载系统
    query_system = load_qa_system()

    if query_system is None:
        st.error("❌ 系统加载失败，请检查配置")
        return

    # 显示系统状态
    try:
        count = query_system.vector_db.get_collection_stats()
        st.success(f"✅ 系统加载完成！向量数据库中共有 **{count}** 个问答对")
    except:
        st.warning("⚠️ 无法获取数据库统计信息")

    # 主界面
    st.markdown('<div class="main-header">🤖 智能问答系统</div>', unsafe_allow_html=True)

    # 查询表单
    with st.form("query_form"):
        col1, col2 = st.columns([2, 1])

        with col1:
            # 如果点击了示例问题，自动填充到输入框
            default_question = st.session_state.example_question if st.session_state.example_clicked else ""
            question = st.text_area(
                "💬 请输入您的问题:",
                value=default_question,
                placeholder="例如：智尊版L有什么特点？是否有特定的温度或湿度要求？...",
                height=100,
                key="question_input"
            )

        with col2:
            top_k = st.slider("📊 返回结果数量:", min_value=1, max_value=10, value=5, key="top_k_slider")
            min_score = st.slider("🎯 最小相似度阈值:", min_value=0.1, max_value=0.9, value=0.3, step=0.1,
                                  key="min_score_slider")
            use_semantic = st.checkbox("🧠 启用语义分析优化", value=True, key="semantic_checkbox")

        # 查询按钮
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🔍 开始查询", use_container_width=True)

    # 处理查询逻辑
    should_query = False
    query_question = ""

    # 检查是否有需要查询的示例问题（自动查询）
    if st.session_state.example_clicked and st.session_state.example_question:
        should_query = True
        query_question = st.session_state.example_question
        # 重置状态，避免重复查询
        st.session_state.example_clicked = False
        st.session_state.example_question = ""
    # 检查是否点击了开始查询按钮
    elif submitted and question and question.strip():
        should_query = True
        query_question = question

    # 执行查询
    if should_query and query_question:
        st.session_state.query_count += 1
        st.session_state.last_query = query_question

        results, query_time, error = execute_query(
            query_system, query_question, top_k, min_score, use_semantic
        )

        if error:
            st.error(f"❌ 查询过程中出现错误: {error}")
            # 清除之前的结果
            st.session_state.current_results = None
        else:
            # 保存当前查询结果到session state
            st.session_state.current_results = results
            st.session_state.current_query_time = query_time
            st.session_state.current_query_question = query_question

    # 显示结果（从session state中读取，避免重复执行）
    if st.session_state.current_results is not None:
        display_results(
            st.session_state.current_results,
            st.session_state.current_query_time,
            st.session_state.current_query_question
        )

    # 侧边栏
    with st.sidebar:
        st.header("ℹ️ 系统信息")
        st.metric("查询次数", st.session_state.query_count)

        if st.session_state.last_query:
            with st.expander("📝 最近查询"):
                st.text_area("", st.session_state.last_query, height=100, key="last_query_display")

        st.header("💡 示例问题")
        st.markdown("点击以下问题自动查询：")

        examples = [
            "智尊版L有什么特点？",
            "是否有特定的温度或湿度要求？",
            "低油耗技术会有什么影响吗对环境？",
            "如何开启智能驾驶功能？",
            "车辆保养周期是多久？",
            "车辆出现故障怎么办？",
            "智能座舱有哪些功能？",
            "电池续航能力如何？"
        ]

        for example in examples:
            # 使用st.button来处理点击事件
            if st.button(
                    f"{example}",
                    key=f"example_{example}",
                    use_container_width=True
            ):
                # 设置示例问题状态
                st.session_state.example_clicked = True
                st.session_state.example_question = example
                # 清除当前结果，准备显示新结果
                st.session_state.current_results = None
                st.rerun()

        # 显示当前查询参数
        with st.expander("⚙️ 当前查询参数"):
            st.write(f"返回结果数量: {top_k}")
            st.write(f"相似度阈值: {min_score}")
            st.write(f"语义分析: {'启用' if use_semantic else '禁用'}")

        # 快速操作
        st.header("🚀 快速操作")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 重新加载", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()

        with col2:
            if st.button("🧹 清空记录", use_container_width=True):
                st.session_state.query_count = 0
                st.session_state.last_query = ""
                st.session_state.current_results = None
                st.rerun()

        # 显示当前查询状态
        with st.expander("📊 当前状态"):
            if st.session_state.current_results is not None:
                st.success(f"有 {len(st.session_state.current_results)} 条查询结果")
            else:
                st.info("暂无查询结果")


if __name__ == "__main__":
    main()
