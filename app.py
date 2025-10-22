# -*- coding=utf-8 -*-
import streamlit as st
import os
import sys
import pandas as pd
from typing import List, Dict
import time
import io

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['STREAMLIT_SERVER_ENABLE_FILE_WATCHER'] = 'false'

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
</style>
""", unsafe_allow_html=True)

def check_dependencies():
    """检查依赖是否安装"""
    missing_deps = []
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    try:
        import chromadb
    except ImportError:
        missing_deps.append("chromadb")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    return missing_deps

# 检查依赖
missing_deps = check_dependencies()
if missing_deps:
    st.error(f"❌ 缺少必要的依赖: {', '.join(missing_deps)}")
    st.info("请在 requirements.txt 中添加以上依赖包")

# 使用缓存加载系统，避免重复初始化
@st.cache_resource(show_spinner="正在加载智能问答系统...")
def load_qa_system():
    """加载QA系统"""
    try:
        # 检查依赖
        missing_deps = check_dependencies()
        if missing_deps:
            raise ImportError(f"缺少依赖: {', '.join(missing_deps)}")
        
        # 延迟导入，避免在缓存时加载
        from vector_db_query import VectorDBQuery
        
        system = VectorDBQuery(
            use_semantic_analysis=True,
            spark_appid="17fd554e",
            spark_apikey="f6e09b7ea35d556c60c16bc8b06822ae",
            spark_apisecret="YzQwOWQ4M2U3NzM2ODYzYzE3ODI0M2M0"
        )
        return system
    except ImportError as e:
        st.error(f"❌ 依赖导入失败: {str(e)}")
        st.info("""
        **解决方案：**
        1. 检查 requirements.txt 是否包含所有必要依赖
        2. 确保 sentence-transformers, chromadb, torch 等包已正确安装
        3. 重新部署应用
        """)
        return None
    except Exception as e:
        st.error(f"❌ 系统初始化失败: {str(e)}")
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
        output = io.BytesIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        return output.getvalue()
    except Exception as e:
        st.error(f"CSV转换错误: {e}")
        try:
            output = io.BytesIO()
            df.to_csv(output, index=False, encoding='gbk')
            return output.getvalue()
        except:
            output = io.BytesIO()
            df.to_csv(output, index=False, encoding='utf-8')
            return output.getvalue()

def display_results(results: List[Dict], query_time: float, query_question: str):
    """显示查询结果"""
    if not results:
        st.warning("⚠️ 未找到与您问题相关的答案，请尝试调整查询条件或重新表述问题。")
        return

    st.success(f"🔍 查询问题: **{query_question}**")
    st.success(f"✅ 找到 **{len(results)}** 个相关结果 (查询耗时: {query_time:.2f}秒)")

    for i, result in enumerate(results, 1):
        with st.container():
            st.markdown(f"""
            <div class="result-card">
                <div class="similarity-score">相似度: {result.get('score', 0):.4f}</div>
                <div style="font-weight: bold; margin-bottom: 0.5rem;">📝 问题: {result.get('question', '')}</div>
                <div style="background: white; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
                    💡 答案: {result.get('standard_answer', '')}
                </div>
                <div style="color: #666; font-size: 0.9rem;">
                    🏷️ 分类: {result.get('header', '')} | 
                    📁 来源: {result.get('file_source', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)

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

    st.markdown('<div class="main-header">🤖 智能问答系统</div>', unsafe_allow_html=True)

    # 加载系统
    query_system = load_qa_system()

    if query_system is None:
        st.error("""
        ## ❌ 系统初始化失败
        
        **可能的原因：**
        1. 缺少必要的Python包
        2. 向量数据库文件不存在
        3. 内存不足
        
        **解决方案：**
        1. 检查 requirements.txt 文件
        2. 确保 vector_db 文件夹已上传
        3. 查看Streamlit Cloud的日志获取详细信息
        """)
        
        # 显示依赖检查
        with st.expander("🔧 依赖检查"):
            missing = check_dependencies()
            if missing:
                st.error(f"缺少依赖: {', '.join(missing)}")
            else:
                st.success("所有依赖已安装")
        return

    # 显示系统状态
    try:
        count = query_system.vector_db.get_collection_stats()
        st.success(f"✅ 系统加载完成！向量数据库中共有 **{count}** 个问答对")
    except:
        st.warning("⚠️ 无法获取数据库统计信息")

    # 查询表单
    with st.form("query_form"):
        col1, col2 = st.columns([2, 1])

        with col1:
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

        submitted = st.form_submit_button("🔍 开始查询", use_container_width=True)

    # 处理查询逻辑
    should_query = False
    query_question = ""

    if st.session_state.example_clicked and st.session_state.example_question:
        should_query = True
        query_question = st.session_state.example_question
        st.session_state.example_clicked = False
        st.session_state.example_question = ""
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
            st.session_state.current_results = None
        else:
            st.session_state.current_results = results
            st.session_state.current_query_time = query_time
            st.session_state.current_query_question = query_question

    # 显示结果
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
            "车辆保养周期是多久？"
        ]

        for example in examples:
            if st.button(
                    f"{example}",
                    key=f"example_{example}",
                    use_container_width=True
            ):
                st.session_state.example_clicked = True
                st.session_state.example_question = example
                st.session_state.current_results = None
                st.rerun()

if __name__ == "__main__":
    main()
