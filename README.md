# GlucoTwin AI — 血糖数字孪生系统

基于 AI 的个性化血糖管理工具。上传食物图片，AI 预测餐后血糖响应；导入 CGM 数据，追踪长期趋势；上传医嘱文件构建私人知识库，让 AI 顾问更了解你的身体。

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![Gemini](https://img.shields.io/badge/Google_Gemini-2.0_Flash-orange)

---

## 功能

| 模块 | 功能 |
|------|------|
| **Dashboard** | 血糖统计（达标率 / 均值 / 估算 HbA1c）+ 交互式趋势图，支持 24h / 7d / 30d |
| **Food Log** | 拍照记录饮食，AI 双模预测（经验模式 / 第一性原理）餐后 1h & 2h 血糖范围 |
| **History** | 活动日志 + 点击查看详情 + AI 事后校准（对比预测与实测 CGM 数据） |
| **AI Advisor** | 基于知识库 + 历史数据的对话式健康顾问，聊天记录本地持久化 |
| **Knowledge Base** | 上传 TXT/PDF 医嘱文件，RAG 向量检索，预测时自动参考 |
| **Settings** | API Key 管理 + CGM CSV 导入 |

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/CedricZhang93/CGM_ai_analysis.git
cd CGM_ai_analysis
```

### 2. 安装依赖

```bash
pip install -r requirements_app.txt
```

> Knowledge Base (RAG) 功能需要额外安装：
> ```bash
> pip install langchain-google-genai langchain-community chromadb pypdf
> ```
> 不需要 RAG 功能可跳过此步，其他功能不受影响。

### 3. 获取 Google Gemini API Key

前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 免费获取 API Key。

### 4. 启动服务

```bash
uvicorn app:app --port 8000
```

打开浏览器访问：**http://localhost:8000**

在页面右上角或 Settings 页面填入 API Key 即可开始使用。

---

## 导入 CGM 数据

在 **Settings → 导入 CGM 数据** 上传 CSV 文件。CSV 格式要求：

| 列名（含以下关键词之一即可） | 说明 |
|------|------|
| `timestamp` / `date` / `time` | 时间列 |
| `glucose` / `value` / `mmol` | 血糖值（单位：mmol/L） |

仓库内附有示例文件 `test_cgm.csv` 可供测试。

---

## 上传知识库

在 **Knowledge Base** 页面上传 TXT 或 PDF 格式的医嘱、体质报告、饮食禁忌等文件。上传后 AI 在每次食物预测和对话时会自动检索参考。

仓库内附有示例文件 `my_health_profile.txt.txt` 可供参考。

---

## 技术栈

- **后端**：Python · FastAPI · SQLite · Google Gemini API
- **前端**：原生 HTML/JS · Tailwind CSS · Chart.js
- **RAG**：LangChain · ChromaDB · Google Embedding API
- **数据**：SQLite（血糖流 + 食物节点 + 图片）

---

## 项目结构

```
├── app.py                  # FastAPI 后端（全部 API + 服务静态页）
├── mvp_rag_V3.html         # 前端 SPA（单文件，无需构建）
├── requirements_app.txt    # Python 依赖
├── test_cgm.csv            # CGM 示例数据
├── my_health_profile.txt   # 知识库示例文件
└── 启动记录.txt             # 快速启动命令
```

运行后自动生成：
```
├── glucose_timeline.db     # SQLite 数据库
└── chroma_db/              # 向量知识库（上传文件后）
```

---

## 注意事项

- API Key 仅存储在**浏览器本地**（localStorage），不会上传到任何服务器
- `glucose_timeline.db` 和 `chroma_db/` 为本地运行数据，已加入 `.gitignore`
- RAG 知识库构建受 Google API 速率限制，每批次间隔 3 秒，文件较大时需耐心等待
