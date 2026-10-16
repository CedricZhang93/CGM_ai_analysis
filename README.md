# GlucoTwin AI — 血糖数字孪生系统

基于 AI 的个性化血糖管理工具。上传食物图片，AI 预测餐后血糖响应；导入 CGM 数据，追踪长期趋势；上传医嘱文件构建私人知识库，让 AI 顾问更了解你的身体。

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-orange)

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

> AI 层不依赖任何厂商 SDK / LangChain，只用标准库 `urllib` 直接请求模型接口。`pypdf` 仅在上传 PDF 知识库时需要。

### 3. 获取 Google Gemini API Key

前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 免费获取 API Key。

### 4. 启动服务

```bash
uvicorn app:app --port 8000
```

打开浏览器访问：**http://localhost:8000**

在页面右上角或 Settings 页面填入 API Key 即可开始使用（也可设置环境变量 `GEMINI_API_KEY`）。

### 可选：更换模型

AI 层走 OpenAI 兼容协议，可通过环境变量切换：

| 变量 | 默认值 | 说明 |
|------|------|------|
| `GLUCOTWIN_MODEL` | `gemini-2.5-flash` | 对话/多模态模型 |
| `GLUCOTWIN_EMBED_MODEL` | `gemini-embedding-001` | 向量模型 |
| `GLUCOTWIN_BASE_URL` | Gemini OpenAI 兼容端点 | 换成 `http://localhost:11434/v1` 即可用本地 Ollama |

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

- **后端**：Python · FastAPI · SQLite
- **AI 层**：自研 `glucotwin` 包 — urllib 直连 OpenAI 兼容接口 + 原生工具调用 + ReAct Agent 循环
- **前端**：原生 HTML/JS · Tailwind CSS · Chart.js
- **RAG**：自研切块 + Embedding API + SQLite 存向量 + numpy 余弦检索

## AI 架构

参考 *An Illustrated Guide to AI Agents* 的 TinyAgent 设计，三个 AI 功能都是一个带工具的 Agent，
模型自己决定查什么数据，而不是把所有上下文一次性塞进 prompt：

```
用户请求 ─► TinyAgent.run()
              │  ┌──────────────── 循环（最多 N 步）────────────────┐
              ├─►│ LLM.generate(messages, tools)                    │
              │  │   ├─ 有 tool_calls → NativeTools.execute → role=tool 观察结果写回 Memory
              │  │   └─ 无 tool_calls → 最终答案，退出               │
              │  └───────────────────────────────────────────────────┘
              └─► 返回答案 + trajectory（每一步的工具调用与观察，API 里一并返回）
```

| 工具 | 作用 |
|------|------|
| `search_knowledge_base` | 检索个人医嘱知识库（禁忌、药物、目标） |
| `get_meal_history` | 历史饮食 + 当时的预测 + **实测** CGM 反应 + 以往校准经验 |
| `get_glucose_curve` | 任意时间窗的血糖曲线（15 分钟重采样） |
| `get_glucose_stats` | N 天的 TIR / 均值 / eA1c / CV |

校准结果末尾的「经验：」会通过 `get_meal_history` 在之后的预测中被读回，形成数字孪生的长期记忆闭环。
- **数据**：SQLite（血糖流 + 食物节点 + 图片）

---

## 项目结构

```
├── app.py                  # FastAPI 路由层
├── glucotwin/              # AI 层（无第三方 SDK）
│   ├── llm.py              #   LLM / EmbeddingModel：urllib 请求 + 重试
│   ├── memory.py           #   短期记忆：消息列表 + 图片 + 对话历史
│   ├── tools.py            #   工具注册表：函数签名/docstring → JSON Schema
│   ├── agent.py            #   TinyAgent：ReAct 循环 + trajectory + JSON 修复
│   ├── knowledge.py        #   长期记忆：切块 / 向量 / 余弦检索
│   ├── cgm_tools.py        #   血糖领域工具
│   ├── advisor.py          #   预测 / 校准 / 对话 三个 Agent 的 prompt 与组装
│   └── db.py               #   SQLite 数据层
├── mvp_rag_V3.html         # 前端 SPA（单文件，无需构建）
├── requirements_app.txt    # Python 依赖
├── test_cgm.csv            # CGM 示例数据
├── my_health_profile.txt   # 知识库示例文件
└── 启动记录.txt             # 快速启动命令
```

运行后自动生成：
```
└── glucose_timeline.db     # SQLite 数据库（含知识库向量表 kb_chunks）
```

---

## 注意事项

- API Key 仅存储在**浏览器本地**（localStorage），不会上传到任何服务器
- `glucose_timeline.db` 为本地运行数据，已加入 `.gitignore`
- 旧版 `chroma_db/` 中的知识库不会自动迁移，升级后请重新上传医嘱文件
- 遇到 429 / 5xx 会自动指数退避重试
