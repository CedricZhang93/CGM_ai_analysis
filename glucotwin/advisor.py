"""The three AI features of GlucoTwin, each built as a TinyAgent with CGM tools."""

import json
from datetime import datetime

from glucotwin import db
from glucotwin.agent import TinyAgent
from glucotwin.cgm_tools import CGMToolkit, meal_outcome
from glucotwin.knowledge import KnowledgeBase
from glucotwin.llm import LLM, EmbeddingModel
from glucotwin.memory import Memory

PREDICT_SYSTEM = """你是一位个性化控糖专家，服务于一个"血糖数字孪生"系统。现在时间：{now}。

你可以调用工具查询用户的真实数据。工作方式：
1. {kb_rule}
2. 调用 get_meal_history 查看用户吃类似食物时的**实测**血糖反应和以往的校准经验；有相似记录→经验模式，否则→第一性原理（根据纤维/脂肪/蛋白/碳水结构推理）。
3. 如有必要，调用 get_glucose_curve 查看餐前 1 小时的血糖作为起点。
4. 信息足够后停止调用工具，给出最终答案。

最终答案只输出一个 JSON 对象，不要其他文字：
{{"food_name":"食物名","is_new_food":true,"reasoning_mode":"经验模式/第一性原理 + 依据","risk_analysis":"风险分析（含个人禁忌）","predicted_1h_range":"X.X-X.X mmol/L","predicted_2h_range":"X.X-X.X mmol/L","advice":"具体建议"}}"""

CALIBRATE_SYSTEM = """你是血糖数字孪生系统的校准模块。你会拿到一次餐前预测和该餐后的实测 CGM 数据。
如需对比，可调用 get_meal_history 查看其它餐次；{kb_rule}
请用中文输出：
1. 预测是否准确（对比 1h / 2h 区间与实测）
2. 偏差原因分析
3. 最后一行以「经验：」开头，总结一条可供以后预测复用的经验（会被存入数字孪生记忆）。"""

CHAT_SYSTEM = """你是用户的私人血糖健康顾问。现在时间：{now}。
回答前请按需调用工具获取用户的真实数据（血糖统计、某段时间的曲线、饮食记录与实测反应），不要编造数据。
{kb_rule}
用中文回答，简洁专业；涉及数字时注明来源时间段。"""

KB_RULE_ON = "用户有个人医嘱知识库：必须先调用 search_knowledge_base 检索与本问题相关的禁忌、药物和控糖目标，个人医嘱优先于通用常识。"
KB_RULE_OFF = "用户暂无个人知识库。"


def _setup(api_key: str) -> tuple[LLM, CGMToolkit, str]:
    llm = LLM(api_key=api_key)
    has_kb = KnowledgeBase.stats()["chunks"] > 0
    kb = KnowledgeBase(EmbeddingModel(api_key=api_key)) if has_kb else None
    return llm, CGMToolkit(kb), KB_RULE_ON if has_kb else KB_RULE_OFF


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def predict_meal(images: list[tuple[bytes, str]], meal_type: str, timestamp: str, api_key: str) -> dict:
    llm, toolkit, kb_rule = _setup(api_key)
    agent = TinyAgent(llm, toolkit.build(), PREDICT_SYSTEM.format(now=_now(), kb_rule=kb_rule), max_steps=6)
    prediction = agent.run_json(
        f"用户上传了{len(images)}张食物照片，餐别：{meal_type}，进食时间：{timestamp}。请预测餐后血糖。",
        images=images,
    )
    return {"prediction": prediction, "rag": "\n---\n".join(toolkit.rag_used), "trajectory": agent.trajectory_dict()}


def calibrate_meal(nid: int, api_key: str) -> dict:
    node = db.get_node(nid)
    if node is None:
        raise LookupError("节点不存在")
    window = db.get_window(node["timestamp"])
    if window.empty:
        raise ValueError("该时段无CGM数据，无法校准")

    llm, toolkit, kb_rule = _setup(api_key)
    agent = TinyAgent(llm, toolkit.build(), CALIBRATE_SYSTEM.format(kb_rule=kb_rule), max_steps=3)
    task = (
        f"【进食时间】{node['timestamp']}（{node['meal_type']}）\n"
        f"【预测】{node['predicted_glucose_range']}\n"
        f"【实测摘要】{json.dumps(meal_outcome(node['timestamp']), ensure_ascii=False)}\n"
        f"【实测CGM数据】\n{window.to_csv(index=False)}"
    )
    result = agent.run(task)
    db.update_calibration(nid, result)
    return {"calibration": result, "trajectory": agent.trajectory_dict()}


def chat(message: str, history: list[dict], api_key: str) -> dict:
    llm, toolkit, kb_rule = _setup(api_key)
    memory = Memory()
    memory.load_history(history)
    agent = TinyAgent(llm, toolkit.build(), CHAT_SYSTEM.format(now=_now(), kb_rule=kb_rule), memory=memory, max_steps=6)
    answer = agent.run(message)
    return {"response": answer, "trajectory": agent.trajectory_dict()}
