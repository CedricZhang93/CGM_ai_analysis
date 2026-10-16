"""Domain tools the agent can call. Docstrings become the tool descriptions sent to the model."""

import json

import pandas as pd

from glucotwin import db
from glucotwin.knowledge import KnowledgeBase
from glucotwin.tools import NativeTools

TIR_LOW, TIR_HIGH = 3.8, 7.8


def glucose_stats(values: pd.Series) -> dict:
    avg = float(values.mean())
    return {
        "time_in_range": round(float(((values >= TIR_LOW) & (values <= TIR_HIGH)).mean() * 100), 1),
        "avg_glucose": round(avg, 1),
        "estimated_a1c": round((avg + 2.59) / 1.59, 1),
        "min": round(float(values.min()), 1),
        "max": round(float(values.max()), 1),
        "cv_percent": round(float(values.std() / avg * 100), 1) if avg else None,
    }


def meal_outcome(timestamp: str) -> dict | None:
    """Actual post-meal response from CGM: baseline, peak, rise, value at ~2h."""
    w = db.get_window(timestamp, h=2)
    if w.empty:
        return None
    w["timestamp"] = pd.to_datetime(w["timestamp"])
    baseline = float(w["value"].iloc[0])
    peak_idx = w["value"].idxmax()
    peak_minutes = (w.loc[peak_idx, "timestamp"] - w["timestamp"].iloc[0]).total_seconds() / 60
    return {
        "baseline": round(baseline, 1),
        "peak": round(float(w["value"].max()), 1),
        "rise": round(float(w["value"].max()) - baseline, 1),
        "minutes_to_peak": int(peak_minutes),
        "value_at_end": round(float(w["value"].iloc[-1]), 1),
    }


class CGMToolkit:
    """Builds the tool registry and records which knowledge-base snippets were used."""

    def __init__(self, kb: KnowledgeBase | None):
        self.kb = kb
        self.rag_used: list[str] = []

    def build(self) -> NativeTools:
        tools = NativeTools()
        tools.add_tool(self.get_glucose_stats)
        tools.add_tool(self.get_glucose_curve)
        tools.add_tool(self.get_meal_history)
        if self.kb is not None:
            tools.add_tool(self.search_knowledge_base)
        return tools

    def get_glucose_stats(self, days: int = 7) -> dict:
        """Summary statistics of the user's CGM data over the last N days (TIR 3.8-7.8, mean, eA1c, CV).

        Args:
            days: Look-back window in days, e.g. 1, 7 or 30.
        """
        cgm, food = db.get_data(int(days))
        if cgm.empty:
            return {"message": f"最近 {days} 天没有 CGM 数据"}
        return {"days": days, "readings": len(cgm), "meals_logged": len(food), **glucose_stats(cgm["value"])}

    def get_glucose_curve(self, start_time: str, hours: float = 3.0) -> dict:
        """CGM readings (mmol/L) for a time window, resampled to 15-minute points.

        Args:
            start_time: Window start, format 'YYYY-MM-DD HH:MM:SS'.
            hours: Window length in hours (max 24).
        """
        w = db.get_window(start_time, h=min(float(hours), 24))
        if w.empty:
            return {"message": "该时段无 CGM 数据"}
        w["timestamp"] = pd.to_datetime(w["timestamp"])
        series = w.set_index("timestamp")["value"].resample("15min").mean().dropna().round(1)
        return {
            "points": {t.strftime("%m-%d %H:%M"): v for t, v in series.items()},
            **glucose_stats(w["value"]),
        }

    def get_meal_history(self, limit: int = 5, keyword: str = "") -> list | dict:
        """Recent logged meals with the AI's prediction, the ACTUAL CGM response, and past calibration lessons.

        Use this to find how the user really reacted to similar foods before.

        Args:
            limit: Number of meals to return (max 20).
            keyword: Optional food name filter, e.g. '米饭'. Empty returns all.
        """
        nodes = db.recent_food_nodes(limit=50 if keyword else min(int(limit), 20))
        if keyword:
            nodes = [n for n in nodes if keyword in (n["ai_prediction"] or "")][: int(limit)]
        if not nodes:
            return {"message": "没有匹配的饮食记录"}
        out = []
        for n in nodes:
            try:
                pred = json.loads(n["predicted_glucose_range"] or "{}")
            except json.JSONDecodeError:
                pred = {}
            out.append({
                "time": n["timestamp"],
                "meal_type": n["meal_type"],
                "food": n["ai_prediction"],
                "predicted_1h": pred.get("predicted_1h_range"),
                "predicted_2h": pred.get("predicted_2h_range"),
                "actual": meal_outcome(n["timestamp"]) or "无CGM数据",
                "calibration_lesson": (n["ai_calibration"] or "")[:400] or None,
            })
        return out

    def search_knowledge_base(self, query: str) -> str:
        """Search the user's personal medical knowledge base (doctor's orders, allergies, medication, targets).

        Args:
            query: What to look up, e.g. '燕麦 禁忌' or '餐后血糖目标'.
        """
        hits = self.kb.search(query, k=3)
        if not hits:
            return "知识库中没有相关内容。"
        text = "\n---\n".join(f"[{h['source']} | 相似度 {h['score']}]\n{h['text']}" for h in hits)
        self.rag_used.extend(h["text"] for h in hits if h["text"] not in self.rag_used)
        return text
