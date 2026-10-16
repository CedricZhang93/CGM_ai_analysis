# app.py - GlucoTwin AI | FastAPI Backend
# Run: uvicorn app:app --reload --port 8000
#
# The AI layer lives in ./glucotwin: a from-scratch LLM client (urllib -> OpenAI-compatible API),
# tool registry, ReAct agent loop and SQLite vector store. No vendor SDK / LangChain required.

import base64
import io
import json
import logging
import os
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from glucotwin import advisor, db
from glucotwin.cgm_tools import glucose_stats
from glucotwin.knowledge import KnowledgeBase
from glucotwin.llm import EmbeddingModel, LLMError

HTML_FILE = Path(__file__).parent / "mvp_rag_V3.html"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def resolve_key(api_key: str) -> str:
    """Browser-supplied key first, then the GEMINI_API_KEY environment variable."""
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise HTTPException(400, "缺少 API Key，请在 Settings 中填写或设置环境变量 GEMINI_API_KEY")
    return key


db.init_db()
app = FastAPI(title="GlucoTwin AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(LLMError)
async def llm_error_handler(request, exc: LLMError):
    return JSONResponse(status_code=502, content={"detail": f"模型调用失败: {exc}"})


@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(HTML_FILE.read_text(encoding="utf-8"))


@app.get("/api/stats")
async def stats(range: str = Query("7d")):
    days = {"24h": 1, "7d": 7, "30d": 30}.get(range, 7)
    cgm, food = db.get_data(days)
    food_count = 0 if food.empty else len(food)
    if cgm.empty:
        return {"time_in_range": None, "avg_glucose": None, "estimated_a1c": None,
                "cgm_count": 0, "food_count": food_count}
    s = glucose_stats(cgm["value"])
    return {"time_in_range": s["time_in_range"], "avg_glucose": s["avg_glucose"],
            "estimated_a1c": s["estimated_a1c"], "cgm_count": len(cgm), "food_count": food_count}


@app.get("/api/timeline")
async def timeline(range: str = Query("7d")):
    days = {"24h": 1, "7d": 7, "30d": 30}.get(range, 7)
    cgm, food = db.get_data(days)
    # Downsample CGM to max 800 points
    cgm_out = []
    if not cgm.empty:
        step = max(1, len(cgm) // 800)
        cgm_out = [{"timestamp": r.timestamp.isoformat(), "value": float(r.value)}
                   for r in cgm.iloc[::step].itertuples()]
    food_out = []
    for _, r in food.iterrows():
        w = db.get_window(r["timestamp"].strftime(db.TS_FMT))
        try:
            pred = json.loads(r["predicted_glucose_range"]) if r["predicted_glucose_range"] else None
        except json.JSONDecodeError:
            pred = None
        food_out.append({
            "id": int(r["id"]),
            "timestamp": r["timestamp"].isoformat(),
            "meal_type": r["meal_type"],
            "ai_prediction": r["ai_prediction"],
            "ai_advice": r["ai_advice"],
            "ai_calibration": r["ai_calibration"],
            "peak_glucose": float(w["value"].max()) if not w.empty else None,
            "prediction": pred,
        })
    return {"cgm": cgm_out, "food": food_out}


@app.post("/api/cgm/import")
async def import_cgm(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(400, f"CSV 解析失败: {e}")
    df.columns = [c.lower().strip() for c in df.columns]
    t = next((c for c in df.columns if "time" in c or "date" in c), None)
    v = next((c for c in df.columns if "glu" in c or "value" in c or "mmol" in c), None)
    if not t or not v:
        raise HTTPException(400, "未找到时间列或血糖列，请确保CSV含 timestamp/glucose 等列名")
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[t], errors="coerce"),
        "Glucose": pd.to_numeric(df[v], errors="coerce"),
    }).dropna()
    return {"success": True, "count": db.save_cgm(out)}


@app.post("/api/knowledge/upload")
def upload_kb(file: UploadFile = File(...), api_key: str = Form("")):
    kb = KnowledgeBase(EmbeddingModel(api_key=resolve_key(api_key)))
    try:
        n = kb.add_document(file.file.read(), file.filename)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    return {"success": True, "message": f"已消化 {n} 条知识规则"}


@app.get("/api/knowledge/status")
async def kb_status():
    s = KnowledgeBase.stats()
    return {"exists": s["chunks"] > 0, "rag_available": True, **s}


@app.post("/api/food/predict")
def predict(
    files: list[UploadFile] = File(...),
    api_key: str = Form(""),
    meal_type: str = Form(...),
    timestamp: str = Form(...),
):
    key = resolve_key(api_key)
    images = [(f.file.read(), f.content_type or "image/jpeg") for f in files]
    try:
        result = advisor.predict_meal(images, meal_type, timestamp, key)
    except ValueError as e:
        raise HTTPException(500, str(e))
    pred = result["prediction"]
    nid = db.log_food(timestamp, meal_type, [raw for raw, _ in images], pred.get("food_name", ""),
                      json.dumps(pred, ensure_ascii=False), pred.get("advice", ""), result["rag"])
    return {"success": True, "node_id": nid, "prediction": pred, "trajectory": result["trajectory"]}


@app.get("/api/food/{nid}/images")
async def food_imgs(nid: int):
    return {"images": [base64.b64encode(i).decode() for i in db.get_node_imgs(nid)]}


@app.post("/api/food/{nid}/calibrate")
def calibrate(nid: int, api_key: str = Form("")):
    try:
        result = advisor.calibrate_meal(nid, resolve_key(api_key))
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"success": True, **result}


@app.post("/api/chat")
def chat(message: str = Form(...), api_key: str = Form(""), history: str = Form("[]")):
    try:
        past = json.loads(history)
    except json.JSONDecodeError:
        past = []
    result = advisor.chat(message, past if isinstance(past, list) else [], resolve_key(api_key))
    return {"success": True, **result}


@app.delete("/api/data/clear")
async def clear_data():
    db.clear_all()
    return {"success": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
