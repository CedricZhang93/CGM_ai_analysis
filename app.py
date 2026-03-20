# app.py - GlucoTwin AI | FastAPI Backend
# Run: uvicorn app:app --reload --port 8000

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sqlite3, pandas as pd, io, os, json, re, base64, time
from datetime import datetime, timedelta
import google.generativeai as genai
from PIL import Image
import uvicorn

# RAG (optional — install: pip install langchain-google-genai langchain-community chromadb)
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

SQL_DB = 'glucose_timeline.db'
CHROMA_PATH = "./chroma_db"
HTML_FILE = Path(__file__).parent / "mvp_rag_V3.html"

# ==========================================================
# Database Layer
# ==========================================================

def init_db():
    conn = sqlite3.connect(SQL_DB)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS glucose_stream (timestamp TEXT PRIMARY KEY, value REAL)')
    c.execute('''CREATE TABLE IF NOT EXISTS food_nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, meal_type TEXT,
        ai_prediction TEXT, predicted_glucose_range TEXT, ai_advice TEXT,
        ai_calibration TEXT, rag_context_used TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS food_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT, node_id INTEGER, image_blob BLOB,
        FOREIGN KEY(node_id) REFERENCES food_nodes(id))''')
    conn.commit()
    conn.close()

def save_cgm(df) -> int:
    conn = sqlite3.connect(SQL_DB)
    n = 0
    for _, r in df.iterrows():
        try:
            conn.execute("INSERT OR REPLACE INTO glucose_stream VALUES (?,?)",
                (r['timestamp'].strftime("%Y-%m-%d %H:%M:%S"), float(r['Glucose'])))
            n += 1
        except:
            pass
    conn.commit()
    conn.close()
    return n

def log_food(ts, meal, imgs, pred, pred_range, advice, rag) -> int:
    conn = sqlite3.connect(SQL_DB)
    c = conn.cursor()
    c.execute('''INSERT INTO food_nodes
        (timestamp,meal_type,ai_prediction,predicted_glucose_range,ai_advice,rag_context_used)
        VALUES (?,?,?,?,?,?)''', (ts, meal, pred, pred_range, advice, rag))
    nid = c.lastrowid
    for img in imgs:
        c.execute('INSERT INTO food_images (node_id,image_blob) VALUES (?,?)', (nid, img))
    conn.commit()
    conn.close()
    return nid

def get_node_imgs(nid):
    conn = sqlite3.connect(SQL_DB)
    rows = conn.execute("SELECT image_blob FROM food_images WHERE node_id=?", (nid,)).fetchall()
    conn.close()
    return [r[0] for r in rows]

def get_data(days=7):
    conn = sqlite3.connect(SQL_DB)
    cut = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    cgm = pd.read_sql_query(
        "SELECT * FROM glucose_stream WHERE timestamp>=? ORDER BY timestamp", conn, params=(cut,))
    food = pd.read_sql_query(
        """SELECT id,timestamp,meal_type,ai_prediction,ai_advice,ai_calibration,predicted_glucose_range
           FROM food_nodes WHERE timestamp>=? ORDER BY timestamp""", conn, params=(cut,))
    conn.close()
    if not cgm.empty:
        cgm['timestamp'] = pd.to_datetime(cgm['timestamp'])
    if not food.empty:
        food['timestamp'] = pd.to_datetime(food['timestamp'])
    return cgm, food

def get_window(ts_str, h=2):
    conn = sqlite3.connect(SQL_DB)
    t = pd.to_datetime(ts_str)
    e = t + timedelta(hours=h)
    df = pd.read_sql_query(
        "SELECT timestamp,value FROM glucose_stream WHERE timestamp>=? AND timestamp<=? ORDER BY timestamp",
        conn, params=(t.strftime("%Y-%m-%d %H:%M:%S"), e.strftime("%Y-%m-%d %H:%M:%S")))
    conn.close()
    return df

def update_calibration(nid, txt):
    conn = sqlite3.connect(SQL_DB)
    conn.execute("UPDATE food_nodes SET ai_calibration=? WHERE id=?", (txt, nid))
    conn.commit()
    conn.close()

def food_history_summary() -> str:
    conn = sqlite3.connect(SQL_DB)
    nodes = pd.read_sql_query(
        "SELECT id,timestamp,ai_prediction FROM food_nodes ORDER BY id DESC LIMIT 5", conn)
    conn.close()
    if nodes.empty:
        return "（无历史记录）"
    lines = []
    for _, r in nodes.iterrows():
        w = get_window(r['timestamp'])
        if not w.empty:
            delta = w['value'].max() - w['value'].iloc[0]
            lines.append(f"- {r['timestamp']} 吃 '{r['ai_prediction']}': 血糖 +{delta:.1f}")
        else:
            lines.append(f"- {r['timestamp']} 吃 '{r['ai_prediction']}': (无数据)")
    return "\n".join(lines)

def latest_context() -> str:
    conn = sqlite3.connect(SQL_DB)
    last = pd.read_sql_query("SELECT * FROM food_nodes ORDER BY id DESC LIMIT 1", conn)
    conn.close()
    if last.empty:
        return "暂无记录。"
    return f"最近一餐: {last['timestamp'].iloc[0]} - {last['ai_prediction'].iloc[0]}. 建议: {last['ai_advice'].iloc[0]}"

# ==========================================================
# RAG Layer
# ==========================================================

def build_kb(content: bytes, fname: str, key: str):
    if not RAG_AVAILABLE:
        return False, "RAG依赖未安装，请运行: pip install langchain-google-genai langchain-community chromadb pypdf"
    fp = f"./temp_{fname}"
    with open(fp, "wb") as f:
        f.write(content)
    try:
        loader = PyPDFLoader(fp) if fname.lower().endswith(".pdf") else TextLoader(fp, encoding="utf-8")
        chunks = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100).split_documents(loader.load())
        if not chunks:
            return False, "文档为空"
        emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)
        vs = Chroma(embedding_function=emb, persist_directory=CHROMA_PATH)
        for i in range(0, len(chunks), 5):
            try:
                vs.add_documents(chunks[i:i+5])
                time.sleep(3)
            except Exception as e:
                if "429" in str(e):
                    time.sleep(60)
                    vs.add_documents(chunks[i:i+5])
                else:
                    raise
        vs.persist()
        os.remove(fp)
        return True, f"已消化 {len(chunks)} 条知识规则"
    except Exception as e:
        if os.path.exists(fp):
            os.remove(fp)
        return False, str(e)

def query_kb(q: str, key: str):
    if not RAG_AVAILABLE or not os.path.exists(CHROMA_PATH):
        return None
    try:
        emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)
        docs = Chroma(persist_directory=CHROMA_PATH, embedding_function=emb).similarity_search(q, k=3)
        return "\n".join([d.page_content for d in docs])
    except:
        return None

# ==========================================================
# AI Layer
# ==========================================================

def ai_predict(imgs: list, key: str):
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    rag = query_kb("饮食禁忌 药物反应 升糖系数", key)
    rag_part = f"\n【RAG医嘱】\n{rag}" if rag else ""
    history = food_history_summary()
    prompt = f"""你是一位个性化控糖专家。{rag_part}
【近期历史】
{history}

分析上传的食物图片：
1. 判断模式：历史有类似食物→经验模式；否则→第一性原理（根据纤维/脂肪/碳水结构推理）
2. 预测餐后 1h 和 2h 血糖范围（mmol/L）
3. 给出控糖建议

严格返回 JSON 格式（无其他内容）：
{{"food_name":"食物名","is_new_food":true,"reasoning_mode":"说明","risk_analysis":"风险分析","predicted_1h_range":"X.X-X.X mmol/L","predicted_2h_range":"X.X-X.X mmol/L","advice":"具体建议"}}"""
    resp = model.generate_content([prompt] + imgs)
    return resp.text, rag or ""

def ai_calibrate(pred_json: str, df, key: str) -> str:
    genai.configure(api_key=key)
    prompt = f"【预测】{pred_json}\n【实测CGM数据】\n{df.to_csv(index=False)}\n请验证预测准确性，说明偏差原因，总结一条经验供数字孪生记忆。"
    return genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt).text

def ai_chat(query: str, key: str) -> str:
    genai.configure(api_key=key)
    rag = query_kb(query, key)
    rag_part = f"\n【知识库参考】\n{rag}" if rag else ""
    prompt = f"""你是私人血糖健康顾问，基于用户的个人数据提供精准建议。
最近记录：{latest_context()}{rag_part}

用户问题：{query}

请用中文回答，简洁专业。"""
    return genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt).text

# ==========================================================
# Init
# ==========================================================
init_db()

# ==========================================================
# FastAPI
# ==========================================================
app = FastAPI(title="GlucoTwin AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(HTML_FILE.read_text(encoding="utf-8"))

@app.get("/api/stats")
async def stats(range: str = Query("7d")):
    days = {"24h": 1, "7d": 7, "30d": 30}.get(range, 7)
    cgm, food = get_data(days)
    food_count = 0 if food.empty else len(food)
    if cgm.empty:
        return {"time_in_range": None, "avg_glucose": None, "estimated_a1c": None,
                "cgm_count": 0, "food_count": food_count}
    avg = float(cgm['value'].mean())
    tir = float(((cgm['value'] >= 3.8) & (cgm['value'] <= 7.8)).mean() * 100)
    a1c = round((avg + 2.59) / 1.59, 1)
    return {"time_in_range": round(tir, 1), "avg_glucose": round(avg, 1),
            "estimated_a1c": a1c, "cgm_count": len(cgm), "food_count": food_count}

@app.get("/api/timeline")
async def timeline(range: str = Query("7d")):
    days = {"24h": 1, "7d": 7, "30d": 30}.get(range, 7)
    cgm, food = get_data(days)
    # Downsample CGM to max 800 points
    cgm_out = []
    if not cgm.empty:
        step = max(1, len(cgm) // 800)
        for _, r in cgm.iloc[::step].iterrows():
            cgm_out.append({"timestamp": r['timestamp'].isoformat(), "value": float(r['value'])})
    food_out = []
    if not food.empty:
        for _, r in food.iterrows():
            w = get_window(r['timestamp'].strftime("%Y-%m-%d %H:%M:%S"))
            peak = float(w['value'].max()) if not w.empty else None
            pred = None
            try:
                if r['predicted_glucose_range']:
                    pred = json.loads(r['predicted_glucose_range'])
            except:
                pass
            food_out.append({
                "id": int(r['id']),
                "timestamp": r['timestamp'].isoformat(),
                "meal_type": r['meal_type'],
                "ai_prediction": r['ai_prediction'],
                "ai_advice": r['ai_advice'],
                "ai_calibration": r['ai_calibration'],
                "peak_glucose": peak,
                "prediction": pred
            })
    return {"cgm": cgm_out, "food": food_out}

@app.post("/api/cgm/import")
async def import_cgm(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        df.columns = [c.lower().strip() for c in df.columns]
        t = next((c for c in df.columns if 'time' in c or 'date' in c), None)
        v = next((c for c in df.columns if 'glu' in c or 'value' in c or 'mmol' in c), None)
        if not t or not v:
            raise HTTPException(400, "未找到时间列或血糖列，请确保CSV含 timestamp/glucose 等列名")
        out = pd.DataFrame({
            'timestamp': pd.to_datetime(df[t]),
            'Glucose': pd.to_numeric(df[v], errors='coerce')
        }).dropna()
        return {"success": True, "count": save_cgm(out)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/knowledge/upload")
async def upload_kb(file: UploadFile = File(...), api_key: str = Form(...)):
    ok, msg = build_kb(await file.read(), file.filename, api_key)
    if ok:
        return {"success": True, "message": msg}
    raise HTTPException(500, msg)

@app.get("/api/knowledge/status")
async def kb_status():
    return {"exists": os.path.exists(CHROMA_PATH) and RAG_AVAILABLE, "rag_available": RAG_AVAILABLE}

@app.post("/api/food/predict")
async def predict(
    files: list[UploadFile] = File(...),
    api_key: str = Form(...),
    meal_type: str = Form(...),
    timestamp: str = Form(...)
):
    try:
        pil_imgs, raw_imgs = [], []
        for f in files:
            b = await f.read()
            raw_imgs.append(b)
            pil_imgs.append(Image.open(io.BytesIO(b)))
        text, rag = ai_predict(pil_imgs, api_key)
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if not m:
            raise HTTPException(500, f"AI返回格式错误: {text[:200]}")
        pred = json.loads(m.group(0))
        nid = log_food(timestamp, meal_type, raw_imgs,
                       pred.get('food_name', ''), m.group(0), pred.get('advice', ''), rag)
        return {"success": True, "node_id": nid, "prediction": pred}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/food/{nid}/images")
async def food_imgs(nid: int):
    return {"images": [base64.b64encode(i).decode() for i in get_node_imgs(nid)]}

@app.post("/api/food/{nid}/calibrate")
async def calibrate(nid: int, api_key: str = Form(...)):
    conn = sqlite3.connect(SQL_DB)
    row = pd.read_sql_query("SELECT * FROM food_nodes WHERE id=?", conn, params=(nid,))
    conn.close()
    if row.empty:
        raise HTTPException(404, "节点不存在")
    df = get_window(row.iloc[0]['timestamp'])
    if df.empty:
        raise HTTPException(400, "该时段无CGM数据，无法校准")
    result = ai_calibrate(row.iloc[0]['predicted_glucose_range'], df, api_key)
    update_calibration(nid, result)
    return {"success": True, "calibration": result}

@app.post("/api/chat")
async def chat(message: str = Form(...), api_key: str = Form(...)):
    try:
        return {"success": True, "response": ai_chat(message, api_key)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/api/data/clear")
async def clear_data():
    conn = sqlite3.connect(SQL_DB)
    for t in ['food_images', 'food_nodes', 'glucose_stream']:
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    conn.close()
    return {"success": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
