"""SQLite storage: CGM stream, food nodes, images, and knowledge-base chunks."""

import sqlite3
from datetime import datetime, timedelta

import pandas as pd

SQL_DB = "glucose_timeline.db"
TS_FMT = "%Y-%m-%d %H:%M:%S"


def connect() -> sqlite3.Connection:
    return sqlite3.connect(SQL_DB)


def init_db():
    with connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS glucose_stream (timestamp TEXT PRIMARY KEY, value REAL)")
        conn.execute("""CREATE TABLE IF NOT EXISTS food_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, meal_type TEXT,
            ai_prediction TEXT, predicted_glucose_range TEXT, ai_advice TEXT,
            ai_calibration TEXT, rag_context_used TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS food_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT, node_id INTEGER, image_blob BLOB,
            FOREIGN KEY(node_id) REFERENCES food_nodes(id))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS kb_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, text TEXT, embedding BLOB)""")


def save_cgm(df) -> int:
    rows = [(r["timestamp"].strftime(TS_FMT), float(r["Glucose"])) for _, r in df.iterrows()]
    with connect() as conn:
        conn.executemany("INSERT OR REPLACE INTO glucose_stream VALUES (?,?)", rows)
    return len(rows)


def log_food(ts, meal, imgs, pred, pred_range, advice, rag) -> int:
    with connect() as conn:
        c = conn.cursor()
        c.execute("""INSERT INTO food_nodes
            (timestamp,meal_type,ai_prediction,predicted_glucose_range,ai_advice,rag_context_used)
            VALUES (?,?,?,?,?,?)""", (ts, meal, pred, pred_range, advice, rag))
        nid = c.lastrowid
        c.executemany("INSERT INTO food_images (node_id,image_blob) VALUES (?,?)", [(nid, i) for i in imgs])
    return nid


def get_node(nid: int) -> dict | None:
    conn = connect()
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM food_nodes WHERE id=?", (nid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_node_imgs(nid):
    with connect() as conn:
        rows = conn.execute("SELECT image_blob FROM food_images WHERE node_id=?", (nid,)).fetchall()
    return [r[0] for r in rows]


def get_data(days=7):
    cut = (datetime.now() - timedelta(days=days)).strftime(TS_FMT)
    with connect() as conn:
        cgm = pd.read_sql_query(
            "SELECT * FROM glucose_stream WHERE timestamp>=? ORDER BY timestamp", conn, params=(cut,))
        food = pd.read_sql_query(
            """SELECT id,timestamp,meal_type,ai_prediction,ai_advice,ai_calibration,predicted_glucose_range
               FROM food_nodes WHERE timestamp>=? ORDER BY timestamp""", conn, params=(cut,))
    if not cgm.empty:
        cgm["timestamp"] = pd.to_datetime(cgm["timestamp"])
    if not food.empty:
        food["timestamp"] = pd.to_datetime(food["timestamp"])
    return cgm, food


def get_window(ts_str, h=2.0):
    t = pd.to_datetime(ts_str)
    e = t + timedelta(hours=h)
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT timestamp,value FROM glucose_stream WHERE timestamp>=? AND timestamp<=? ORDER BY timestamp",
            conn, params=(t.strftime(TS_FMT), e.strftime(TS_FMT)))


def update_calibration(nid, txt):
    with connect() as conn:
        conn.execute("UPDATE food_nodes SET ai_calibration=? WHERE id=?", (txt, nid))


def recent_food_nodes(limit=5) -> list[dict]:
    conn = connect()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id,timestamp,meal_type,ai_prediction,predicted_glucose_range,ai_advice,ai_calibration
           FROM food_nodes ORDER BY timestamp DESC LIMIT ?""", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_all():
    with connect() as conn:
        for t in ["food_images", "food_nodes", "glucose_stream"]:
            conn.execute(f"DELETE FROM {t}")
