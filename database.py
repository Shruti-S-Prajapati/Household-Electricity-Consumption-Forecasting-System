import sqlite3
from datetime import datetime

DB_PATH = "predictions.db"


def init_db():
    """Create the predictions table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature_c REAL,
            humidity_pct REAL,
            day_of_week TEXT,
            is_weekend INTEGER,
            is_holiday INTEGER,
            people_at_home INTEGER,
            hours_at_home REAL,
            ac_hours REAL,
            fan_hours REAL,
            tv_hours REAL,
            washing_machine_used INTEGER,
            predicted_kwh REAL
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(data: dict, predicted_kwh: float):
    """Insert one prediction record."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, temperature_c, humidity_pct, day_of_week,
            is_weekend, is_holiday, people_at_home, hours_at_home,
            ac_hours, fan_hours, tv_hours, washing_machine_used,
            predicted_kwh
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        data["temperature_c"],
        data["humidity_pct"],
        data["day_of_week"],
        data["is_weekend"],
        data["is_holiday"],
        data["people_at_home"],
        data["hours_at_home"],
        data["ac_hours"],
        data["fan_hours"],
        data["tv_hours"],
        data["washing_machine_used"],
        predicted_kwh
    ))
    conn.commit()
    conn.close()


def get_history(limit: int = 50):
    """Return the most recent `limit` predictions, newest first."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_stats():
    """Return simple summary stats across all stored predictions."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*),
            AVG(predicted_kwh),
            MIN(predicted_kwh),
            MAX(predicted_kwh)
        FROM predictions
    """)
    count, avg_kwh, min_kwh, max_kwh = cursor.fetchone()
    conn.close()
    return {
        "total_predictions": count,
        "average_kwh": round(avg_kwh, 3) if avg_kwh is not None else None,
        "min_kwh": min_kwh,
        "max_kwh": max_kwh
    }


def clear_history():
    """Delete all stored predictions (useful for demo/testing)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()