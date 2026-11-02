"""
Aplikasi Peramalan Statistik
=============================
Streamlit app for statistical forecasting (Naïve, SARIMA/SARFIMA, SARIMAX, MLP),
in-sample vs out-of-sample comparison, confidence intervals,
and SQLite persistence.

Author: BPS Kota Bontang
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import io
import logging
import re
from datetime import datetime
from scipy import stats as sp_stats

# ── Forecasting libraries ──
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──
st.set_page_config(
    page_title="PRIMA — Peramalan Interaktif dengan Model Adaptif",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; }
    .stMetric { background: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #4CAF50; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    .best-model { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  color: white; padding: 20px; border-radius: 12px; text-align: center; }
    .rima-header {
        background: linear-gradient(135deg, #1B4F72 0%, #2E86C1 100%);
        color: white; padding: 18px 28px; border-radius: 14px;
        margin-bottom: 12px; text-align: center;
    }
    .rima-header h1 { margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: 4px; }
    .rima-header p { margin: 4px 0 0 0; font-size: 1.05rem; opacity: 0.9; font-weight: 300; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CHART DOWNLOAD HELPER
# ══════════════════════════════════════════════════════════════

def _chart_to_png(fig):
    """Convert matplotlib figure to PNG bytes for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def make_line_chart_png(periodes, series_dict, title="", ylabel="Nilai",
                        dashed_series=None, unit_label=""):
    """Create a matplotlib line chart and return PNG bytes.

    Args:
        periodes: list of x-axis labels
        series_dict: dict of {series_name: list_of_values}
        title: chart title
        ylabel: y-axis label
        dashed_series: set of series names to draw with dashed line
        unit_label: e.g. " (Juta Rupiah)"
    """
    if dashed_series is None:
        dashed_series = set()

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for idx, (name, vals) in enumerate(series_dict.items()):
        style = "--" if name in dashed_series else "-"
        c = colors[idx % len(colors)]
        # Filter None/NaN
        x_plot, y_plot = [], []
        for i, v in enumerate(vals):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                x_plot.append(i)
                y_plot.append(v)
        ax.plot(x_plot, y_plot, style, color=c, label=name, linewidth=1.8,
                marker="o", markersize=3)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(f"{ylabel}{unit_label}", fontsize=10)
    ax.set_xlabel("Periode", fontsize=10)

    # X-axis ticks
    n = len(periodes)
    if n > 20:
        step = max(1, n // 15)
        tick_pos = list(range(0, n, step))
    else:
        tick_pos = list(range(n))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([periodes[i] for i in tick_pos], rotation=45,
                       ha="right", fontsize=8)

    # Y-axis formatting
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _chart_to_png(fig)


# ══════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════
DB_PATH = os.path.join(os.path.dirname(__file__), "forecasting_data.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    # Users table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    # Migration: add is_admin column if not exists (for existing databases)
    try:
        conn.execute("SELECT is_admin FROM users LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT NULL,
            name TEXT NOT NULL,
            frequency TEXT NOT NULL,
            unit TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            period_label TEXT NOT NULL,
            value REAL NOT NULL,
            sort_order INTEGER NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
        )
    """)
    # Migration: add user_id column if not exists (for existing databases)
    try:
        conn.execute("SELECT user_id FROM datasets LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE datasets ADD COLUMN user_id INTEGER DEFAULT NULL")
    # Tabel untuk menyimpan data eksogen
    conn.execute("""
        CREATE TABLE IF NOT EXISTS exog_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            col_name TEXT NOT NULL,
            col_order INTEGER NOT NULL,
            row_order INTEGER NOT NULL,
            value REAL NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
        )
    """)
    # Tabel untuk menyimpan data eksogen periode ramalan (future)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS exog_future_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            col_name TEXT NOT NULL,
            col_order INTEGER NOT NULL,
            row_order INTEGER NOT NULL,
            value REAL NOT NULL,
            horizon INTEGER NOT NULL DEFAULT 4,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    return conn


# ── User management ──

def list_users():
    conn = get_db()
    rows = conn.execute("SELECT id, name, is_admin, created_at FROM users ORDER BY name").fetchall()
    conn.close()
    return rows  # each row: (id, name, is_admin, created_at)


def create_user(name, is_admin=0):
    conn = get_db()
    now = datetime.now().isoformat()
    # If this is the first user, automatically make them admin
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count == 0:
        is_admin = 1
    cur = conn.execute("INSERT INTO users (name, is_admin, created_at) VALUES (?,?,?)",
                       (name, is_admin, now))
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    return uid


def delete_user(uid):
    conn = get_db()
    # Delete user's datasets and their data points
    ds_ids = [r[0] for r in conn.execute("SELECT id FROM datasets WHERE user_id=?", (uid,)).fetchall()]
    for did in ds_ids:
        conn.execute("DELETE FROM data_points WHERE dataset_id=?", (did,))
    conn.execute("DELETE FROM datasets WHERE user_id=?", (uid,))
    conn.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit()
    conn.close()


def get_user_info(uid):
    conn = get_db()
    row = conn.execute("SELECT id, name, is_admin FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    return row  # (id, name, is_admin)


def set_admin(uid, is_admin):
    """Set or revoke admin status for a user."""
    conn = get_db()
    conn.execute("UPDATE users SET is_admin=? WHERE id=?", (1 if is_admin else 0, uid))
    conn.commit()
    conn.close()


def count_admins():
    """Return number of admin users."""
    conn = get_db()
    n = conn.execute("SELECT COUNT(*) FROM users WHERE is_admin=1").fetchone()[0]
    conn.close()
    return n


def list_datasets(user_id=None):
    conn = get_db()
    if user_id is not None:
        rows = conn.execute(
            "SELECT id, name, frequency, unit, created_at, updated_at FROM datasets WHERE user_id=? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, name, frequency, unit, created_at, updated_at FROM datasets ORDER BY updated_at DESC"
        ).fetchall()
    conn.close()
    return rows


def list_all_datasets_with_owner():
    """Return all datasets with owner name for admin view."""
    conn = get_db()
    rows = conn.execute(
        "SELECT d.id, d.name, d.frequency, d.unit, d.created_at, d.updated_at, "
        "COALESCE(u.name, '(tanpa pemilik)') AS owner_name "
        "FROM datasets d LEFT JOIN users u ON d.user_id = u.id "
        "ORDER BY u.name, d.updated_at DESC"
    ).fetchall()
    conn.close()
    return rows


def count_data_points(did):
    """Return number of data points in a dataset."""
    conn = get_db()
    n = conn.execute("SELECT COUNT(*) FROM data_points WHERE dataset_id=?", (did,)).fetchone()[0]
    conn.close()
    return n


def create_dataset(name, frequency, unit="", user_id=None):
    conn = get_db()
    now = datetime.now().isoformat()
    cur = conn.execute(
        "INSERT INTO datasets (user_id, name, frequency, unit, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (user_id, name, frequency, unit, now, now),
    )
    conn.commit()
    did = cur.lastrowid
    conn.close()
    return did


def delete_dataset(did):
    conn = get_db()
    conn.execute("DELETE FROM data_points WHERE dataset_id=?", (did,))
    conn.execute("DELETE FROM datasets WHERE id=?", (did,))
    conn.commit()
    conn.close()


def get_dataset_info(did):
    conn = get_db()
    row = conn.execute("SELECT id, name, frequency, unit FROM datasets WHERE id=?", (did,)).fetchone()
    conn.close()
    return row


def get_data_points(did):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, period_label, value, sort_order FROM data_points WHERE dataset_id=? ORDER BY sort_order",
        (did,),
    ).fetchall()
    conn.close()
    return rows


def add_data_point(did, label, value):
    conn = get_db()
    mx = conn.execute("SELECT COALESCE(MAX(sort_order),0) FROM data_points WHERE dataset_id=?", (did,)).fetchone()[0]
    conn.execute(
        "INSERT INTO data_points (dataset_id, period_label, value, sort_order) VALUES (?,?,?,?)",
        (did, label, value, mx + 1),
    )
    conn.execute("UPDATE datasets SET updated_at=? WHERE id=?", (datetime.now().isoformat(), did))
    conn.commit()
    conn.close()


def delete_data_point(pid):
    conn = get_db()
    conn.execute("DELETE FROM data_points WHERE id=?", (pid,))
    conn.commit()
    conn.close()


def bulk_insert_points(did, records):
    import math
    conn = get_db()
    mx = conn.execute("SELECT COALESCE(MAX(sort_order),0) FROM data_points WHERE dataset_id=?", (did,)).fetchone()[0]
    for i, (lbl, val) in enumerate(records, start=1):
        # Skip NaN/None values
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        conn.execute(
            "INSERT INTO data_points (dataset_id, period_label, value, sort_order) VALUES (?,?,?,?)",
            (did, lbl, val, mx + i),
        )
    conn.execute("UPDATE datasets SET updated_at=? WHERE id=?", (datetime.now().isoformat(), did))
    conn.commit()
    conn.close()


def clear_data_points(did):
    conn = get_db()
    conn.execute("DELETE FROM data_points WHERE dataset_id=?", (did,))
    conn.execute("DELETE FROM exog_data WHERE dataset_id=?", (did,))
    conn.commit()
    conn.close()


# ── Exogenous data persistence ──

def save_exog_data(dataset_id, exog_array, col_names):
    """Simpan data eksogen ke database. Hapus data lama dulu."""
    conn = get_db()
    conn.execute("DELETE FROM exog_data WHERE dataset_id=?", (dataset_id,))
    if exog_array is not None and len(col_names) > 0:
        rows = []
        for col_idx, cname in enumerate(col_names):
            for row_idx in range(len(exog_array)):
                if exog_array.ndim == 2:
                    val = float(exog_array[row_idx, col_idx])
                else:
                    val = float(exog_array[row_idx])
                rows.append((dataset_id, cname, col_idx, row_idx, val))
        conn.executemany(
            "INSERT INTO exog_data (dataset_id, col_name, col_order, row_order, value) "
            "VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def load_exog_data(dataset_id):
    """Muat data eksogen dari database. Returns (exog_array, col_names) atau (None, [])."""
    conn = get_db()
    rows = conn.execute(
        "SELECT col_name, col_order, row_order, value FROM exog_data "
        "WHERE dataset_id=? ORDER BY col_order, row_order", (dataset_id,)
    ).fetchall()
    conn.close()
    if not rows:
        return None, []
    # Susun ulang ke array
    col_names_ordered = []
    col_data = {}
    for col_name, col_order, row_order, value in rows:
        if col_name not in col_data:
            col_names_ordered.append(col_name)
            col_data[col_name] = []
        col_data[col_name].append(value)
    arrays = [col_data[c] for c in col_names_ordered]
    exog_array = np.column_stack(arrays) if len(arrays) > 1 else np.array(arrays[0]).reshape(-1, 1)
    return exog_array, col_names_ordered


def clear_exog_data(dataset_id):
    """Hapus data eksogen dari database."""
    conn = get_db()
    conn.execute("DELETE FROM exog_data WHERE dataset_id=?", (dataset_id,))
    conn.commit()
    conn.close()


def save_exog_future(dataset_id, exog_array, col_names, horizon):
    """Simpan data eksogen periode ramalan ke database."""
    conn = get_db()
    conn.execute("DELETE FROM exog_future_data WHERE dataset_id=?", (dataset_id,))
    if exog_array is not None and len(col_names) > 0:
        rows = []
        for col_idx, cname in enumerate(col_names):
            for row_idx in range(exog_array.shape[0]):
                if exog_array.ndim == 2:
                    val = float(exog_array[row_idx, col_idx])
                else:
                    val = float(exog_array[row_idx])
                rows.append((dataset_id, cname, col_idx, row_idx, val, horizon))
        conn.executemany(
            "INSERT INTO exog_future_data (dataset_id, col_name, col_order, row_order, value, horizon) "
            "VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def load_exog_future(dataset_id):
    """Muat data eksogen periode ramalan dari database.
    Returns (exog_array, col_names, horizon) atau (None, [], 0)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT col_name, col_order, row_order, value, horizon FROM exog_future_data "
        "WHERE dataset_id=? ORDER BY col_order, row_order", (dataset_id,)
    ).fetchall()
    conn.close()
    if not rows:
        return None, [], 0
    col_names_ordered = []
    col_data = {}
    horizon = rows[0][4]
    for col_name, col_order, row_order, value, h in rows:
        if col_name not in col_data:
            col_names_ordered.append(col_name)
            col_data[col_name] = []
        col_data[col_name].append(value)
    arrays = [col_data[c] for c in col_names_ordered]
    exog_array = np.column_stack(arrays) if len(arrays) > 1 else np.array(arrays[0]).reshape(-1, 1)
    return exog_array, col_names_ordered, horizon


def clear_exog_future(dataset_id):
    """Hapus data eksogen ramalan dari database."""
    conn = get_db()
    conn.execute("DELETE FROM exog_future_data WHERE dataset_id=?", (dataset_id,))
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════
# CONFIDENCE INTERVAL HELPERS
# ══════════════════════════════════════════════════════════════

def residual_ci(residuals, point_forecast, h, confidence=0.95):
    """Compute CI from residuals using t-distribution.
    Returns (lower, upper) arrays of length h."""
    residuals = np.array(residuals)
    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) < 3:
        return None, None
    se = np.std(residuals, ddof=1)
    df = len(residuals) - 1
    t_val = sp_stats.t.ppf((1 + confidence) / 2, df)
    # Widen CI with sqrt(h) for multi-step ahead
    margin = np.array([t_val * se * np.sqrt(i + 1) for i in range(h)])
    pf = np.array(point_forecast[:h])
    return pf - margin, pf + margin


# ══════════════════════════════════════════════════════════════
# FORECASTING MODELS (12 models + SARFIMA otomatis + SARIMAX opsional)
# ══════════════════════════════════════════════════════════════

def calc_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calc_mae(actual, predicted):
    return mean_absolute_error(actual, predicted)


def calc_rmse(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))


def parse_number(val):
    """Parse angka dalam format Indonesia (1.234,56) maupun Inggris (1,234.56)."""
    import math
    if val is None:
        raise ValueError("None value")
    if isinstance(val, float) and math.isnan(val):
        raise ValueError("NaN value")
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(" ", "")
    if not s or s.lower() in ("nan", "none", "null", "", "-", "n/a", "na"):
        raise ValueError("empty or invalid")
    # Deteksi format berdasarkan posisi terakhir titik dan koma
    last_dot = s.rfind(".")
    last_comma = s.rfind(",")
    if last_dot > 0 and last_comma > 0:
        if last_comma > last_dot:
            # Format Indonesia: 1.234,56 → koma = desimal, titik = ribuan
            s = s.replace(".", "").replace(",", ".")
        else:
            # Format Inggris: 1,234.56 → titik = desimal, koma = ribuan
            s = s.replace(",", "")
    elif last_comma > 0 and last_dot < 0:
        # Hanya koma: bisa desimal (3,14) atau ribuan (1,000)
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) != 3:
            # Desimal: 3,14 atau 0,5
            s = s.replace(",", ".")
        elif len(parts) == 2 and len(parts[1]) == 3:
            # Ambigu: 1,000 — asumsikan desimal jika bagian kiri pendek
            if len(parts[0].lstrip("-")) <= 3:
                s = s.replace(",", ".")
            else:
                s = s.replace(",", "")
        else:
            # Banyak koma: 1,000,000 → ribuan
            s = s.replace(",", "")
    elif last_dot > 0 and last_comma < 0:
        # Hanya titik
        parts = s.split(".")
        if len(parts) == 2:
            # Satu titik: asumsikan desimal
            pass
        else:
            # Banyak titik: 1.000.000 → ribuan Indonesia
            s = s.replace(".", "")
    return float(s)


def calc_mdape(actual, predicted):
    """Median Absolute Percentage Error — lebih robust terhadap outlier dibanding MAPE."""
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.median(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def uji_stasioneritas(series, sp=None):
    """Uji stasioneritas non-musiman (ADF & KPSS) dan musiman (ADF pada data diff musiman).

    Returns dict with keys:
        adf_stat, adf_pvalue, adf_stasioner,
        kpss_stat, kpss_pvalue, kpss_stasioner,
        seasonal_adf_stat, seasonal_adf_pvalue, seasonal_stasioner  (if sp given)
        rekomendasi_d, rekomendasi_D
    """
    result = {}
    y = np.array(series, dtype=float)
    y = y[~np.isnan(y)]

    # ── ADF Test (non-musiman) ──
    # H0: ada unit root (tidak stasioner)
    try:
        adf_out = adfuller(y, autolag="AIC")
        result["adf_stat"] = round(adf_out[0], 4)
        result["adf_pvalue"] = round(adf_out[1], 4)
        result["adf_stasioner"] = adf_out[1] < 0.05
    except Exception:
        result["adf_stat"] = np.nan
        result["adf_pvalue"] = np.nan
        result["adf_stasioner"] = None

    # ── KPSS Test (konfirmasi) ──
    # H0: data stasioner (kebalikan ADF)
    try:
        kpss_out = kpss(y, regression="c", nlags="auto")
        result["kpss_stat"] = round(kpss_out[0], 4)
        result["kpss_pvalue"] = round(kpss_out[1], 4)
        result["kpss_stasioner"] = kpss_out[1] >= 0.05  # gagal tolak H0 = stasioner
    except Exception:
        result["kpss_stat"] = np.nan
        result["kpss_pvalue"] = np.nan
        result["kpss_stasioner"] = None

    # ── Rekomendasi d ──
    if result["adf_stasioner"] is True and result["kpss_stasioner"] is True:
        result["rekomendasi_d"] = 0
        result["kesimpulan_nonseasonal"] = "Stasioner (ADF ✅ & KPSS ✅)"
    elif result["adf_stasioner"] is False and result["kpss_stasioner"] is False:
        result["rekomendasi_d"] = 1
        result["kesimpulan_nonseasonal"] = "Tidak stasioner (ADF ❌ & KPSS ❌) → perlu differencing (d=1)"
    elif result["adf_stasioner"] is True and result["kpss_stasioner"] is False:
        result["rekomendasi_d"] = 0
        result["kesimpulan_nonseasonal"] = "Kemungkinan stasioner tapi ada tren deterministik (ADF ✅, KPSS ❌)"
    elif result["adf_stasioner"] is False and result["kpss_stasioner"] is True:
        result["rekomendasi_d"] = 1
        result["kesimpulan_nonseasonal"] = "Kemungkinan unit root (ADF ❌, KPSS ✅) → perlu differencing (d=1)"
    else:
        result["rekomendasi_d"] = 1
        result["kesimpulan_nonseasonal"] = "Tidak dapat ditentukan — default d=1"

    # ── Uji musiman (ADF pada data setelah differencing musiman) ──
    if sp is not None and len(y) > 2 * sp:
        # Cek apakah data musiman stasioner dengan ADF pada seasonal diff
        try:
            seasonal_diff = y[sp:] - y[:-sp]
            seasonal_diff = seasonal_diff[~np.isnan(seasonal_diff)]
            if len(seasonal_diff) >= 8:
                sadf = adfuller(seasonal_diff, autolag="AIC")
                result["seasonal_adf_stat"] = round(sadf[0], 4)
                result["seasonal_adf_pvalue"] = round(sadf[1], 4)
                result["seasonal_stasioner"] = sadf[1] < 0.05
            else:
                result["seasonal_adf_stat"] = np.nan
                result["seasonal_adf_pvalue"] = np.nan
                result["seasonal_stasioner"] = None
        except Exception:
            result["seasonal_adf_stat"] = np.nan
            result["seasonal_adf_pvalue"] = np.nan
            result["seasonal_stasioner"] = None

        # Uji ADF langsung pada data asli untuk cek unit root musiman
        # Jika seasonal diff stasioner → musiman sudah stasioner → D=0
        # Jika seasonal diff tidak stasioner → perlu D=1
        if result.get("seasonal_stasioner") is True:
            result["rekomendasi_D"] = 0
            result["kesimpulan_seasonal"] = (
                "Setelah differencing musiman, data stasioner → "
                "pola musiman tidak memerlukan differencing tambahan (D=0)"
            )
        elif result.get("seasonal_stasioner") is False:
            result["rekomendasi_D"] = 1
            result["kesimpulan_seasonal"] = (
                "Setelah differencing musiman, data masih tidak stasioner → "
                "diperlukan differencing musiman (D=1)"
            )
        else:
            result["rekomendasi_D"] = 0
            result["kesimpulan_seasonal"] = "Data tidak cukup panjang untuk uji musiman — default D=0"
    else:
        result["rekomendasi_D"] = None
        result["kesimpulan_seasonal"] = "Tidak berlaku (bukan data musiman)"

    return result


def calc_mase(actual, predicted, train_series):
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).
    Membandingkan MAE model terhadap MAE naïve in-sample.
    MASE < 1 berarti model lebih baik dari Naïve.
    """
    actual, predicted = np.array(actual), np.array(predicted)
    mae_model = np.mean(np.abs(actual - predicted))

    # MAE dari naïve in-sample: |Y_t - Y_{t-1}|
    train_series = np.array(train_series)
    naive_errors = np.abs(np.diff(train_series))
    mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        return np.nan  # data konstan, naïve sempurna
    return mae_model / mae_naive


# ── 1. Naïve ──
def run_naive(train, test):
    """Naïve: forecast = last observed value (baseline)."""
    last = train[-1]
    in_pred = np.array([np.nan] + list(train[:-1]))  # F(t) = Y(t-1)
    in_pred = in_pred[1:]
    in_actual = train[1:]

    out_pred = np.array([last] * len(test))
    residuals = in_actual - in_pred

    def fc(h):
        pf = [last] * h
        lo, hi = residual_ci(residuals, pf, h, 0.95)
        lo80, hi80 = residual_ci(residuals, pf, h, 0.80)
        return pf, lo, hi, lo80, hi80

    return {
        "name": "Naïve (Baseline)",
        "in_actual": in_actual, "in_pred": in_pred,
        "out_actual": test, "out_pred": out_pred,
        "forecast_func": fc, "residuals": residuals,
    }








# ── Fractional Seasonal Differencing ──

def frac_diff_weights(d, max_lag):
    """Hitung bobot fractional differencing (1-B^s)^d menggunakan ekspansi binomial."""
    w = [1.0]
    for k in range(1, max_lag):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)


def frac_nonseasonal_diff(series, d):
    """Terapkan fractional differencing (1 - B)^d pada series (lag=1).
    d: orde differencing pecahan (misal 0.3, 0.5)
    Mengembalikan series yang sudah di-difference dan offset.
    """
    return frac_seasonal_diff(series, d, 1)


def frac_nonseasonal_undiff(diffed_values, original_series, d):
    """Inverse fractional differencing (1 - B)^d — kembalikan ke skala asli."""
    return frac_seasonal_undiff(diffed_values, original_series, d, 1)


def frac_seasonal_diff(series, d, s):
    """Terapkan fractional seasonal differencing (1 - B^s)^d pada series.
    d: orde differencing (bisa pecahan, misal 0.1..0.9)
    s: periode musiman (misal 4 untuk triwulanan)
    Mengembalikan series yang sudah di-difference dan indeks mulai yang valid.
    """
    n = len(series)
    max_terms = n // s + 1  # Jumlah maks suku dalam ekspansi
    weights = frac_diff_weights(d, max_terms)
    result = np.full(n, np.nan)
    for t in range(n):
        val = 0.0
        valid = True
        for k in range(len(weights)):
            idx = t - k * s
            if idx < 0:
                break
            val += weights[k] * series[idx]
        if t >= 0:
            result[t] = val
    # Potong leading NaN
    first_valid = 0
    for i in range(n):
        if not np.isnan(result[i]):
            first_valid = i
            break
    return result[first_valid:], first_valid


def frac_seasonal_undiff(diffed_values, original_series, d, s):
    """Inverse fractional seasonal differencing — kembalikan ke skala asli.
    Gunakan untuk mengkonversi prediksi kembali ke level asli.
    """
    n_orig = len(original_series)
    n_pred = len(diffed_values)
    weights = frac_diff_weights(d, n_orig // s + n_pred // s + 2)

    result = np.zeros(n_pred)
    # Gabungkan original_series (sebagai riwayat) + result (yang sedang dihitung)
    full = np.concatenate([original_series, np.zeros(n_pred)])

    for t in range(n_pred):
        # diffed_values[t] = sum_k weights[k] * full[n_orig + t - k*s]
        # Jadi: full[n_orig + t] = (diffed_values[t] - sum_{k>=1} weights[k] * full[n_orig + t - k*s]) / weights[0]
        val = diffed_values[t]
        for k in range(1, len(weights)):
            idx = (n_orig + t) - k * s
            if idx < 0:
                break
            val -= weights[k] * full[idx]
        full[n_orig + t] = val / weights[0]
        result[t] = full[n_orig + t]
    return result


def _frac_insample_undiff(fitted_diffed, actual_original, d, s):
    """Undiff in-sample: gunakan nilai AKTUAL sebagai history di setiap langkah.

    Ini berbeda dari frac_seasonal_undiff (untuk forecast) yang menggunakan
    nilai prediksi sendiri sebagai history untuk langkah berikutnya.

    fitted_diffed[t] ≈ Σ_k w[k] * actual_original[t - k*s]
    → actual_hat[t] = (fitted_diffed[t] - Σ_{k≥1} w[k] * actual[t-k*s]) / w[0]
    """
    n = len(fitted_diffed)
    weights = frac_diff_weights(d, n // s + 2)
    result = np.zeros(n)
    for t in range(n):
        val = fitted_diffed[t]
        for k in range(1, len(weights)):
            idx = t - k * s
            if idx < 0:
                break
            val -= weights[k] * actual_original[idx]
        result[t] = val / weights[0]
    return result


def run_sarima_frac_d(train, test, order=(1, 0, 1), seasonal_pqd=(1, 1), sp=4,
                      frac_D=None, frac_d=None):
    """ARFIMA/SARFIMA: ARMA dengan fractional differencing.

    Langkah:
    1) Terapkan (1-B^s)^D lalu (1-B)^d pada data training.
    2) Fit ARMA(p,q) × (P,Q)_s pada data yang sudah di-difference.
    3) Inverse-transform prediksi kembali ke level asli.

    In-sample: undiff menggunakan nilai aktual sebagai history (one-step-ahead).
    Out-of-sample: undiff menggunakan actual train sebagai history awal,
                   lalu prediksi sendiri sebagai history berikutnya.
    """
    try:
        train_arr = np.array(train, dtype=float)
        test_arr = np.array(test, dtype=float)

        # ── Langkah 1: Fractional differencing pada training ──
        # Urutan: seasonal (D) dulu, lalu non-seasonal (d)
        step = train_arr.copy()
        train_after_D = None
        if frac_D is not None and frac_D > 0:
            step, _ = frac_seasonal_diff(step, frac_D, sp)
            train_after_D = step.copy()
        if frac_d is not None and frac_d > 0:
            step, _ = frac_nonseasonal_diff(step, frac_d)
        train_diffed = step

        if len(train_diffed) < 6:
            return None

        # ── Langkah 2: Fit ARMA pada data yang sudah di-difference ──
        P, Q = seasonal_pqd
        p, _, q = order

        model = SARIMAX(train_diffed, order=(p, 0, q),
                        seasonal_order=(P, 0, Q, sp),
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)

        fitted_d = np.array(model.fittedvalues)
        forecast_d = np.array(model.get_forecast(len(test_arr)).predicted_mean)

        # ── Langkah 3a: In-sample undiff (one-step-ahead) ──
        # Gunakan nilai AKTUAL sebagai history, bukan prediksi sendiri.
        # Urutan undiff: non-seasonal (d) dulu, lalu seasonal (D).
        in_pred = fitted_d.copy()
        if frac_d is not None and frac_d > 0:
            # Undiff nonseasonal: history = actual after_D values
            actual_after_D = train_after_D if train_after_D is not None else train_arr
            in_pred = _frac_insample_undiff(in_pred, actual_after_D, frac_d, 1)
        if frac_D is not None and frac_D > 0:
            # Undiff seasonal: history = actual original train values
            in_pred = _frac_insample_undiff(in_pred, train_arr, frac_D, sp)

        in_pred_final = in_pred
        residuals = train_arr - in_pred_final

        # ── Langkah 3b: Out-of-sample undiff (forecast) ──
        # Gunakan actual train sebagai history awal.
        # Urutan undiff: non-seasonal (d) dulu, lalu seasonal (D).
        out_pred = forecast_d.copy()
        if frac_d is not None and frac_d > 0:
            # History = actual train setelah seasonal diff (at after_D level)
            history_d = train_after_D if train_after_D is not None else train_arr
            out_pred = frac_nonseasonal_undiff(out_pred, history_d, frac_d)
        if frac_D is not None and frac_D > 0:
            # History = actual train (at original level)
            out_pred = frac_seasonal_undiff(out_pred, train_arr, frac_D, sp)

        # ── Buat nama model ──
        d_label = f"{frac_d:g}" if frac_d is not None and frac_d > 0 else "0"
        D_label = f"{frac_D:g}" if frac_D is not None and frac_D > 0 else "0"
        is_frac = (frac_d is not None and frac_d > 0 and frac_d != int(frac_d)) or \
                  (frac_D is not None and frac_D > 0 and frac_D != int(frac_D))
        no_seasonal = (P == 0 and Q == 0 and (frac_D is None or frac_D < 0.01))
        if is_frac:
            prefix = "ARFIMA" if no_seasonal else "SARFIMA"
        else:
            prefix = "ARIMA" if no_seasonal else "SARIMA"
        if no_seasonal:
            name = f"{prefix}({p},{d_label},{q})"
        else:
            name = f"{prefix}({p},{d_label},{q})x({P},{D_label},{Q},{sp})"

        def fc(h):
            fc_all = model.get_forecast(len(test_arr) + h)
            pred_d_fc = np.array(fc_all.predicted_mean[len(test_arr):])
            combined = np.concatenate([train_arr, test_arr])
            pf = pred_d_fc.copy()
            if frac_d is not None and frac_d > 0:
                hist_d = combined.copy()
                if frac_D is not None and frac_D > 0:
                    hist_d, _ = frac_seasonal_diff(combined, frac_D, sp)
                pf = frac_nonseasonal_undiff(pf, hist_d, frac_d)
            if frac_D is not None and frac_D > 0:
                pf = frac_seasonal_undiff(pf, combined, frac_D, sp)
            pf = list(pf)
            lo, hi = residual_ci(residuals, pf, h, 0.95)
            lo80, hi80 = residual_ci(residuals, pf, h, 0.80)
            return pf, lo, hi, lo80, hi80

        return {
            "name": name,
            "in_actual": train_arr, "in_pred": in_pred_final,
            "out_actual": test_arr, "out_pred": out_pred[:len(test_arr)],
            "forecast_func": fc, "residuals": residuals,
        }
    except Exception:
        return None


# ── 14. SARIMA ──
def run_sarima(train, test, order=(1, 1, 1), seasonal_order_s=None):
    if seasonal_order_s is None:
        return None
    try:
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order_s,
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        in_pred = model.fittedvalues
        fc_obj = model.get_forecast(len(test))
        out_pred = fc_obj.predicted_mean
        residuals = np.array(train) - np.array(in_pred)

        def fc(h):
            fc_o = model.get_forecast(h)
            pf = list(fc_o.predicted_mean)
            try:
                ci95 = fc_o.conf_int(alpha=0.05)
                ci80 = fc_o.conf_int(alpha=0.20)
                lo = np.array(ci95.iloc[:, 0])
                hi = np.array(ci95.iloc[:, 1])
                lo80 = np.array(ci80.iloc[:, 0])
                hi80 = np.array(ci80.iloc[:, 1])
            except Exception:
                lo, hi = residual_ci(residuals, pf, h, 0.95)
                lo80, hi80 = residual_ci(residuals, pf, h, 0.80)
            return pf, lo, hi, lo80, hi80

        P_s, D_s, Q_s, sp_s = seasonal_order_s
        p_o, d_o, q_o = order
        # Deteksi fractional d atau D
        d_is_frac = isinstance(d_o, float) and d_o != int(d_o) and d_o > 0
        D_is_frac = isinstance(D_s, float) and D_s != int(D_s) and D_s > 0
        is_frac = d_is_frac or D_is_frac
        d_str = f"{d_o:g}" if d_is_frac else str(int(d_o))
        D_str = f"{D_s:g}" if D_is_frac else str(int(D_s))
        no_seasonal = (P_s == 0 and (D_s == 0 or D_s < 0.01) and Q_s == 0)
        if no_seasonal:
            prefix = "ARFIMA" if is_frac else "ARIMA"
            sarima_label = f"{prefix}({p_o},{d_str},{q_o})"
        else:
            prefix = "SARFIMA" if is_frac else "SARIMA"
            sarima_label = f"{prefix}({p_o},{d_str},{q_o})x({P_s},{D_str},{Q_s},{sp_s})"
        return {
            "name": sarima_label,
            "in_actual": train, "in_pred": np.array(in_pred),
            "out_actual": test, "out_pred": np.array(out_pred),
            "forecast_func": fc, "residuals": residuals,
        }
    except Exception:
        return None


# ── 17. Neural Network (MLP) ──

def run_mlp(train, test, lookback=3, hidden=(64,), activation='relu', max_iter=500):
    """MLPRegressor (scikit-learn) untuk peramalan time series.
    Fitur: sliding window [Y_{t-lb}..Y_{t-1}] + ΔY (differencing) + rolling mean.
    Total fitur per sampel = lookback * 3 (raw + diff + rolling mean).
    """
    try:
        combined = np.array(list(train) + list(test), dtype=float).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(combined).flatten()

        # Buat fitur yang diperkaya: raw window + differencing + rolling mean
        def build_features(seq, lookback):
            """Dari sequence 1D, buat X dan y dengan fitur tambahan."""
            X, y = [], []
            for i in range(lookback, len(seq)):
                window = seq[i - lookback:i]
                # Fitur 1: raw sliding window
                feat_raw = list(window)
                # Fitur 2: differencing (ΔY) dalam window
                feat_diff = [0.0] + [window[j] - window[j - 1] for j in range(1, lookback)]
                # Fitur 3: rolling mean (expanding mean sampai posisi j dalam window)
                feat_rmean = [np.mean(window[:j + 1]) for j in range(lookback)]
                X.append(feat_raw + feat_diff + feat_rmean)
                y.append(seq[i])
            return np.array(X), np.array(y)

        def build_single_feature(window):
            """Buat vektor fitur dari satu window."""
            feat_raw = list(window)
            feat_diff = [0.0] + [window[j] - window[j - 1] for j in range(1, len(window))]
            feat_rmean = [np.mean(window[:j + 1]) for j in range(len(window))]
            return np.array(feat_raw + feat_diff + feat_rmean)

        scaled_train = scaled[:len(train)]

        X_train, y_train = build_features(scaled_train, lookback)
        if len(X_train) < 3:
            return None

        # Build & fit MLPRegressor
        model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=activation,
            solver='adam',
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # In-sample predictions
        in_pred_scaled = model.predict(X_train)
        in_pred_full = np.full(len(train), np.nan)
        in_pred_unscaled = scaler.inverse_transform(in_pred_scaled.reshape(-1, 1)).flatten()
        in_pred_full[lookback:] = in_pred_unscaled
        if lookback > 0 and len(in_pred_unscaled) > 0:
            in_pred_full[:lookback] = in_pred_unscaled[0]

        # Out-of-sample predictions (recursive)
        last_seq = list(scaled_train[-lookback:])
        out_pred_scaled = []
        for _ in range(len(test)):
            window = np.array(last_seq[-lookback:])
            x_input = build_single_feature(window).reshape(1, -1)
            pred_val = model.predict(x_input)[0]
            out_pred_scaled.append(pred_val)
            last_seq.append(pred_val)

        out_pred = scaler.inverse_transform(np.array(out_pred_scaled).reshape(-1, 1)).flatten()
        residuals = np.array(train) - in_pred_full

        h_str = "x".join(str(h) for h in hidden)
        label = f"MLP(lb={lookback}, h={h_str}, {activation})"

        # Closure for forecast
        _model = model
        _scaler = scaler
        _lookback = lookback
        _last_known = list(scaled[:len(train) + len(test)])

        def fc(h):
            seq = list(_last_known[-_lookback:])
            preds_scaled = []
            for _ in range(h):
                window = np.array(seq[-_lookback:])
                x_in = build_single_feature(window).reshape(1, -1)
                p_val = _model.predict(x_in)[0]
                preds_scaled.append(p_val)
                seq.append(p_val)
            pf = list(_scaler.inverse_transform(
                np.array(preds_scaled).reshape(-1, 1)).flatten())
            lo, hi = residual_ci(residuals, pf, h, 0.95)
            lo80, hi80 = residual_ci(residuals, pf, h, 0.80)
            return pf, lo, hi, lo80, hi80

        return {
            "name": label,
            "in_actual": train, "in_pred": in_pred_full,
            "out_actual": test, "out_pred": out_pred,
            "forecast_func": fc, "residuals": residuals,
        }
    except Exception:
        return None


def _get_lookbacks(frequency):
    """Tentukan kandidat lookback berdasarkan frekuensi data."""
    lb_map = {
        "Harian": [3, 5, 7, 14],
        "Mingguan": [2, 4, 8],
        "Bulanan": [3, 6, 12],
        "Triwulanan": [2, 3, 4],
        "Semesteran": [2, 3, 4],
        "Tahunan": [2, 3, 4],
    }
    return lb_map.get(frequency, [2, 3, 4])


def auto_best_mlp(train, test, frequency="Triwulanan"):
    """Coba beberapa kombinasi hyperparameter MLP, pilih MDAPE terkecil.

    Lookback disesuaikan dengan frekuensi data.
    Hidden layer: single layer [6, 12, 18, 24, 30].
    """
    lookbacks = _get_lookbacks(frequency)
    hidden_list = [(6,), (12,), (18,), (24,), (30,)]
    activations = ['relu', 'tanh']

    best, best_cfg = None, (lookbacks[0], (12,), 'relu')
    for lb in lookbacks:
        if lb >= len(train) - 2:
            continue
        for hid in hidden_list:
            for act in activations:
                try:
                    r = run_mlp(train, test, lookback=lb, hidden=hid,
                                activation=act, max_iter=300)
                    if r is not None:
                        mdape = calc_mdape(r["out_actual"], r["out_pred"])
                        if not np.isnan(mdape) and (best is None or mdape < best):
                            best, best_cfg = mdape, (lb, hid, act)
                except Exception:
                    continue
    return best_cfg


# ── 15b. MLPX — MLP dengan variabel eksogen ──
def run_mlpx(train, test, exog_train, exog_test,
             lookback=3, hidden=(12,), activation='relu', max_iter=500,
             exog_col_names=None, all_exog_col_names=None):
    """MLPRegressor dengan variabel eksogen.

    Fitur: sliding window + ΔY + rolling mean + variabel eksogen pada waktu t.
    all_exog_col_names: daftar lengkap nama kolom eksogen (untuk subsetting exog_future).
    """
    try:
        combined = np.array(list(train) + list(test), dtype=float).reshape(-1, 1)
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaled_y = scaler_y.fit_transform(combined).flatten()

        exog_all = np.vstack([exog_train, exog_test])
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaled_x = scaler_x.fit_transform(exog_all)
        n_exog_cols = scaled_x.shape[1]

        def build_features(seq_y, exog_scaled, lookback):
            X, y = [], []
            for i in range(lookback, len(seq_y)):
                window = seq_y[i - lookback:i]
                feat_raw = list(window)
                feat_diff = [0.0] + [window[j] - window[j-1] for j in range(1, lookback)]
                feat_rmean = [np.mean(window[:j+1]) for j in range(lookback)]
                feat_exog = list(exog_scaled[i])
                X.append(feat_raw + feat_diff + feat_rmean + feat_exog)
                y.append(seq_y[i])
            return np.array(X), np.array(y)

        def build_single_feature(window, exog_row):
            feat_raw = list(window)
            feat_diff = [0.0] + [window[j] - window[j-1] for j in range(1, len(window))]
            feat_rmean = [np.mean(window[:j+1]) for j in range(len(window))]
            feat_exog = list(exog_row)
            return np.array(feat_raw + feat_diff + feat_rmean + feat_exog)

        scaled_train_y = scaled_y[:len(train)]
        scaled_train_x = scaled_x[:len(train)]

        X_train, y_train = build_features(scaled_train_y, scaled_train_x, lookback)
        if len(X_train) < 3:
            return None

        model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=activation,
            solver='adam',
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # In-sample
        in_pred_scaled = model.predict(X_train)
        in_pred_full = np.full(len(train), np.nan)
        in_pred_unscaled = scaler_y.inverse_transform(in_pred_scaled.reshape(-1, 1)).flatten()
        in_pred_full[lookback:] = in_pred_unscaled
        if lookback > 0 and len(in_pred_unscaled) > 0:
            in_pred_full[:lookback] = in_pred_unscaled[0]

        # Out-of-sample
        last_seq = list(scaled_train_y[-lookback:])
        scaled_test_x = scaled_x[len(train):]
        out_pred_scaled = []
        for t in range(len(test)):
            window = np.array(last_seq[-lookback:])
            x_input = build_single_feature(window, scaled_test_x[t]).reshape(1, -1)
            pred_val = model.predict(x_input)[0]
            out_pred_scaled.append(pred_val)
            last_seq.append(pred_val)

        out_pred = scaler_y.inverse_transform(np.array(out_pred_scaled).reshape(-1, 1)).flatten()
        residuals = np.array(train) - in_pred_full

        h_str = "x".join(str(hh) for hh in hidden)
        if exog_col_names:
            exog_label = ",".join(exog_col_names)
        else:
            exog_label = f"{n_exog_cols}var"
        label = f"MLPX(lb={lookback}, h={h_str}, {activation}, [{exog_label}])"

        _model = model
        _scaler_y = scaler_y
        _scaler_x = scaler_x
        _lookback = lookback
        _last_known = list(scaled_y[:len(train) + len(test)])
        _exog_col_names = exog_col_names or []
        _all_exog_col_names = all_exog_col_names or []

        def fc(h, exog_future=None):
            if exog_future is None:
                # Fallback tanpa exog: gunakan exog terakhir
                last_exog = scaled_x[-1:]
                exog_fut_scaled = np.repeat(last_exog, h, axis=0)
            else:
                ef = np.array(exog_future, dtype=float)
                # Jika exog_future memiliki lebih banyak kolom, subset ke kolom yang dipakai
                if ef.ndim == 2 and len(_all_exog_col_names) > 0 and ef.shape[1] > len(_exog_col_names):
                    col_idx = [_all_exog_col_names.index(c) for c in _exog_col_names
                               if c in _all_exog_col_names]
                    if col_idx:
                        ef = ef[:, col_idx]
                exog_fut_scaled = _scaler_x.transform(ef)

            seq = list(_last_known[-_lookback:])
            preds_scaled = []
            for t in range(h):
                window = np.array(seq[-_lookback:])
                x_in = build_single_feature(window, exog_fut_scaled[t]).reshape(1, -1)
                p_val = _model.predict(x_in)[0]
                preds_scaled.append(p_val)
                seq.append(p_val)
            pf = list(_scaler_y.inverse_transform(
                np.array(preds_scaled).reshape(-1, 1)).flatten())
            lo, hi = residual_ci(residuals, pf, h, 0.95)
            lo80, hi80 = residual_ci(residuals, pf, h, 0.80)
            return pf, lo, hi, lo80, hi80

        return {
            "name": label,
            "in_actual": train, "in_pred": in_pred_full,
            "out_actual": test, "out_pred": out_pred,
            "forecast_func": fc, "residuals": residuals,
            "is_mlpx": True,
            "mlpx_exog_cols": exog_col_names or [],
        }
    except Exception:
        return None


def auto_best_mlpx(train, test, exog_train, exog_test,
                   frequency="Triwulanan", exog_col_names=None):
    """Auto-tuning MLPX: coba semua kombinasi hyperparameter + subset eksogen.

    Membandingkan penggunaan seluruh variabel eksogen vs setiap variabel tunggal,
    lalu memilih kombinasi dengan MDAPE terkecil.
    """
    lookbacks = _get_lookbacks(frequency)
    hidden_list = [(6,), (12,), (18,), (24,), (30,)]
    activations = ['relu', 'tanh']

    n_cols = exog_train.shape[1] if exog_train.ndim > 1 else 1

    # Susun daftar subset eksogen: seluruh + masing-masing variabel tunggal
    exog_subsets = []
    if n_cols > 1:
        # Seluruh variabel
        all_names = exog_col_names if exog_col_names else [f"X{i+1}" for i in range(n_cols)]
        exog_subsets.append((exog_train, exog_test, all_names))
        # Masing-masing variabel tunggal
        for c in range(n_cols):
            col_tr = exog_train[:, c:c+1]
            col_te = exog_test[:, c:c+1]
            c_name = [all_names[c]]
            exog_subsets.append((col_tr, col_te, c_name))
    else:
        c_names = exog_col_names if exog_col_names else ["X1"]
        if exog_train.ndim == 1:
            exog_subsets.append((exog_train.reshape(-1, 1), exog_test.reshape(-1, 1), c_names))
        else:
            exog_subsets.append((exog_train, exog_test, c_names))

    all_names = exog_col_names if exog_col_names else [f"X{i+1}" for i in range(n_cols)]
    best, best_result = None, None
    for ex_tr, ex_te, ex_names in exog_subsets:
        for lb in lookbacks:
            if lb >= len(train) - 2:
                continue
            for hid in hidden_list:
                for act in activations:
                    try:
                        r = run_mlpx(train, test, ex_tr, ex_te,
                                     lookback=lb, hidden=hid,
                                     activation=act, max_iter=300,
                                     exog_col_names=ex_names,
                                     all_exog_col_names=all_names)
                        if r is not None:
                            mdape = calc_mdape(r["out_actual"], r["out_pred"])
                            if not np.isnan(mdape) and (best is None or mdape < best):
                                best = mdape
                                best_result = r
                    except Exception:
                        continue

    return best_result


# ══════════════════════════════════════════════════════════════
# AUTO SARIMA ORDER
# ══════════════════════════════════════════════════════════════

def gph_estimate_d(series, alpha=0.65):
    """Estimasi parameter fractional differencing d menggunakan metode GPH
    (Geweke & Porter-Hudak, 1983).

    Regresi log-periodogram pada frekuensi rendah:
        log I(λ_j) = c − d · log(4 sin²(λ_j/2)) + ε_j
    di mana d = −slope.

    Parameters
    ----------
    series : array-like — data runtun waktu
    alpha  : float — eksponen bandwidth, m = n^alpha (default 0.65)

    Returns
    -------
    dict dengan 'd' (estimasi), 'se' (standard error), 'conf_lo', 'conf_hi'
    """
    from scipy import signal
    y = np.asarray(series, dtype=float)
    n = len(y)
    if n < 10:
        return {"d": 0.5, "se": np.nan, "conf_lo": 0.0, "conf_hi": 1.0}

    # Hitung periodogram
    freqs, psd = signal.periodogram(y, scaling="density")

    # Bandwidth: m frekuensi Fourier terendah
    m = max(3, int(n ** alpha))
    freqs_sel = freqs[1:m + 1]
    psd_sel = psd[1:m + 1]

    # Buang nilai PSD ≤ 0
    valid = psd_sel > 0
    freqs_sel = freqs_sel[valid]
    psd_sel = psd_sel[valid]

    if len(freqs_sel) < 3:
        return {"d": 0.5, "se": np.nan, "conf_lo": 0.0, "conf_hi": 1.0}

    # Regresi GPH
    omega = 2 * np.pi * freqs_sel
    X = np.log(4 * np.sin(omega / 2) ** 2)
    Y = np.log(psd_sel)

    slope, intercept, r_val, p_val, std_err = sp_stats.linregress(X, Y)
    d_hat = -slope

    # Clamp ke [0, 1]
    d_hat = float(np.clip(d_hat, 0.0, 1.0))

    # Confidence interval 95%
    conf_lo = max(0.0, d_hat - 1.96 * std_err)
    conf_hi = min(1.0, d_hat + 1.96 * std_err)

    return {"d": round(d_hat, 2), "se": float(std_err),
            "conf_lo": round(conf_lo, 2), "conf_hi": round(conf_hi, 2)}


def gph_estimate_seasonal_D(series, sp, alpha=0.65):
    """Estimasi parameter fractional seasonal differencing D menggunakan
    metode GPH pada frekuensi musiman (λ = 2π/s).

    Periodogram dievaluasi di sekitar frekuensi musiman, lalu regresi
    log-periodogram digunakan untuk memperkirakan D.

    Parameters
    ----------
    series : array-like — data runtun waktu
    sp     : int — periode musiman (misal 12 untuk bulanan)
    alpha  : float — eksponen bandwidth

    Returns
    -------
    dict dengan 'D', 'se', 'conf_lo', 'conf_hi'
    """
    from scipy import signal
    y = np.asarray(series, dtype=float)
    n = len(y)

    if sp <= 1 or n < 2 * sp:
        return {"D": 0.0, "se": np.nan, "conf_lo": 0.0, "conf_hi": 0.0}

    # Hitung periodogram
    freqs, psd = signal.periodogram(y, scaling="density")

    # Frekuensi musiman fundamental: 1/sp (dalam unit normalized freq)
    seasonal_freq = 1.0 / sp
    m = max(3, int(n ** alpha))

    # Ambil frekuensi di sekitar frekuensi musiman
    freq_diffs = np.abs(freqs - seasonal_freq)
    # Ambil m frekuensi terdekat dari frekuensi musiman
    closest_idx = np.argsort(freq_diffs)
    # Buang indeks 0 (frekuensi nol)
    closest_idx = closest_idx[closest_idx > 0][:m]
    closest_idx = np.sort(closest_idx)

    if len(closest_idx) < 3:
        return {"D": 0.0, "se": np.nan, "conf_lo": 0.0, "conf_hi": 0.0}

    freqs_sel = freqs[closest_idx]
    psd_sel = psd[closest_idx]

    valid = psd_sel > 0
    freqs_sel = freqs_sel[valid]
    psd_sel = psd_sel[valid]

    if len(freqs_sel) < 3:
        return {"D": 0.0, "se": np.nan, "conf_lo": 0.0, "conf_hi": 0.0}

    # Regresi GPH di sekitar frekuensi musiman
    omega = 2 * np.pi * freqs_sel
    omega_s = 2 * np.pi * seasonal_freq
    X = np.log(4 * np.sin((omega - omega_s) / 2) ** 2 + 1e-12)
    Y = np.log(psd_sel)

    try:
        slope, intercept, r_val, p_val, std_err = sp_stats.linregress(X, Y)
        D_hat = float(np.clip(-slope, 0.0, 1.0))
    except Exception:
        return {"D": 0.0, "se": np.nan, "conf_lo": 0.0, "conf_hi": 0.0}

    conf_lo = max(0.0, D_hat - 1.96 * std_err)
    conf_hi = min(1.0, D_hat + 1.96 * std_err)

    return {"D": round(D_hat, 2), "se": float(std_err),
            "conf_lo": round(conf_lo, 2), "conf_hi": round(conf_hi, 2)}


def auto_sarima_order(train, test, sp, max_p=2, max_d=1, max_q=2,
                      max_P=1, max_D=1, max_Q=1):
    """Pencarian order ARFIMA/SARFIMA optimal berbasis GPH + grid search.

    Strategi (Hyndman & Khandakar, 2008):
    1) Tetapkan d̂ (non-seasonal) dan D̂ (seasonal) via GPH.
    2) Grid search hanya pada (p, q, P, Q) dengan d dan D tetap.
    3) Fractional differencing dilakukan manual, ARMA fit pada data
       yang sudah di-difference.

    Referensi:
    - Geweke & Porter-Hudak (1983) — estimasi d via log-periodogram.
    - Hyndman & Khandakar (2008) — two-step: fix d, lalu search ARMA.

    Returns: (best_order, best_seasonal, frac_d, frac_D)
    """
    best_mdape = np.inf
    best_order = (1, 0, 1)
    best_seasonal_pq = (0, 0) if sp <= 1 else (1, 1)

    # Adaptif berdasarkan panjang data
    n = len(train)
    if sp > 1:
        max_p = min(max_p, n // (2 * sp))
        max_q = min(max_q, n // (2 * sp))
    else:
        max_P, max_D, max_Q = 0, 0, 0
    max_P = min(max_P, 1)
    max_Q = min(max_Q, 1)

    # ── Estimasi GPH: d̂ dan D̂ ──
    gph_d = gph_estimate_d(train)
    d_hat = gph_d["d"]

    gph_D_result = gph_estimate_seasonal_D(train, sp) if sp > 1 else {"D": 0.0}
    D_hat = gph_D_result["D"]

    # ── Kandidat d: tuning penuh 0.1–0.9 ──
    # Eksplorasi seluruh range fractional d karena GPH bisa noisy
    d_candidates = [round(x * 0.1, 1) for x in range(1, 10)]  # 0.1, 0.2, ..., 0.9

    # ── Kandidat D berdasarkan uji stasioneritas musiman ──
    if sp > 1 and D_hat > 0:
        try:
            seasonal_diff = np.array(train[sp:]) - np.array(train[:-sp])
            sadf_p = adfuller(seasonal_diff, autolag="AIC")[1]
            is_seasonal_stationary = sadf_p < 0.05
        except Exception:
            is_seasonal_stationary = False

        if is_seasonal_stationary:
            D_compare = 0.0
        else:
            D_compare = 0.1 if D_hat < 0.5 else 0.9

        D_candidates = sorted(set([round(max(0.0, D_hat), 2),
                                    round(max(0.0, D_compare), 2)]))
    else:
        D_candidates = [0.0]

    best_frac_d = d_hat if d_hat > 0 else None
    best_frac_D = D_hat if sp > 1 and D_hat > 0 else None

    # Grid search

    # ── Grid search pada (p, q, P, Q) × kandidat d × kandidat D ──
    for d_try in d_candidates:
        fd = d_try if d_try > 0 else None
        for D_try in D_candidates:
            fD = D_try if D_try > 0 else None
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    for P in range(max_P + 1):
                        for Q in range(max_Q + 1):
                            if p == 0 and q == 0 and P == 0 and Q == 0:
                                continue
                            try:
                                r = run_sarima_frac_d(
                                    train, test,
                                    order=(p, 0, q),
                                    seasonal_pqd=(P, Q),
                                    sp=sp,
                                    frac_D=fD,
                                    frac_d=fd)
                                if r is None:
                                    continue
                                mdape = calc_mdape(test, np.array(r["out_pred"]))
                                if np.isnan(mdape):
                                    continue
                                if mdape < best_mdape:
                                    best_mdape = mdape
                                    best_order = (p, 0, q)
                                    best_seasonal_pq = (P, Q)
                                    best_frac_d = fd
                                    best_frac_D = fD
                            except Exception:
                                continue

    return best_order, best_seasonal_pq, best_frac_d, best_frac_D



# ── 16. SARIMAX (SARIMA + variabel eksogen) ──
def run_sarimax(train, test, exog_train, exog_test, order=(1, 1, 1),
                seasonal_order_s=None, exog_col_names=None,
                all_exog_col_names=None):
    """SARIMAX — SARIMA dengan variabel eksogen."""
    if seasonal_order_s is None or exog_train is None:
        return None
    try:
        model = SARIMAX(train, exog=exog_train, order=order,
                        seasonal_order=seasonal_order_s,
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)
        in_pred = model.fittedvalues
        fc_obj = model.get_forecast(len(test), exog=exog_test)
        out_pred = fc_obj.predicted_mean
        residuals = np.array(train) - np.array(in_pred)

        # Store model and exog info for forecasting
        stored_model = model
        n_exog = exog_train.shape[1] if exog_train.ndim > 1 else 1
        _exog_col_names = exog_col_names or []
        _all_exog_col_names = all_exog_col_names or []

        def fc(h, exog_future=None):
            if exog_future is None:
                # Fallback: gunakan residual CI jika tidak ada exog future
                pf = list(stored_model.forecast(h))
                lo, hi = residual_ci(residuals, pf, h, 0.95)
                lo80, hi80 = residual_ci(residuals, pf, h, 0.80)
                return pf, lo, hi, lo80, hi80
            ef = np.array(exog_future, dtype=float)
            # Subset kolom jika exog_future memiliki lebih banyak kolom
            if ef.ndim == 2 and len(_all_exog_col_names) > 0 and ef.shape[1] > len(_exog_col_names):
                col_idx = [_all_exog_col_names.index(c) for c in _exog_col_names
                           if c in _all_exog_col_names]
                if col_idx:
                    ef = ef[:, col_idx]
            fc_o = stored_model.get_forecast(h, exog=ef)
            pf = list(fc_o.predicted_mean)
            try:
                ci95 = fc_o.conf_int(alpha=0.05)
                ci80 = fc_o.conf_int(alpha=0.20)
                lo = np.array(ci95.iloc[:, 0])
                hi = np.array(ci95.iloc[:, 1])
                lo80 = np.array(ci80.iloc[:, 0])
                hi80 = np.array(ci80.iloc[:, 1])
            except Exception:
                lo, hi = residual_ci(residuals, pf, h, 0.95)
                lo80, hi80 = residual_ci(residuals, pf, h, 0.80)
            return pf, lo, hi, lo80, hi80

        P_s, D_s, Q_s, sp_s = seasonal_order_s
        p_o, d_o, q_o = order
        d_str = f"{d_o:g}" if not isinstance(d_o, int) and d_o != int(d_o) else str(int(d_o))
        D_str = f"{D_s:g}" if not isinstance(D_s, int) and D_s != int(D_s) else str(int(D_s))
        no_seasonal = (P_s == 0 and D_s == 0 and Q_s == 0)
        # Tambahkan info variabel eksogen terpilih di label
        if exog_col_names:
            exog_tag = f" [{','.join(exog_col_names)}]"
        else:
            exog_tag = f" [{n_exog}var]"
        if no_seasonal:
            model_label = f"ARIMAX({p_o},{d_str},{q_o}){exog_tag}"
        else:
            model_label = f"SARIMAX({p_o},{d_str},{q_o})x({P_s},{D_str},{Q_s},{sp_s}){exog_tag}"
        # Simpan estimasi parameter (koefisien, std error, z, p-value)
        try:
            param_df = pd.DataFrame({
                "Parameter": model.params.index,
                "Koefisien": model.params.values,
                "Std. Error": model.bse.values,
                "z": model.zvalues.values,
                "P>|z|": model.pvalues.values,
            })
        except Exception:
            param_df = None

        return {
            "name": model_label,
            "in_actual": train, "in_pred": np.array(in_pred),
            "out_actual": test, "out_pred": np.array(out_pred),
            "forecast_func": fc, "residuals": residuals,
            "is_sarimax": True, "n_exog": n_exog,
            "sarimax_exog_cols": exog_col_names or [],
            "param_table": param_df,
        }
    except Exception as e:
        logging.warning(f"run_sarimax failed: {e}")
        return None


def auto_sarimax_order(train, test, exog_train, exog_test, sp,
                       max_p=2, max_d=1, max_q=2,
                       max_P=1, max_D=1, max_Q=1,
                       exog_col_names=None,
                       sarima_best_order=None, sarima_best_seasonal_pq=None):
    """Pencarian order SARIMAX optimal berbasis tuning SARIMA + grid search MDAPE.

    Strategi:
    1) Gunakan parameter SARIMA terbaik (dari auto_sarima_order) sebagai
       kandidat prioritas — komponen ARMA sudah optimal untuk data endogen.
    2) Tambahkan variasi d=0 (tanpa differencing) karena variabel eksogen
       sering menggantikan peran differencing dalam menangkap tren.
    3) Grid search pada (p, q, P, Q) × kandidat d × subset eksogen.
    4) Bandingkan subset eksogen: seluruh variabel + masing-masing individu.

    Returns: (best_order, best_seasonal, best_exog_train, best_exog_test, best_exog_names)
    """
    best_mdape = np.inf
    best_order = (1, 0, 1)
    best_seasonal = (0, 0, 0, sp) if sp <= 1 else (1, 0, 1, sp)
    best_ex_tr = exog_train
    best_ex_te = exog_test
    best_ex_names = exog_col_names

    n = len(train)
    if sp > 1:
        max_p = min(max_p, n // (2 * sp))
        max_q = min(max_q, n // (2 * sp))
    else:
        max_P, max_D, max_Q = 0, 0, 0
    max_P = min(max_P, 1)
    max_Q = min(max_Q, 1)

    # ── Estimasi d: integer, prioritas d=0 untuk model dengan eksogen ──
    gph_d = gph_estimate_d(train)
    d_gph = int(round(gph_d["d"]))
    d_gph = max(0, min(d_gph, max_d))

    gph_D_result = gph_estimate_seasonal_D(train, sp) if sp > 1 else {"D": 0.0}
    D_gph = int(round(gph_D_result["D"])) if sp > 1 else 0
    D_gph = max(0, min(D_gph, max_D))

    # Kandidat d dan D: selalu sertakan 0 (eksogen sering menggantikan differencing)
    d_candidates = sorted(set([0, d_gph]))
    D_candidates = sorted(set([0, D_gph]))

    # ── Bangun daftar kombinasi (p, q, P, Q) yang akan dicoba ──
    # Prioritas 1: parameter SARIMA terbaik (sudah di-tuning untuk data endogen)
    # Prioritas 2: variasi sekitar parameter SARIMA terbaik
    # Prioritas 3: grid search standar
    priority_pqPQ = []

    if sarima_best_order is not None and sarima_best_seasonal_pq is not None:
        s_p, _, s_q = sarima_best_order
        s_P, s_Q = sarima_best_seasonal_pq
        # Parameter SARIMA terbaik (tanpa d, karena d ditentukan terpisah)
        priority_pqPQ.append((s_p, s_q, s_P, s_Q))
        # Variasi ±1 dari parameter terbaik
        for dp in [-1, 0, 1]:
            for dq in [-1, 0, 1]:
                pp, qq = s_p + dp, s_q + dq
                if 0 <= pp <= max_p and 0 <= qq <= max_q and (pp, qq) != (s_p, s_q):
                    priority_pqPQ.append((pp, qq, s_P, s_Q))
        # Juga coba tanpa komponen musiman
        if s_P > 0 or s_Q > 0:
            priority_pqPQ.append((s_p, s_q, 0, 0))

    # Grid search standar (semua kombinasi)
    grid_pqPQ = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for P in range(max_P + 1):
                for Q in range(max_Q + 1):
                    if p == 0 and q == 0 and P == 0 and Q == 0:
                        continue
                    grid_pqPQ.append((p, q, P, Q))

    # Gabung: prioritas dulu, lalu grid (tanpa duplikat)
    seen = set()
    all_pqPQ = []
    for combo in priority_pqPQ + grid_pqPQ:
        if combo not in seen:
            seen.add(combo)
            all_pqPQ.append(combo)

    # ── Susun subset eksogen: seluruh + masing-masing individu ──
    n_cols = exog_train.shape[1] if exog_train.ndim > 1 else 1
    all_names = exog_col_names if exog_col_names else [f"X{i+1}" for i in range(n_cols)]
    exog_subsets = []
    if n_cols > 1:
        exog_subsets.append((exog_train, exog_test, all_names))
        for c in range(n_cols):
            col_tr = exog_train[:, c:c+1]
            col_te = exog_test[:, c:c+1]
            exog_subsets.append((col_tr, col_te, [all_names[c]]))
    else:
        if exog_train.ndim == 1:
            exog_subsets.append((exog_train.reshape(-1, 1), exog_test.reshape(-1, 1), all_names))
        else:
            exog_subsets.append((exog_train, exog_test, all_names))

    # ── Grid search pada (p, d, q, P, D, Q) × subset eksogen ──
    for ex_tr, ex_te, ex_names in exog_subsets:
        for d_use in d_candidates:
            for D_use in D_candidates:
                for p, q, P, Q in all_pqPQ:
                    try:
                        m = SARIMAX(train, exog=ex_tr,
                                    order=(p, d_use, q),
                                    seasonal_order=(P, D_use, Q, sp),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit(disp=False)
                        out_pred = m.get_forecast(len(test), exog=ex_te).predicted_mean
                        mdape = calc_mdape(test, np.array(out_pred))
                        if not np.isnan(mdape) and mdape < best_mdape:
                            best_mdape = mdape
                            best_order = (p, d_use, q)
                            best_seasonal = (P, D_use, Q, sp)
                            best_ex_tr = ex_tr
                            best_ex_te = ex_te
                            best_ex_names = ex_names
                    except Exception:
                        continue
    return best_order, best_seasonal, best_ex_tr, best_ex_te, best_ex_names


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

FREQ_SEASONAL = {
    "Harian": 7,
    "Mingguan": 52,
    "Bulanan": 12,
    "Triwulanan": 4,
    "Semesteran": None,
    "Tahunan": None,
}

FREQ_MIN_DATA = {
    "Harian": 30, "Mingguan": 16, "Bulanan": 24,
    "Triwulanan": 12, "Semesteran": 8, "Tahunan": 8,
}


# ══════════════════════════════════════════════════════════════
# MODEL DESCRIPTIONS
# ══════════════════════════════════════════════════════════════

MODEL_INFO = {
    "Naïve": {
        "desc": "Menggunakan nilai observasi terakhir sebagai ramalan untuk semua periode ke depan. "
                "Model paling sederhana yang sering dijadikan *baseline* pembanding.",
        "pros": [
            "Sangat sederhana, tidak perlu estimasi parameter",
            "Cocok sebagai baseline — jika model lain tidak bisa mengalahkan Naïve, berarti data sulit diprediksi",
            "Bekerja cukup baik pada data yang bersifat *random walk* (misal harga saham harian)",
        ],
        "cons": [
            "Tidak menangkap tren maupun pola musiman",
            "Ramalan selalu datar (flat) untuk semua horizon",
            "Sangat sensitif terhadap outlier pada observasi terakhir",
        ],
        "assumptions": [
            "Data mengikuti proses *random walk* — nilai terbaik untuk masa depan adalah nilai saat ini",
            "Tidak ada tren, musiman, atau pola sistematis dalam data",
            "Perubahan antar periode bersifat acak (*white noise*)",
        ],
        "formula": [
            r"\hat{Y}_{t+h} = Y_t",
        ],
        "symbols": {
            r"$\hat{Y}_{t+h}$": "Nilai ramalan pada $h$ periode ke depan dari waktu $t$",
            r"$Y_t$": "Nilai observasi terakhir (pada waktu $t$)",
            r"$h$": "Horizon ramalan (jumlah periode ke depan)",
        },
    },
    "SARIMA": {
        "desc": "Seasonal ARIMA — perluasan ARIMA dengan komponen musiman tambahan (P,D,Q,s) "
                "menggunakan D integer. Hanya tersedia untuk data dengan frekuensi musiman.",
        "pros": [
            "Menangkap tren DAN pola musiman secara bersamaan",
            "Confidence interval analitik yang akurat",
            "Sangat cocok untuk data triwulanan dan bulanan dengan pola berulang",
            "Interpretasi parameter yang jelas dan standar",
        ],
        "cons": [
            "Membutuhkan data minimal 2× periode musiman",
            "Estimasi parameter bisa lambat (grid search)",
            "Lebih banyak parameter — risiko overfitting pada data pendek",
            "Interpretasi parameter lebih kompleks dari Holt-Winters",
        ],
        "assumptions": [
            "Data harus stasioner setelah differencing non-musiman ($d$) dan musiman ($D$)",
            "Pola musiman memiliki periode tetap dan diketahui ($s$)",
            "Error bersifat *white noise* — independen, rata-rata nol, variansi konstan",
            "Error berdistribusi normal (untuk validitas confidence interval)",
            "Data minimal 2× periode musiman agar komponen musiman teridentifikasi",
            "Hubungan autoregressive (AR dan SAR) bersifat linier",
        ],
        "formula": [
            r"(1 - \phi_1 B - \cdots - \phi_p B^p)(1 - \Phi_1 B^s - \cdots - \Phi_P B^{Ps})(1 - B)^d(1 - B^s)^D Y_t",
            r"\quad = c + (1 + \theta_1 B + \cdots + \theta_q B^q)(1 + \Theta_1 B^s + \cdots + \Theta_Q B^{Qs})\varepsilon_t",
        ],
        "symbols": {
            r"$\phi_i$": "Koefisien AR non-musiman (orde $p$)",
            r"$\Phi_j$": "Koefisien AR musiman (orde $P$)",
            r"$\theta_i$": "Koefisien MA non-musiman (orde $q$)",
            r"$\Theta_j$": "Koefisien MA musiman (orde $Q$)",
            r"$d$": "Orde differencing non-musiman",
            r"$D$": "Orde differencing musiman (integer: 0 atau 1)",
            r"$s$": "Panjang periode musiman (misal 4 untuk triwulanan)",
            r"$B$": "Operator backshift: $BY_t = Y_{t-1}$, $B^sY_t = Y_{t-s}$",
            r"$\varepsilon_t$": "White noise error",
            r"$(p,d,q)(P,D,Q)_s$": "Notasi lengkap order SARIMA",
        },
    },
    "SARFIMA": {
        "desc": "Seasonal ARFIMA (Fractionally Integrated) — perluasan SARIMA dengan orde differencing "
                "pecahan pada d dan/atau D (kelipatan 0.1, dari 0.1 s/d 0.9). "
                "Menangkap long memory yang lebih fleksibel dari differencing integer.",
        "pros": [
            "Menangkap long memory (lebih fleksibel dari d/D integer)",
            "Menangani data yang 'antara stasioner dan non-stasioner'",
            "Cocok untuk data dengan autokorelasi yang meluruh perlahan",
            "Diferensiasi lebih halus — mengurangi over-differencing",
        ],
        "cons": [
            "Membutuhkan data minimal 2× periode musiman",
            "Estimasi lebih lambat (grid search d + D integer + pecahan)",
            "Lebih kompleks — risiko overfitting pada data pendek",
            "Inverse transform manual menambah potensi error numerik kecil",
        ],
        "assumptions": [
            "Data memiliki long memory (autokorelasi meluruh lambat)",
            "Pola musiman memiliki periode tetap dan diketahui ($s$)",
            "Error bersifat *white noise* — independen, rata-rata nol, variansi konstan",
            "Error berdistribusi normal (untuk validitas confidence interval)",
            "Hubungan autoregressive bersifat linier",
        ],
        "formula": [
            r"(1 - \phi_1 B - \cdots - \phi_p B^p)(1 - \Phi_1 B^s - \cdots - \Phi_P B^{Ps})(1 - B)^d(1 - B^s)^D Y_t",
            r"\quad = c + (1 + \theta_1 B + \cdots + \theta_q B^q)(1 + \Theta_1 B^s + \cdots + \Theta_Q B^{Qs})\varepsilon_t",
            r"\text{dengan } d, D \in \{0.1, 0.2, \ldots, 0.9\} \text{ (pecahan)}",
        ],
        "symbols": {
            r"$d$": "Orde differencing non-musiman (integer atau pecahan 0.1..0.9)",
            r"$D$": "Orde differencing musiman (integer atau pecahan 0.1..0.9), menangkap long memory",
            r"$(1-B)^d$": "Operator fractional differencing non-musiman via ekspansi binomial",
            r"$(1-B^s)^D$": "Operator fractional differencing musiman via ekspansi binomial",
            r"$\phi_i$": "Koefisien AR non-musiman (orde $p$)",
            r"$\Phi_j$": "Koefisien AR musiman (orde $P$)",
            r"$\theta_i$": "Koefisien MA non-musiman (orde $q$)",
            r"$\Theta_j$": "Koefisien MA musiman (orde $Q$)",
            r"$s$": "Panjang periode musiman",
            r"$B$": "Operator backshift",
        },
    },
    "SARIMAX": {
        "desc": "SARIMAX — perluasan SARIMA/SARFIMA dengan variabel eksogen (prediktor eksternal). "
                "Menggabungkan komponen ARIMA, musiman, dan pengaruh variabel luar untuk peramalan yang lebih akurat.",
        "pros": [
            "Dapat memanfaatkan informasi dari variabel prediktor eksternal",
            "Menggabungkan kekuatan SARIMA/SARFIMA (tren + musiman) dengan regresi",
            "Confidence interval analitik yang memperhitungkan semua komponen",
            "Cocok ketika ada faktor luar yang diketahui mempengaruhi variabel target",
        ],
        "cons": [
            "Membutuhkan nilai variabel eksogen untuk periode ramalan ke depan",
            "Lebih banyak parameter — risiko overfitting lebih tinggi",
            "Estimasi lebih lambat dari SARIMA/SARFIMA biasa",
            "Kualitas ramalan bergantung pada ketersediaan dan kualitas data eksogen",
        ],
        "assumptions": [
            "Asumsi SARIMA/SARFIMA tetap berlaku (stasioneritas setelah differencing, white noise error)",
            "Hubungan antara variabel eksogen dan target bersifat linier",
            "Variabel eksogen tidak memiliki multikolinearitas sempurna",
            "Nilai variabel eksogen di periode ramalan harus diketahui/diperkirakan",
            "Error berdistribusi normal (untuk validitas confidence interval)",
        ],
        "formula": [
            r"(1 - \phi_1 B - \cdots - \phi_p B^p)(1 - \Phi_1 B^s - \cdots - \Phi_P B^{Ps})(1 - B)^d(1 - B^s)^D Y_t",
            r"\quad = c + \beta_1 X_{1,t} + \cdots + \beta_k X_{k,t} + (1 + \theta_1 B + \cdots + \theta_q B^q)(1 + \Theta_1 B^s + \cdots + \Theta_Q B^{Qs})\varepsilon_t",
        ],
        "symbols": {
            r"$\beta_i$": "Koefisien regresi untuk variabel eksogen ke-$i$",
            r"$X_{i,t}$": "Nilai variabel eksogen ke-$i$ pada waktu $t$",
            r"$k$": "Jumlah variabel eksogen",
            r"$\phi_i, \Phi_j$": "Koefisien AR non-musiman dan musiman",
            r"$\theta_i, \Theta_j$": "Koefisien MA non-musiman dan musiman",
            r"$d, D$": "Orde differencing non-musiman dan musiman",
            r"$\varepsilon_t$": "White noise error",
        },
    },
    "Neural Network (MLP)": {
        "desc": "Multi-Layer Perceptron — jaringan saraf tiruan feedforward yang mempelajari pola non-linier "
                "dalam data deret waktu melalui sliding window. Hyperparameter (lookback, hidden layers, "
                "activation) dipilih otomatis berdasarkan MDAPE terkecil. Menggunakan scikit-learn MLPRegressor.",
        "pros": [
            "Mampu menangkap pola non-linier yang kompleks",
            "Ringan dan cepat — tidak memerlukan TensorFlow/GPU",
            "Hasil deterministik (reproducible) dengan random_state tetap",
            "Fleksibel — bisa mempelajari berbagai pola dari data",
            "Sudah termasuk dalam scikit-learn (tanpa dependensi tambahan)",
        ],
        "cons": [
            "Membutuhkan data yang lebih banyak dibanding model statistik tradisional",
            "Tidak menangkap dependensi temporal secara eksplisit (berbeda dengan RNN/LSTM)",
            "Sulit diinterpretasikan (black box) dibanding model statistik",
            "Performa sangat bergantung pada pemilihan lookback window yang tepat",
        ],
        "assumptions": [
            "Pola dalam data historis akan berlanjut di masa depan",
            "Data telah dinormalisasi (dilakukan otomatis via MinMaxScaler)",
            "Jumlah data cukup untuk pelatihan neural network (minimal 6 titik)",
            "Lookback window yang tepat menentukan seberapa jauh ke belakang model melihat",
            "Hubungan antar waktu bisa bersifat non-linier",
        ],
        "formula": [
            r"z^{(l)} = W^{(l)} \cdot a^{(l-1)} + b^{(l)}",
            r"a^{(l)} = g(z^{(l)})",
            r"\text{ReLU}(z) = \max(0, z)",
            r"\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}",
            r"\hat{y} = W^{(L)} \cdot a^{(L-1)} + b^{(L)}",
            r"L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2",
        ],
        "symbols": {
            r"$z^{(l)}$": "Pre-activation pada layer ke-$l$",
            r"$a^{(l)}$": "Aktivasi (output) pada layer ke-$l$",
            r"$W^{(l)}, b^{(l)}$": "Bobot dan bias pada layer ke-$l$ (dipelajari selama pelatihan)",
            r"$g(\cdot)$": "Fungsi aktivasi (ReLU atau tanh)",
            r"$\hat{y}$": "Output prediksi dari layer terakhir",
            r"$L$": "Loss function (Mean Squared Error)",
            r"$n$": "Jumlah sampel pelatihan",
        },
    },
    "MLPX": {
        "desc": "MLP with eXogenous variables — perluasan Neural Network MLP yang menyertakan variabel eksogen "
                "sebagai fitur tambahan selain sliding window. Secara otomatis membandingkan penggunaan seluruh "
                "variabel eksogen vs. masing-masing variabel individu, dan memilih kombinasi terbaik berdasarkan MDAPE.",
        "pros": [
            "Menangkap pengaruh non-linier dari variabel eksogen terhadap variabel target",
            "Otomatis memilih subset variabel eksogen terbaik (seluruh vs. individu)",
            "Mampu mempelajari interaksi kompleks antara lagged values dan variabel eksogen",
            "Tetap ringan — menggunakan scikit-learn MLPRegressor",
        ],
        "cons": [
            "Membutuhkan data eksogen yang tersedia juga untuk periode ramalan",
            "Lebih banyak fitur — risiko overfitting pada data pendek",
            "Sulit menginterpretasikan kontribusi masing-masing variabel (black box)",
            "Kualitas sangat bergantung pada relevansi variabel eksogen yang diberikan",
        ],
        "assumptions": [
            "Variabel eksogen memiliki pengaruh (langsung atau tidak langsung) terhadap variabel target",
            "Nilai variabel eksogen diketahui atau bisa diprediksi untuk periode ramalan",
            "Data telah dinormalisasi (dilakukan otomatis via MinMaxScaler terpisah untuk Y dan exog)",
            "Hubungan antara variabel target dan eksogen bisa bersifat non-linier",
        ],
        "formula": [
            r"\mathbf{x}_t = [\underbrace{Y_{t-1},\ldots,Y_{t-lb}}_{\text{sliding window}},\;"
            r"\underbrace{\Delta Y_t, \bar{Y}_t}_{\text{fitur tambahan}},\;"
            r"\underbrace{X_{1,t},\ldots,X_{k,t}}_{\text{variabel eksogen}}]",
            r"\hat{Y}_t = f_{\text{MLP}}(\mathbf{x}_t)",
        ],
        "symbols": {
            r"$\mathbf{x}_t$": "Vektor fitur pada waktu $t$",
            r"$Y_{t-i}$": "Nilai target pada $i$ langkah sebelumnya (sliding window)",
            r"$\Delta Y_t$": "Differencing: $Y_t - Y_{t-1}$",
            r"$\bar{Y}_t$": "Rolling mean (rata-rata bergerak)",
            r"$X_{j,t}$": "Variabel eksogen ke-$j$ pada waktu $t$",
            r"$lb$": "Lookback — jumlah langkah ke belakang",
            r"$k$": "Jumlah variabel eksogen",
        },
    },
}


# ══════════════════════════════════════════════════════════════
# AUTO-SELECT BEST PARAMETERS
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
# RUN ALL MODELS
# ══════════════════════════════════════════════════════════════

def run_all_models(values, frequency, train_ratio=0.8,
                   exog_data=None, use_sarimax=False,
                   max_p=3, max_q=3, max_P=1, max_Q=1,
                   exog_col_names=None):
    n = len(values)
    warnings_list = []

    if n < 4:
        st.error("Data terlalu sedikit. Minimal 4 titik data.")
        return None

    min_rec = FREQ_MIN_DATA.get(frequency, 8)
    if n < min_rec:
        warnings_list.append(
            f"⚠️ Data ({n} titik) kurang dari rekomendasi untuk {frequency} ({min_rec} titik). "
            f"Beberapa model mungkin dilewati."
        )

    split = max(int(n * train_ratio), 3)
    split = min(split, n - 2)
    if split < 3:
        split = min(3, n - 2)

    train = np.array(values[:split], dtype=float)
    test = np.array(values[split:], dtype=float)
    sp = FREQ_SEASONAL.get(frequency)

    # Split exogenous data if provided
    exog_train, exog_test = None, None
    if use_sarimax and exog_data is not None:
        exog_arr = np.array(exog_data, dtype=float)
        exog_train = exog_arr[:split]
        exog_test = exog_arr[split:]

    results = []
    skipped = []

    # 1. Naïve
    r = run_naive(train, test)
    if r: results.append(r)

    # ── Uji stasioneritas (untuk informasi di Tab 2) ──
    stationarity_info = uji_stasioneritas(train, sp=sp)

    # ── Estimasi GPH untuk fractional d dan D (informasi di Tab 2) ──
    gph_d_info = gph_estimate_d(train)
    gph_D_info = gph_estimate_seasonal_D(train, sp if sp is not None else 1) if sp is not None and sp > 1 else None
    stationarity_info["gph_d"] = gph_d_info
    stationarity_info["gph_D"] = gph_D_info

    # 2. ARFIMA / SARFIMA — d̂ dan D̂ dari GPH (Hyndman & Khandakar, 2008)
    #    Fractional d dan D langsung diteruskan ke statsmodels SARIMAX.
    #    Untuk data non-musiman (sp=None), gunakan sp=1 → ARFIMA (P=Q=0)
    sarima_sp = sp if sp is not None else 1
    sarima_min_data = (2 * sp) if sp is not None else 4
    _sarima_best_order = None      # Akan diisi parameter SARIMA terbaik
    _sarima_best_seasonal_pq = None  # untuk dipakai SARIMAX sebagai titik awal
    if len(train) >= sarima_min_data:
        try:
            s_order, s_seasonal_pq, s_frac_d, s_frac_D = auto_sarima_order(
                train, test, sarima_sp,
                max_p=max_p, max_q=max_q,
                max_P=max_P, max_Q=max_Q)
            # Simpan parameter SARIMA terbaik untuk dipakai SARIMAX
            _sarima_best_order = s_order
            _sarima_best_seasonal_pq = s_seasonal_pq
            r = run_sarima_frac_d(
                train, test, order=s_order,
                seasonal_pqd=s_seasonal_pq, sp=sarima_sp,
                frac_D=s_frac_D, frac_d=s_frac_d)
            if r: results.append(r)
            else: skipped.append("ARFIMA/SARFIMA")
        except Exception:
            skipped.append("ARFIMA/SARFIMA")

    # 3. SARIMAX — only if user opted in and provided exog data
    #    Untuk data non-musiman (sp=None), gunakan sp=1 → ARIMAX (P=D=Q=0)
    #    Menggunakan parameter SARIMA terbaik sebagai titik awal tuning.
    sarimax_sp = sp if sp is not None else 1
    sarimax_min_data = (2 * sp) if sp is not None else 4
    if use_sarimax and exog_train is not None and len(train) >= sarimax_min_data:
        try:
            sx_order, sx_seasonal, sx_ex_tr, sx_ex_te, sx_ex_names = auto_sarimax_order(
                train, test, exog_train, exog_test, sarimax_sp,
                max_p=max_p, max_q=max_q,
                max_P=max_P, max_Q=max_Q,
                exog_col_names=exog_col_names,
                sarima_best_order=_sarima_best_order,
                sarima_best_seasonal_pq=_sarima_best_seasonal_pq)
            r = run_sarimax(train, test, sx_ex_tr, sx_ex_te,
                            order=sx_order, seasonal_order_s=sx_seasonal,
                            exog_col_names=sx_ex_names,
                            all_exog_col_names=exog_col_names)
            if r:
                results.append(r)
            else:
                # Fallback: coba beberapa parameter sederhana tanpa differencing
                fallback_configs = [
                    ((1, 0, 1), (0, 0, 0, sarimax_sp)),  # ARIMAX sederhana
                    ((1, 0, 0), (0, 0, 0, sarimax_sp)),  # AR(1) + exog
                    ((0, 0, 1), (0, 0, 0, sarimax_sp)),  # MA(1) + exog
                    ((2, 0, 1), (0, 0, 0, sarimax_sp)),  # ARIMAX(2,0,1)
                ]
                if sarimax_sp > 1:
                    fallback_configs.append(
                        ((1, 0, 1), (1, 0, 0, sarimax_sp)))  # + SAR(1)
                r_fallback = None
                for fb_order, fb_seasonal in fallback_configs:
                    try:
                        r2 = run_sarimax(train, test, exog_train, exog_test,
                                         order=fb_order,
                                         seasonal_order_s=fb_seasonal,
                                         exog_col_names=exog_col_names,
                                         all_exog_col_names=exog_col_names)
                        if r2:
                            r_fallback = r2
                            break
                    except Exception:
                        continue
                if r_fallback:
                    results.append(r_fallback)
                else:
                    skipped.append("SARIMAX (semua kombinasi parameter gagal — "
                                   "kemungkinan data tidak cocok untuk model ini)")
        except Exception as e:
            skipped.append(f"SARIMAX ({e})")
    elif use_sarimax and exog_train is None:
        skipped.append("SARIMAX (data eksogen tidak tersedia)")

    # 4. Neural Network (MLP) — auto-select best hyperparameters
    if len(train) >= 6:
        try:
            best_lb, best_hid, best_act = auto_best_mlp(train, test, frequency=frequency)
            r = run_mlp(train, test, lookback=best_lb, hidden=best_hid,
                        activation=best_act, max_iter=500)
            if r: results.append(r)
            else: skipped.append("Neural Network (MLP)")
        except Exception:
            skipped.append("Neural Network (MLP)")
    else:
        skipped.append("Neural Network (MLP)")

    # 5. MLPX — MLP dengan variabel eksogen (jika tersedia)
    if use_sarimax and exog_train is not None and len(train) >= 6:
        try:
            r_mlpx = auto_best_mlpx(train, test, exog_train, exog_test,
                                    frequency=frequency,
                                    exog_col_names=exog_col_names)
            if r_mlpx:
                results.append(r_mlpx)
            else:
                skipped.append("MLPX (Neural Network + Eksogen)")
        except Exception:
            skipped.append("MLPX (Neural Network + Eksogen)")

    if not results:
        st.error("Tidak ada model yang berhasil dijalankan.")
        return None

    # Calculate metrics
    comparison = []
    for r in results:
        ia, ip = np.array(r["in_actual"]), np.array(r["in_pred"])
        mn = min(len(ia), len(ip))
        ia, ip = ia[-mn:], ip[-mn:]
        mask = ~(np.isnan(ia) | np.isnan(ip))
        ia, ip = ia[mask], ip[mask]

        oa, op = np.array(r["out_actual"]), np.array(r["out_pred"])
        mn2 = min(len(oa), len(op))
        oa, op = oa[:mn2], op[:mn2]
        mask2 = ~(np.isnan(oa) | np.isnan(op))
        oa, op = oa[mask2], op[mask2]

        rmse_out = calc_rmse(oa, op) if len(oa) > 0 else np.nan
        mape_out = calc_mape(oa, op) if len(oa) > 0 else np.nan
        mdape_out = calc_mdape(oa, op) if len(oa) > 0 else np.nan
        mase_out = calc_mase(oa, op, train) if len(oa) > 0 else np.nan

        comparison.append({
            "Model": r["name"],
            "MDAPE (%)": round(mdape_out, 2),
            "MASE": round(mase_out, 4),
            "MAPE (%)": round(mape_out, 2),
            "RMSE": round(rmse_out, 2),
        })

    comp_df = pd.DataFrame(comparison)
    return results, comp_df, (split, train, test), warnings_list, skipped, stationarity_info


# ══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════

def main():
    st.markdown("""
    <div class="rima-header">
        <h1>📊 PRIMA</h1>
        <p>Peramalan Interaktif dengan Model Adaptif</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Masukkan data → bandingkan model peramalan → dapatkan ramalan terbaik dengan confidence interval")

    # ── Sidebar: User & Dataset management ──
    with st.sidebar:
        # ── User section ──
        st.header("👤 Pengguna")
        users = list_users()  # each: (id, name, is_admin, created_at)

        with st.expander("➕ Tambah Pengguna Baru", expanded=len(users) == 0):
            new_user_name = st.text_input("Nama Pengguna", placeholder="contoh: Arifah")
            if st.button("Daftar", type="primary", use_container_width=True):
                if new_user_name.strip():
                    try:
                        create_user(new_user_name.strip())
                        first_msg = " (otomatis menjadi Admin)" if len(users) == 0 else ""
                        st.success(f"Pengguna '{new_user_name.strip()}' berhasil ditambahkan!{first_msg}")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.warning("Nama pengguna sudah ada. Silakan pilih nama lain.")
                else:
                    st.warning("Nama pengguna tidak boleh kosong.")

        if not users:
            st.info("Belum ada pengguna. Pengguna pertama otomatis menjadi Admin.")
            st.stop()

        # Build user labels with role indicator
        user_labels = {}
        for u in users:
            role_tag = " 👑" if u[2] == 1 else ""
            user_labels[f"{u[1]}{role_tag}"] = u[0]
        selected_user_label = st.selectbox("Pilih Pengguna", list(user_labels.keys()))
        selected_user_id = user_labels[selected_user_label]

        # Get current user's info
        current_user = get_user_info(selected_user_id)
        selected_user_name = current_user[1]
        is_current_admin = current_user[2] == 1

        if is_current_admin:
            st.caption("🔑 Anda login sebagai **Admin**")
        else:
            st.caption(f"👤 Login sebagai **{selected_user_name}**")

        # ── Admin Panel (only visible to admins) ──
        if is_current_admin:
            with st.expander("🔧 Panel Admin"):
                st.markdown("##### Kelola Pengguna")

                # List all users with admin toggle
                for u in users:
                    uid, uname, uadmin = u[0], u[1], u[2]
                    col_name, col_action = st.columns([2, 1])
                    with col_name:
                        badge = "👑 Admin" if uadmin else "👤 User"
                        st.markdown(f"**{uname}** — {badge}")
                    with col_action:
                        if uid == selected_user_id:
                            st.caption("(Anda)")
                        elif uadmin:
                            if st.button("Cabut Admin", key=f"revoke_{uid}",
                                         use_container_width=True):
                                # Don't allow revoking if this is the last admin
                                if count_admins() <= 1:
                                    st.warning("Tidak bisa cabut admin terakhir!")
                                else:
                                    set_admin(uid, False)
                                    st.rerun()
                        else:
                            if st.button("Jadikan Admin", key=f"grant_{uid}",
                                         use_container_width=True):
                                set_admin(uid, True)
                                st.rerun()

                st.divider()
                st.markdown("##### Hapus Pengguna")
                # Admin can delete other users (not themselves)
                other_users = [u for u in users if u[0] != selected_user_id]
                if other_users:
                    del_options = {u[1]: u[0] for u in other_users}
                    del_target = st.selectbox("Pilih pengguna untuk dihapus",
                                              list(del_options.keys()), key="del_user_select")
                    del_uid = del_options[del_target]
                    if st.button(f"🗑️ Hapus '{del_target}' beserta datanya",
                                 use_container_width=True):
                        delete_user(del_uid)
                        st.success(f"Pengguna '{del_target}' dan semua datasetnya dihapus.")
                        st.rerun()
                else:
                    st.caption("Tidak ada pengguna lain untuk dihapus.")

        st.divider()

        # ── Dataset section ──
        st.header("📁 Dataset")

        # Admin: option to view all users' datasets
        view_all = False
        if is_current_admin:
            view_all = st.checkbox("👁️ Lihat dataset semua pengguna", value=False,
                                   key="admin_view_all")

        if view_all:
            all_ds = list_all_datasets_with_owner()
            datasets = [(d[0], d[1], d[2], d[3], d[4], d[5]) for d in all_ds]
            if datasets:
                st.markdown(f"**Semua dataset** ({len(datasets)} total):")
                for d in all_ds:
                    n_pts = count_data_points(d[0])
                    _, exog_names_d = load_exog_data(d[0])
                    n_exog = len(exog_names_d)
                    exog_info = f" — eksogen: {n_exog}" if n_exog > 0 else " — eksogen: 0"
                    st.caption(f"• {d[1]} ({d[2]}) — {n_pts} data{exog_info} — *{d[6]}*")
            else:
                st.info("Belum ada dataset dari pengguna manapun.")
        else:
            datasets = list_datasets(user_id=selected_user_id)
            if datasets:
                st.markdown(f"**{selected_user_name}** memiliki **{len(datasets)}** dataset:")
                for d in datasets:
                    n_pts = count_data_points(d[0])
                    _, exog_names_d = load_exog_data(d[0])
                    n_exog = len(exog_names_d)
                    exog_info = f" — eksogen: {n_exog}" if n_exog > 0 else " — eksogen: 0"
                    st.caption(f"• {d[1]} ({d[2]}) — {n_pts} data{exog_info}")

        with st.expander("➕ Buat Dataset Baru", expanded=len(datasets) == 0):
            new_name = st.text_input("Nama Dataset", placeholder="contoh: Harga Beras 2020-2026")
            new_freq = st.selectbox("Frekuensi Data",
                                    ["Harian", "Mingguan", "Bulanan", "Triwulanan", "Semesteran", "Tahunan"])
            new_unit = st.text_input("Satuan (opsional)", placeholder="contoh: Rp/Kg")
            if st.button("Buat Dataset", type="primary", use_container_width=True):
                if new_name.strip():
                    create_dataset(new_name.strip(), new_freq, new_unit.strip(), user_id=selected_user_id)
                    st.success(f"Dataset '{new_name}' berhasil dibuat untuk {selected_user_name}!")
                    st.rerun()
                else:
                    st.warning("Nama dataset tidak boleh kosong.")

        if datasets:
            st.divider()
            if view_all:
                all_ds = list_all_datasets_with_owner()
                ds_options = {f"{d[1]} ({d[2]}) — {d[6]}": d[0] for d in all_ds}
            else:
                ds_options = {f"{d[1]} ({d[2]})": d[0] for d in datasets}
            selected_label = st.selectbox("Pilih Dataset", list(ds_options.keys()))
            selected_id = ds_options[selected_label]
            # Only admin can delete datasets
            if is_current_admin:
                if st.button("🗑️ Hapus Dataset Ini", use_container_width=True):
                    delete_dataset(selected_id)
                    st.success("Dataset dihapus.")
                    st.rerun()
        else:
            st.info(f"Belum ada dataset untuk {selected_user_name}. Buat dataset baru di atas.")
            st.stop()

    # ── Main area ──
    ds_info = get_dataset_info(selected_id)
    if ds_info is None:
        st.error("Dataset tidak ditemukan.")
        st.stop()

    ds_id, ds_name, ds_freq, ds_unit = ds_info
    st.session_state["selected_ds_id"] = ds_id
    unit_label = f" ({ds_unit})" if ds_unit else ""
    # Label sumbu Y: nama dataset + satuan, misal "Harga Beras (Ribu Rupiah/Kg)"
    ylabel_full = f"{ds_name}{unit_label}"

    # ── Muat data eksogen dari database jika belum ada di session_state ──
    if st.session_state.get("exog_data") is None or st.session_state.get("_exog_ds_id") != ds_id:
        saved_exog, saved_exog_names = load_exog_data(ds_id)
        if saved_exog is not None and len(saved_exog_names) > 0:
            st.session_state["exog_data"] = saved_exog
            st.session_state["exog_col_names"] = saved_exog_names
            st.session_state["use_sarimax"] = True
        else:
            st.session_state["use_sarimax"] = False
            st.session_state.pop("exog_data", None)
            st.session_state.pop("exog_col_names", None)
        # Muat juga data eksogen ramalan (future) dari database
        db_ef, db_ef_cols, db_ef_h = load_exog_future(ds_id)
        if db_ef is not None:
            st.session_state["_exog_future"] = db_ef
            st.session_state["_exog_future_h"] = db_ef_h
        st.session_state["_exog_ds_id"] = ds_id

    st.subheader(f"📋 {ds_name}")
    st.caption(f"Pengguna: **{selected_user_name}** | Frekuensi: **{ds_freq}** | Satuan: **{ds_unit or '-'}**")

    tab_input, tab_analysis, tab_forecast, tab_guide = st.tabs(
        ["📝 Input Data", "📊 Analisis & Perbandingan", "🔮 Hasil Ramalan", "📖 Panduan Model"])

    # ══════════════════════════════════════════════
    # TAB 1: Data Input
    # ══════════════════════════════════════════════
    with tab_input:
        col_upload, col_table = st.columns([1, 2])

        nilai_col_label = f"Nilai ({ds_unit})" if ds_unit else "Nilai"

        with col_upload:
            st.markdown("#### 📂 Upload File Data")
            st.caption("Upload satu file CSV atau Excel yang berisi kolom Periode, "
                       "kolom Nilai (Y), dan kolom variabel eksogen (jika ada).")
            uploaded = st.file_uploader("Upload CSV atau Excel", type=["csv", "xlsx", "xls"])
            if uploaded is not None:
                try:
                    if uploaded.name.endswith(".csv"):
                        try:
                            udf = pd.read_csv(uploaded, sep=None, engine="python")
                        except Exception:
                            uploaded.seek(0)
                            udf = pd.read_csv(uploaded)
                    else:
                        udf = pd.read_excel(uploaded)

                    st.dataframe(udf.head(), use_container_width=True)
                    ucols = list(udf.columns)
                    col_period = st.selectbox("Kolom Periode", ucols, index=0)
                    col_value = st.selectbox("Kolom Nilai (Y)", ucols, index=min(1, len(ucols) - 1))

                    # ── Eksogen: pilih kolom tambahan dari file yang sama ──
                    remaining_cols = [c for c in ucols if c != col_period and c != col_value]
                    use_sarimax = False
                    exog_selected = []
                    if remaining_cols:
                        use_sarimax = st.checkbox(
                            "🔗 Gunakan variabel eksogen (SARIMAX & MLPX)",
                            value=st.session_state.get("use_sarimax", False),
                            help="Centang jika kolom lain di file ini adalah variabel prediktor eksternal "
                                 "(misalnya: harga BBM, curah hujan, jumlah penduduk).",
                            key="use_sarimax_cb",
                        )
                        if use_sarimax:
                            sp_check = FREQ_SEASONAL.get(ds_freq)
                            if sp_check is None:
                                st.info("ℹ️ Frekuensi **" + ds_freq + "** tidak memiliki komponen musiman. "
                                        "Model akan berjalan sebagai **ARIMAX**.")
                            exog_selected = st.multiselect(
                                "Pilih kolom variabel eksogen",
                                remaining_cols,
                                default=remaining_cols,
                                key="exog_select_input",
                            )

                            # ── Input data eksogen untuk periode ramalan ──
                            if exog_selected:
                                st.markdown("---")
                                st.markdown("**📈 Data Eksogen untuk Periode Ramalan (opsional)**")
                                st.caption(
                                    "Isi nilai variabel eksogen untuk periode yang akan diramalkan. "
                                    "Jika belum diisi sekarang, bisa diisi nanti di tab Ramalan."
                                )
                                h_upload = st.number_input(
                                    "Jumlah periode ramalan",
                                    min_value=1, max_value=52, value=st.session_state.get("forecast_h", 4),
                                    key="exog_fc_h_upload",
                                )
                                exog_fc_upload_cols = []
                                exog_fc_upload_valid = True
                                for ei, ecol in enumerate(exog_selected):
                                    fc_input = st.text_input(
                                        f"Nilai **{ecol}** untuk {h_upload} periode ke depan (pisah koma)",
                                        placeholder=f"contoh: " + ", ".join([str(100 + j*5) for j in range(h_upload)]),
                                        key=f"exog_fc_upload_{ei}",
                                    )
                                    if fc_input.strip():
                                        try:
                                            raw_fc = fc_input.strip()
                                            if ";" in raw_fc or "\n" in raw_fc:
                                                raw_fc = raw_fc.replace("\n", ";").replace(",", ".")
                                                toks = [t.strip() for t in raw_fc.split(";") if t.strip()]
                                            else:
                                                toks = [t.strip() for t in raw_fc.split(",") if t.strip()]
                                            vals_upload = [parse_number(t) for t in toks]
                                            if len(vals_upload) != h_upload:
                                                st.error(f"❌ {ecol}: jumlah ({len(vals_upload)}) ≠ {h_upload} periode")
                                                exog_fc_upload_valid = False
                                            else:
                                                exog_fc_upload_cols.append(vals_upload)
                                        except ValueError:
                                            st.error(f"❌ {ecol}: data tidak valid.")
                                            exog_fc_upload_valid = False
                                    else:
                                        exog_fc_upload_valid = False

                    # Pastikan variabel exog forecast ada meskipun form tidak ditampilkan
                    if not (use_sarimax and exog_selected):
                        exog_fc_upload_valid = False
                        exog_fc_upload_cols = []
                        h_upload = st.session_state.get("forecast_h", 4)

                    if st.button("📥 Import Data", type="primary", use_container_width=True):
                        # Hapus data lama agar data Y dan eksogen tetap selaras
                        clear_data_points(ds_id)
                        records = []
                        for _, row in udf.iterrows():
                            try:
                                val = parse_number(row[col_value])
                                records.append((str(row[col_period]), val))
                            except (ValueError, TypeError):
                                continue
                        if records:
                            bulk_insert_points(ds_id, records)

                            # Simpan data eksogen ke session_state
                            if use_sarimax and exog_selected:
                                exog_from_file = udf[exog_selected].values.astype(float)
                                if np.isnan(exog_from_file).any():
                                    st.warning("⚠️ Data eksogen mengandung nilai kosong (NaN). "
                                               "SARIMAX/MLPX tidak akan diaktifkan.")
                                    st.session_state["use_sarimax"] = False
                                    st.session_state.pop("exog_data", None)
                                    st.session_state.pop("exog_col_names", None)
                                else:
                                    st.session_state["exog_data"] = exog_from_file
                                    st.session_state["exog_col_names"] = list(exog_selected)
                                    st.session_state["use_sarimax"] = True
                                    # Simpan eksogen ke database
                                    save_exog_data(ds_id, exog_from_file, list(exog_selected))
                            else:
                                st.session_state["use_sarimax"] = False
                                st.session_state.pop("exog_data", None)
                                st.session_state.pop("exog_col_names", None)
                                clear_exog_data(ds_id)

                            # Simpan data eksogen ramalan jika diisi
                            if (use_sarimax and exog_selected
                                    and st.session_state.get("use_sarimax")
                                    and exog_fc_upload_valid
                                    and len(exog_fc_upload_cols) == len(exog_selected)):
                                exog_future_arr = np.column_stack(exog_fc_upload_cols)
                                st.session_state["_exog_future"] = exog_future_arr
                                st.session_state["_exog_future_h"] = h_upload
                                st.session_state["forecast_h"] = h_upload
                                # Simpan ke database agar persisten
                                save_exog_future(ds_id, exog_future_arr,
                                                 list(exog_selected), h_upload)

                            exog_msg = ""
                            if use_sarimax and exog_selected and st.session_state.get("use_sarimax"):
                                exog_msg = (f" + {len(exog_selected)} variabel eksogen "
                                            f"({', '.join(exog_selected)})")
                                if (exog_fc_upload_valid
                                        and len(exog_fc_upload_cols) == len(exog_selected)):
                                    exog_msg += f" — data ramalan {h_upload} periode tersimpan ✅"
                            st.success(f"✅ {len(records)} data berhasil diimport!{exog_msg}")
                            st.rerun()
                        else:
                            st.error("Tidak ada data valid.")
                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")

            # Only admin can clear all data
            if is_current_admin:
                st.divider()
                if st.button("🗑️ Hapus Semua Data", use_container_width=True):
                    clear_data_points(ds_id)
                    st.session_state["use_sarimax"] = False
                    st.session_state.pop("exog_data", None)
                    st.session_state.pop("exog_col_names", None)
                    st.rerun()

        with col_table:
            st.markdown("#### Data Saat Ini")
            points = get_data_points(ds_id)
            if points:
                df_display = pd.DataFrame(points, columns=["ID", "Periode", "Nilai", "Urutan"])
                df_display = df_display.rename(columns={"Nilai": nilai_col_label})
                df_display = df_display[["Periode", nilai_col_label]]

                # Tambahkan kolom eksogen ke tabel jika ada
                exog_active = (st.session_state.get("use_sarimax")
                               and st.session_state.get("exog_data") is not None)
                if exog_active:
                    exog_data_display = st.session_state["exog_data"]
                    exog_names_display = st.session_state.get("exog_col_names", [])
                    if len(exog_data_display) == len(df_display):
                        for ci, cname in enumerate(exog_names_display):
                            if exog_data_display.ndim == 2 and ci < exog_data_display.shape[1]:
                                df_display[cname] = exog_data_display[:, ci]
                            elif exog_data_display.ndim == 1 and ci == 0:
                                df_display[cname] = exog_data_display

                df_display.index = range(1, len(df_display) + 1)
                df_display.index.name = "No"
                st.dataframe(df_display, use_container_width=True, height=400)
                st.caption(f"Total: **{len(points)}** data")

                # Info variabel eksogen
                if exog_active:
                    exog_names = st.session_state.get("exog_col_names", [])
                    st.info(f"🔗 **Variabel eksogen aktif ({len(exog_names)}):** "
                            f"{', '.join(exog_names)}\n\n"
                            f"Variabel Y: **{nilai_col_label}** — "
                            f"Variabel X: **{', '.join(exog_names)}**")

                chart_df = pd.DataFrame({
                    "Periode": [p[1] for p in points],
                    nilai_col_label: [p[2] for p in points],
                })
                st.line_chart(chart_df, x="Periode", y=nilai_col_label, use_container_width=True)

                # Download grafik Tab 1
                png_tab1 = make_line_chart_png(
                    periodes=[p[1] for p in points],
                    series_dict={nilai_col_label: [p[2] for p in points]},
                    title=f"Data {ds_name}",
                    ylabel=ylabel_full,
                )
                st.download_button(
                    "📥 Download Grafik (PNG)", data=png_tab1,
                    file_name=f"grafik_data_{ds_name}.png",
                    mime="image/png", key="dl_chart_tab1",
                )

                # Only admin can delete individual data points
                if is_current_admin:
                    with st.expander("Hapus data tertentu"):
                        del_options = {f"{p[1]} = {p[2]}": p[0] for p in points}
                        del_sel = st.multiselect("Pilih data yang akan dihapus", list(del_options.keys()))
                        if st.button("Hapus yang dipilih") and del_sel:
                            for s in del_sel:
                                delete_data_point(del_options[s])
                            st.rerun()
            else:
                st.info("Belum ada data. Upload file CSV atau Excel di sebelah kiri.")

    # ══════════════════════════════════════════════
    # TAB 2: Analysis
    # ══════════════════════════════════════════════
    with tab_analysis:
        points = get_data_points(ds_id)
        if len(points) < 4:
            st.warning(f"Minimal 4 titik data. Saat ini: {len(points)} titik.")
            st.stop()

        min_rec = FREQ_MIN_DATA.get(ds_freq, 8)
        if len(points) < min_rec:
            st.info(f"💡 Rekomendasi {min_rec} titik untuk {ds_freq}. "
                    f"Saat ini {len(points)} — analisis tetap berjalan, beberapa model mungkin dilewati.")

        values = [p[2] for p in points]
        labels = [p[1] for p in points]

        train_pct = st.slider("Rasio Training (%)", 50, 90, 80, 5)
        h = st.slider("Panjang Periode Ramalan", 1, 24, 4, 1,
                       help="Jumlah periode ke depan yang akan diramalkan.",
                       key="forecast_horizon")
        st.session_state["forecast_h"] = h

        # ── Pengaturan Tuning SARIMA / SARFIMA / SARIMAX ──
        with st.expander("⚙️ Pengaturan Tuning SARIMA/SARFIMA/SARIMAX", expanded=False):
            st.caption(
                "Tentukan batas atas parameter yang dicoba saat tuning otomatis. "
                "Semakin besar nilainya, semakin banyak kombinasi yang dicoba "
                "(hasil bisa lebih optimal, tapi waktu komputasi lebih lama)."
            )
            ar_col1, ar_col2, ar_col3, ar_col4 = st.columns(4)
            with ar_col1:
                user_max_p = st.number_input(
                    "Maks p (AR)", min_value=1, max_value=5, value=3, step=1,
                    help="Orde autoregressive (AR) maksimum. "
                         "Untuk SARIMA/SARFIMA: p=0..maks p. Default 3.")
            with ar_col2:
                user_max_q = st.number_input(
                    "Maks q (MA)", min_value=1, max_value=5, value=3, step=1,
                    help="Orde moving average (MA) maksimum. "
                         "Untuk SARIMA/SARFIMA: q=0..maks q. Default 3.")
            with ar_col3:
                user_max_P = st.number_input(
                    "Maks P (SAR)", min_value=1, max_value=3, value=1, step=1,
                    help="Orde seasonal AR maksimum untuk SARIMA/SARFIMA/SARIMAX. "
                         "P=0..maks P. Default 1.")
            with ar_col4:
                user_max_Q = st.number_input(
                    "Maks Q (SMA)", min_value=1, max_value=3, value=1, step=1,
                    help="Orde seasonal MA maksimum untuk SARIMA/SARFIMA/SARIMAX. "
                         "Q=0..maks Q. Default 1.")

            # Hitung estimasi kombinasi (d dan D tetap dari GPH, hanya p,q,P,Q yang di-search)
            n_arma = (user_max_p + 1) * (user_max_q + 1) * (user_max_P + 1) * (user_max_Q + 1)
            # d̂ vs pembanding (2 kandidat), D̂ vs pembanding (2 kandidat)
            n_sarima_combos = n_arma * 9 * 2
            st.info(
                f"📊 Estimasi kombinasi ARFIMA/SARFIMA ≈ {n_sarima_combos} "
                f"({n_arma} ARMA × 9 d × 2 D; "
                f"d: tuning 0.1–0.9, D: D̂ GPH vs pembanding uji stasioneritas). "
                f"Nilai lebih besar = lebih optimal tapi lebih lama."
            )

        # ── SARIMAX: Baca dari session_state (input di Tab Input Data) ──
        use_sarimax = st.session_state.get("use_sarimax", False)
        exog_data = st.session_state.get("exog_data", None)
        exog_col_names = st.session_state.get("exog_col_names", [])

        if use_sarimax and exog_data is not None:
            st.info(f"🔗 **SARIMAX/MLPX aktif** — {len(exog_col_names)} variabel eksogen: "
                    f"{', '.join(exog_col_names)}. "
                    f"(Pengaturan di tab **📝 Input Data**)")
        elif use_sarimax and exog_data is None:
            st.warning("⚠️ SARIMAX diaktifkan tetapi data eksogen belum lengkap. "
                       "Lengkapi di tab **📝 Input Data**.")

        n_models = 15 + (2 if use_sarimax and exog_data is not None else 0)
        if st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True):
            with st.spinner(f"Menjalankan {n_models} model dengan auto-tuning parameter (mungkin memakan waktu beberapa saat)..."):
                output = run_all_models(values, ds_freq, train_ratio=train_pct / 100,
                                        exog_data=exog_data, use_sarimax=use_sarimax,
                                        max_p=user_max_p, max_q=user_max_q,
                                        max_P=user_max_P, max_Q=user_max_Q,
                                        exog_col_names=exog_col_names)
            if output is None:
                st.stop()

            results, comp_df, split_info, warn_list, skip_list, stat_info = output
            st.session_state["results"] = results
            st.session_state["comp_df"] = comp_df
            st.session_state["split_info"] = split_info
            st.session_state["warn_list"] = warn_list
            st.session_state["skip_list"] = skip_list
            st.session_state["stationarity_info"] = stat_info
            st.session_state["labels"] = labels
            st.session_state["values"] = values

        if "comp_df" not in st.session_state:
            st.info("Klik 'Jalankan Analisis' untuk memulai.")
            st.stop()

        comp_df = st.session_state["comp_df"]
        results = st.session_state["results"]
        split_info = st.session_state["split_info"]
        labels = st.session_state["labels"]
        values_stored = st.session_state["values"]
        warn_list = st.session_state.get("warn_list", [])
        skip_list = st.session_state.get("skip_list", [])
        split_idx, train, test = split_info

        for w in warn_list:
            st.warning(w)
        if skip_list:
            st.info(f"ℹ️ Model dilewati: **{', '.join(skip_list)}**")

        # ── Best model (by MDAPE) ──
        best_idx = comp_df["MDAPE (%)"].idxmin()
        best_model = comp_df.loc[best_idx, "Model"]
        best_mdape = comp_df.loc[best_idx, "MDAPE (%)"]
        best_mase = comp_df.loc[best_idx, "MASE"]
        best_mape = comp_df.loc[best_idx, "MAPE (%)"]
        best_rmse = comp_df.loc[best_idx, "RMSE"]
        mase_label = "✅ Lebih baik dari Naïve" if best_mase < 1 else "⚠️ Tidak lebih baik dari Naïve"

        st.markdown(f"""
        <div class="best-model">
            <h3>🏆 Model Terbaik: {best_model}</h3>
            <p>MDAPE: <b>{best_mdape:.2f}%</b> | MASE: <b>{best_mase:.4f}</b> ({mase_label}) | MAPE: <b>{best_mape:.2f}%</b> | RMSE: <b>{best_rmse:,.2f}</b></p>
            <p style="font-size:0.85em; opacity:0.9;">Training: {split_idx} data, Testing: {len(test)} data
            ({len(results)} model berhasil dijalankan)</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        # ── Comparison table (sorted by MDAPE ascending = best first) ──
        st.markdown("#### 📋 Perbandingan Model")
        sorted_df = comp_df.sort_values("MDAPE (%)", ascending=True).reset_index(drop=True)

        def highlight_best_sorted(row):
            if row.name == 0:  # first row = best MDAPE after sorting
                return ["background-color: #d4edda; font-weight: bold"] * len(row)
            return [""] * len(row)

        styled_df = sorted_df.style.apply(highlight_best_sorted, axis=1).format({
            "MDAPE (%)": "{:.2f}", "MASE": "{:.4f}", "MAPE (%)": "{:.2f}", "RMSE": "{:,.2f}",
        })
        st.dataframe(styled_df, use_container_width=True)

        # ── Rumus metrik ──
        with st.expander("📐 Rumus Metrik Evaluasi", expanded=False):
            st.markdown(r"""
**MDAPE (Median Absolute Percentage Error)** — *digunakan untuk auto-tuning parameter*

$$\text{MDAPE} = \text{median}\left(\left|\frac{Y_t - \hat{Y}_t}{Y_t}\right|\right) \times 100\%$$

- Menggunakan **median** sehingga **robust terhadap outlier**
- Lebih stabil untuk data pendek (sedikit observasi)
- Digunakan sebagai kriteria utama pemilihan model terbaik dan auto-tuning parameter

---

**MASE (Mean Absolute Scaled Error)** — *Hyndman & Koehler, 2006*

$$\text{MASE} = \frac{\displaystyle\frac{1}{h}\sum_{t=1}^{h}|Y_t - \hat{Y}_t|}{\displaystyle\frac{1}{n-1}\sum_{i=2}^{n}|Y_i - Y_{i-1}|}$$

- Pembilang: MAE model pada data out-of-sample
- Penyebut: MAE naïve forecast pada data training (in-sample)
- **MASE < 1** → model lebih baik dari Naïve ✅
- **MASE > 1** → model lebih buruk dari Naïve ⚠️

---

**MAPE (Mean Absolute Percentage Error)**

$$\text{MAPE} = \frac{100\%}{h}\sum_{t=1}^{h}\left|\frac{Y_t - \hat{Y}_t}{Y_t}\right|$$

- Mengukur rata-rata persentase kesalahan absolut
- Mudah diinterpretasi (misal: "akurasi 95%" = MAPE 5%)
- ⚠️ Tidak stabil jika ada nilai aktual mendekati nol

---

**RMSE (Root Mean Squared Error)**

$$\text{RMSE} = \sqrt{\frac{1}{h}\sum_{t=1}^{h}(Y_t - \hat{Y}_t)^2}$$

- Sensitif terhadap error besar (outlier) karena dikuadratkan
- Satuan sama dengan data asli
- Cocok untuk mendeteksi model yang kadang meleset jauh

---

**Keterangan simbol:**
- $Y_t$ = nilai aktual pada periode $t$
- $\hat{Y}_t$ = nilai prediksi pada periode $t$
- $h$ = jumlah data out-of-sample (testing)
- $n$ = jumlah data training

---

**Uji Stasioneritas (untuk SARIMA / SARFIMA / SARIMAX):**

**ADF (Augmented Dickey-Fuller):**
- H₀: data memiliki unit root (tidak stasioner)
- Jika p-value < 0.05 → tolak H₀ → data **stasioner**

**KPSS (Kwiatkowski-Phillips-Schmidt-Shin):**
- H₀: data **stasioner** (kebalikan ADF)
- Jika p-value ≥ 0.05 → gagal tolak H₀ → data **stasioner**

| ADF | KPSS | Kesimpulan |
|-----|------|------------|
| ✅ Tolak H₀ | ✅ Gagal tolak H₀ | Stasioner → d=0 |
| ❌ Gagal tolak | ❌ Tolak H₀ | Tidak stasioner → d=1 |
| ✅ Tolak H₀ | ❌ Tolak H₀ | Tren deterministik → d=0 |
| ❌ Gagal tolak | ✅ Gagal tolak H₀ | Unit root → d=1 |

**Catatan:** Parameter d dan D pada aplikasi ini dipilih berdasarkan **MDAPE out-of-sample**
(bukan berdasarkan uji stasioneritas), karena tujuan utama adalah akurasi peramalan.
Uji ADF/KPSS ditampilkan sebagai **informasi pendukung** untuk analisis.
            """)

        # ── Notes ──
        st.markdown("#### 📝 Catatan")
        for r_data in results:
            out_pred = r_data["out_pred"]
            is_flat = len(set(np.round(out_pred, 4))) <= 1
            if is_flat:
                st.caption(f"⚠️ **{r_data['name']}**: ramalan out-of-sample **datar** (flat) — "
                           "tidak memiliki komponen tren/musiman.")

        # ── Charts per model ──
        st.markdown("#### 📈 Grafik Per Model")
        import altair as alt

        for r in results:
            with st.expander(f"📉 {r['name']}"):
                # ── Model description ──
                _base = r["name"].split("(")[0].strip() if "(" in r["name"] else r["name"]
                if _base == "ARIMAX":
                    _base = "SARIMAX"
                elif _base == "ARIMA":
                    _base = "SARIMA"
                elif _base == "ARFIMA":
                    _base = "SARFIMA"
                info = MODEL_INFO.get(r["name"]) or MODEL_INFO.get(_base)
                if info:
                    st.markdown(f"**Deskripsi:** {info['desc']}")
                    # ── Asumsi model ──
                    if "assumptions" in info:
                        st.markdown("**📋 Asumsi:**")
                        for a in info["assumptions"]:
                            st.caption(f"• {a}")
                    # ── Rumus matematika ──
                    if "formula" in info:
                        st.markdown("**📐 Rumus:**")
                        for eq in info["formula"]:
                            st.latex(eq)
                        if "symbols" in info:
                            st.markdown("**Keterangan simbol:**")
                            for sym, meaning in info["symbols"].items():
                                st.caption(f"{sym} = {meaning}")
                    col_pro, col_con = st.columns(2)
                    with col_pro:
                        st.markdown("**✅ Kelebihan:**")
                        for p in info["pros"]:
                            st.caption(f"• {p}")
                    with col_con:
                        st.markdown("**⚠️ Kekurangan:**")
                        for c in info["cons"]:
                            st.caption(f"• {c}")
                    st.divider()

                # ── Uji Stasioneritas (untuk SARIMA/SARFIMA & SARIMAX) ──
                stat_info = st.session_state.get("stationarity_info", {})
                if r["name"].startswith(("SARIMA", "SARFIMA", "ARFIMA", "ARIMA", "SARIMAX", "ARIMAX", "Avg(")):
                    if stat_info:
                        st.markdown("**🔬 Uji Stasioneritas Data Training**")

                        # Non-seasonal
                        st.markdown("**Non-Musiman:**")
                        col_adf, col_kpss = st.columns(2)
                        with col_adf:
                            adf_p = stat_info.get("adf_pvalue")
                            adf_s = stat_info.get("adf_stat")
                            adf_ok = stat_info.get("adf_stasioner")
                            icon_adf = "✅" if adf_ok else "❌"
                            if adf_p is not None and not np.isnan(adf_p):
                                st.metric("ADF Test",
                                          f"p = {adf_p:.4f} {icon_adf}",
                                          delta=f"stat = {adf_s:.4f}",
                                          delta_color="off")
                                st.caption("H₀: data tidak stasioner. "
                                           f"{'Tolak H₀ → **stasioner**' if adf_ok else 'Gagal tolak H₀ → **tidak stasioner**'}")
                            else:
                                st.caption("ADF test tidak dapat dihitung")
                        with col_kpss:
                            kpss_p = stat_info.get("kpss_pvalue")
                            kpss_s = stat_info.get("kpss_stat")
                            kpss_ok = stat_info.get("kpss_stasioner")
                            icon_kpss = "✅" if kpss_ok else "❌"
                            if kpss_p is not None and not np.isnan(kpss_p):
                                st.metric("KPSS Test",
                                          f"p = {kpss_p:.4f} {icon_kpss}",
                                          delta=f"stat = {kpss_s:.4f}",
                                          delta_color="off")
                                st.caption("H₀: data stasioner. "
                                           f"{'Gagal tolak H₀ → **stasioner**' if kpss_ok else 'Tolak H₀ → **tidak stasioner**'}")
                            else:
                                st.caption("KPSS test tidak dapat dihitung")
                        st.caption(f"📌 Kesimpulan non-musiman: {stat_info.get('kesimpulan_nonseasonal', '-')}")
                        st.caption(f"📎 Rekomendasi uji statistik: **d = {stat_info.get('rekomendasi_d', '?')}**")

                        # GPH estimation info
                        gph_d_info = stat_info.get("gph_d")
                        gph_D_info = stat_info.get("gph_D")
                        if gph_d_info and not np.isnan(gph_d_info.get("se", float("nan"))):
                            d_est = gph_d_info["d"]
                            d_se = gph_d_info["se"]
                            d_lo = gph_d_info["conf_lo"]
                            d_hi = gph_d_info["conf_hi"]
                            st.caption(f"📐 Estimasi GPH (Geweke & Porter-Hudak): "
                                       f"**d̂ = {d_est}** "
                                       f"(SE={d_se:.3f}, CI 95%: [{d_lo}, {d_hi}])")

                        # Seasonal (for SARIMA/SARFIMA)
                        if r["name"].startswith(("SARIMA", "SARFIMA", "ARFIMA", "ARIMA", "SARIMAX", "ARIMAX", "Avg(")) and stat_info.get("rekomendasi_D") is not None:
                            st.markdown("**Musiman:**")
                            sadf_p = stat_info.get("seasonal_adf_pvalue")
                            sadf_s = stat_info.get("seasonal_adf_stat")
                            sadf_ok = stat_info.get("seasonal_stasioner")
                            if sadf_p is not None and not np.isnan(sadf_p):
                                icon_s = "✅" if sadf_ok else "❌"
                                st.metric("ADF pada Seasonal Difference",
                                          f"p = {sadf_p:.4f} {icon_s}",
                                          delta=f"stat = {sadf_s:.4f}",
                                          delta_color="off")
                                st.caption("Menguji stasioneritas setelah differencing musiman "
                                           f"(Yₜ − Yₜ₋ₛ). "
                                           f"{'Data stasioner setelah diff musiman' if sadf_ok else 'Data masih tidak stasioner setelah diff musiman'}")
                            else:
                                st.caption("Uji musiman tidak dapat dihitung (data terlalu pendek)")
                            st.caption(f"📌 Kesimpulan musiman: {stat_info.get('kesimpulan_seasonal', '-')}")
                            st.caption(f"📎 Rekomendasi uji statistik: **D = {stat_info.get('rekomendasi_D', '?')}**")
                            if gph_D_info and not np.isnan(gph_D_info.get("se", float("nan"))):
                                D_est = gph_D_info["D"]
                                D_se = gph_D_info["se"]
                                D_lo = gph_D_info["conf_lo"]
                                D_hi = gph_D_info["conf_hi"]
                                st.caption(f"📐 Estimasi GPH Seasonal: "
                                           f"**D̂ = {D_est}** "
                                           f"(SE={D_se:.3f}, CI 95%: [{D_lo}, {D_hi}])")

                        # Compare with actual MDAPE selection
                        actual_name = r["name"]
                        # Extract actual d and D from model name
                        m_order = re.search(r'\((\d+),\s*([0-9.]+),\s*(\d+)\)', actual_name)
                        if m_order:
                            actual_d_str = m_order.group(2)
                            actual_d = float(actual_d_str) if '.' in actual_d_str else int(actual_d_str)
                            rek_d = stat_info.get("rekomendasi_d", "?")
                            if actual_d == rek_d:
                                match_d = "✅ sama dengan rekomendasi uji"
                            elif isinstance(actual_d, float) and actual_d != int(actual_d):
                                match_d = f"(GPH, uji statistik: d={rek_d})"
                            else:
                                match_d = f"≠ uji menyarankan d={rek_d}"
                            note_d = f"d={actual_d_str} {match_d}"
                        else:
                            note_d = ""
                        m_seasonal = re.search(r'x\((\d+),\s*([0-9.]+),\s*(\d+),', actual_name)
                        if m_seasonal and r["name"].startswith(("SARIMA", "SARFIMA", "SARIMAX", "ARIMAX", "Avg(")):
                            actual_D_str = m_seasonal.group(2)
                            actual_D = float(actual_D_str) if '.' in actual_D_str else int(actual_D_str)
                            rek_D = stat_info.get("rekomendasi_D", "?")
                            if actual_D == rek_D:
                                match_D = "✅ sama dengan rekomendasi uji"
                            elif isinstance(actual_D, float) and actual_D != int(actual_D):
                                match_D = f"(GPH, uji statistik: D={rek_D})"
                            else:
                                match_D = f"≠ uji menyarankan D={rek_D}"
                            note_D = f", D={actual_D_str} {match_D}"
                        else:
                            note_D = ""
                        if note_d:
                            st.info(f"🔍 **Perbandingan:** Parameter terpilih (MDAPE): {note_d}{note_D}. "
                                    "Parameter dipilih berdasarkan MDAPE out-of-sample, "
                                    "bukan berdasarkan uji stasioneritas.")
                        st.divider()

                # ── Tabel Estimasi Parameter (SARIMAX) ──
                if r.get("param_table") is not None:
                    st.markdown("**📊 Estimasi Parameter Model**")
                    ptbl = r["param_table"].copy()
                    ptbl["Koefisien"] = ptbl["Koefisien"].map(lambda x: f"{x:.6f}")
                    ptbl["Std. Error"] = ptbl["Std. Error"].map(lambda x: f"{x:.6f}")
                    ptbl["z"] = ptbl["z"].map(lambda x: f"{x:.4f}")
                    ptbl["P>|z|"] = ptbl["P>|z|"].map(lambda x: f"{x:.4f}")
                    ptbl.index = range(1, len(ptbl) + 1)
                    ptbl.index.name = "No"
                    st.dataframe(ptbl, use_container_width=True)
                    # Interpretasi signifikansi
                    pvals = r["param_table"]["P>|z|"].values
                    params = r["param_table"]["Parameter"].values
                    sig_005 = [p for p, pv in zip(params, pvals) if pv < 0.05]
                    not_sig = [p for p, pv in zip(params, pvals) if pv >= 0.05]
                    if sig_005:
                        st.caption(f"✅ **Signifikan (α=5%):** {', '.join(sig_005)}")
                    if not_sig:
                        st.caption(f"⚠️ **Tidak signifikan (α=5%):** {', '.join(not_sig)}")
                    st.divider()

                ia = np.array(r["in_actual"])
                ip = np.array(r["in_pred"])
                oa = np.array(r["out_actual"])
                op = np.array(r["out_pred"])

                n_total = len(values_stored)
                in_start = n_total - len(test) - len(ia)

                chart_data = {"Periode": [], "Aktual": [], "Prediksi": []}
                for i in range(len(ia)):
                    idx = in_start + i
                    lbl = labels[idx] if idx < len(labels) else f"t{idx+1}"
                    chart_data["Periode"].append(lbl)
                    chart_data["Aktual"].append(float(ia[i]))
                    chart_data["Prediksi"].append(float(ip[i]) if i < len(ip) else None)
                for i in range(len(oa)):
                    idx = split_idx + i
                    lbl = labels[idx] if idx < len(labels) else f"t{idx+1}"
                    chart_data["Periode"].append(lbl)
                    chart_data["Aktual"].append(float(oa[i]))
                    chart_data["Prediksi"].append(float(op[i]) if i < len(op) else None)

                cdf = pd.DataFrame(chart_data)
                melted = cdf.melt(id_vars=["Periode"], value_vars=["Aktual", "Prediksi"],
                                  var_name="Series", value_name="Nilai")
                melted = melted.dropna(subset=["Nilai"])

                # Hitung domain Y dengan margin 5% agar lebih rapat
                _yvals2 = melted["Nilai"].dropna()
                _ymin2, _ymax2 = float(_yvals2.min()), float(_yvals2.max())
                _ymargin2 = (_ymax2 - _ymin2) * 0.05 if _ymax2 > _ymin2 else abs(_ymax2) * 0.05
                _ydomain2 = [_ymin2 - _ymargin2, _ymax2 + _ymargin2]

                line = alt.Chart(melted).mark_line(point=True).encode(
                    x=alt.X("Periode:N", sort=None, title="Periode"),
                    y=alt.Y("Nilai:Q", title=ylabel_full,
                            scale=alt.Scale(zero=False, domain=_ydomain2)),
                    color=alt.Color("Series:N", scale=alt.Scale(
                        domain=["Aktual", "Prediksi"], range=["#1f77b4", "#ff7f0e"])),
                    strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(
                        domain=["Aktual", "Prediksi"], range=[[0], [5, 5]])),
                ).properties(height=350)
                st.altair_chart(line, use_container_width=True)

                # Download out-of-sample chart
                safe_mn2 = re.sub(r'[^\w\-]', '_', r["name"])
                png_oos = make_line_chart_png(
                    periodes=chart_data["Periode"],
                    series_dict={"Aktual": chart_data["Aktual"],
                                 "Prediksi": chart_data["Prediksi"]},
                    title=f"Out-of-Sample — {r['name']}",
                    ylabel=ylabel_full,
                    dashed_series={"Prediksi"},
                )
                st.download_button(
                    "📥 Download Grafik (PNG)", data=png_oos,
                    file_name=f"grafik_oos_{safe_mn2}.png",
                    mime="image/png", key=f"dl_oos_{safe_mn2}",
                )

                mc1, mc2, mc3, mc4 = st.columns(4)
                ci = comp_df[comp_df["Model"] == r["name"]].iloc[0]
                mase_val = ci['MASE']
                mase_status = "✅" if mase_val < 1 else "⚠️"
                mc1.metric("MDAPE", f"{ci['MDAPE (%)']:.2f}%")
                mc2.metric("MASE", f"{mase_val:.4f} {mase_status}")
                mc3.metric("MAPE", f"{ci['MAPE (%)']:.2f}%")
                mc4.metric("RMSE", f"{ci['RMSE']:,.2f}")

    # ══════════════════════════════════════════════
    # TAB 3: Forecast with Confidence Intervals
    # ══════════════════════════════════════════════
    with tab_forecast:
        if "results" not in st.session_state:
            st.info("Jalankan analisis terlebih dahulu di tab 'Analisis & Perbandingan'.")
            st.stop()

        results = st.session_state["results"]
        comp_df = st.session_state["comp_df"]
        labels = st.session_state["labels"]
        values_stored = st.session_state["values"]

        best_idx = comp_df["MDAPE (%)"].idxmin()
        best_model_name = comp_df.loc[best_idx, "Model"]

        # ── Settings ──
        h = st.session_state.get("forecast_h", 4)
        st.caption(f"📏 Panjang ramalan: **{h} periode** (diatur di tab Analisis)")
        show_ci = st.checkbox("Tampilkan Confidence Interval", value=True)

        # ── Input exog future for SARIMAX ──
        exog_future = None
        has_exog_model = any(r.get("is_sarimax") or r.get("is_mlpx") for r in results)
        if has_exog_model and st.session_state.get("use_sarimax"):
            exog_col_names = st.session_state.get("exog_col_names", [])
            n_exog_vars = len(exog_col_names) if exog_col_names else 1
            st.markdown("#### 🔗 Data Variabel Eksogen untuk Periode Ramalan")
            st.caption(f"Masukkan nilai variabel eksogen untuk **{h} periode** ke depan. "
                       "SARIMAX/MLPX membutuhkan nilai ini untuk menghasilkan ramalan.")

            exog_future_cols = []
            exog_future_valid = True
            for i in range(n_exog_vars):
                col_nm = exog_col_names[i] if i < len(exog_col_names) else f"X{i+1}"
                exog_fc_input = st.text_input(
                    f"Nilai {col_nm} untuk {h} periode ke depan (pisahkan dengan koma)",
                    placeholder=f"contoh: 110, 115, 120, 125",
                    key=f"exog_fc_{i}",
                )
                if exog_fc_input.strip():
                    try:
                        raw = exog_fc_input.strip()
                        if ";" in raw or "\n" in raw:
                            raw = raw.replace("\n", ";").replace(",", ".")
                            tokens = [t.strip() for t in raw.split(";") if t.strip()]
                        else:
                            tokens = [t.strip() for t in raw.split(",") if t.strip()]
                        vals_fc = [parse_number(t) for t in tokens]
                        if len(vals_fc) != h:
                            st.error(f"❌ {col_nm}: jumlah data ({len(vals_fc)}) ≠ {h} periode.")
                            exog_future_valid = False
                        else:
                            exog_future_cols.append(vals_fc)
                    except ValueError:
                        st.error(f"❌ {col_nm}: data tidak valid.")
                        exog_future_valid = False
                else:
                    exog_future_valid = False

            if exog_future_valid and len(exog_future_cols) == n_exog_vars:
                exog_future = np.column_stack(exog_future_cols)
                st.session_state["_exog_future"] = exog_future
                st.session_state["_exog_future_h"] = h
                # Simpan ke database agar persisten
                ds_id_current = st.session_state.get("selected_ds_id")
                if ds_id_current:
                    save_exog_future(ds_id_current, exog_future,
                                     exog_col_names, h)
                st.success(f"✅ Data eksogen ramalan valid: {n_exog_vars} variabel × {h} periode — tersimpan di database")
            else:
                # Coba ambil dari session_state
                saved_ef = st.session_state.get("_exog_future")
                saved_ef_h = st.session_state.get("_exog_future_h")
                if saved_ef is not None and saved_ef_h == h:
                    exog_future = saved_ef
                    st.success(f"✅ Menggunakan data eksogen ramalan yang sudah diisi sebelumnya.")
                else:
                    # Fallback: coba muat dari database
                    ds_id_current = st.session_state.get("selected_ds_id")
                    if ds_id_current:
                        db_ef, db_ef_cols, db_ef_h = load_exog_future(ds_id_current)
                        if db_ef is not None and db_ef_h == h:
                            exog_future = db_ef
                            st.session_state["_exog_future"] = db_ef
                            st.session_state["_exog_future_h"] = db_ef_h
                            st.success(f"✅ Data eksogen ramalan dimuat dari database "
                                       f"({len(db_ef_cols)} variabel × {db_ef_h} periode).")
                        elif db_ef is not None and db_ef_h != h:
                            st.warning(f"⚠️ Data eksogen tersimpan untuk {db_ef_h} periode, "
                                       f"tetapi panjang ramalan saat ini {h} periode. "
                                       f"Silakan isi ulang data eksogen.")
                        else:
                            st.info("💡 Masukkan data eksogen untuk periode ramalan agar SARIMAX/MLPX dapat meramal. "
                                    "Model lain tetap bisa meramal tanpa data eksogen.")

        # ── Run forecast for ALL models ──
        import altair as alt

        all_forecasts = []
        for r in results:
            try:
                if (r.get("is_sarimax") or r.get("is_mlpx")) and exog_future is not None:
                    fc_result = r["forecast_func"](h, exog_future=exog_future)
                elif (r.get("is_sarimax") or r.get("is_mlpx")) and exog_future is None:
                    # Skip SARIMAX/MLPX forecast if no exog_future
                    all_forecasts.append({
                        "name": r["name"],
                        "fc_values": [np.nan] * h,
                        "lo95": None, "hi95": None,
                        "lo80": None, "hi80": None,
                        "success": False,
                        "error": "Data eksogen periode ramalan belum diisi",
                    })
                    continue
                else:
                    fc_result = r["forecast_func"](h)
                fc_values, lo95, hi95, lo80, hi80 = fc_result
                all_forecasts.append({
                    "name": r["name"],
                    "fc_values": fc_values,
                    "lo95": lo95, "hi95": hi95,
                    "lo80": lo80, "hi80": hi80,
                    "success": True,
                })
            except Exception as e:
                all_forecasts.append({
                    "name": r["name"],
                    "fc_values": [np.nan] * h,
                    "lo95": None, "hi95": None,
                    "lo80": None, "hi80": None,
                    "success": False,
                    "error": str(e),
                })

        # ── Combined forecast comparison table ──
        st.markdown(f"#### 📊 Perbandingan Ramalan {h} Periode ke Depan — Semua Model")

        combined_data = {"Periode": [f"+{i+1}" for i in range(h)]}
        for fc in all_forecasts:
            if fc["success"]:
                combined_data[fc["name"]] = [round(v, 2) for v in fc["fc_values"]]
        combined_table = pd.DataFrame(combined_data)
        combined_table.index = range(1, h + 1)

        def highlight_best_model_col(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            if best_model_name in df.columns:
                styles[best_model_name] = "background-color: #d4edda; font-weight: bold"
            return styles

        st.dataframe(
            combined_table.style.apply(highlight_best_model_col, axis=None),
            use_container_width=True,
        )
        st.caption(f"🏆 Kolom hijau = model terbaik ({best_model_name}) berdasarkan MDAPE")

        # ── Combined chart: all models forecast ──
        st.markdown("#### 📈 Grafik Ramalan Semua Model")

        hist_n = min(20, len(values_stored))
        hist_labels = labels[-hist_n:]
        hist_values = values_stored[-hist_n:]
        fc_labels = [f"+{i+1}" for i in range(h)]
        all_labels = hist_labels + fc_labels

        chart_rows = []
        # Historical data
        for i, lbl in enumerate(hist_labels):
            chart_rows.append({
                "Periode": lbl,
                "order": i,
                "Nilai": float(hist_values[i]),
                "Model": "Aktual",
            })

        # Forecast data for each model
        for fc in all_forecasts:
            if not fc["success"]:
                continue
            # Connection point from last actual
            chart_rows.append({
                "Periode": hist_labels[-1],
                "order": hist_n - 1,
                "Nilai": float(hist_values[-1]),
                "Model": fc["name"],
            })
            for j, lbl in enumerate(fc_labels):
                chart_rows.append({
                    "Periode": lbl,
                    "order": hist_n + j,
                    "Nilai": float(fc["fc_values"][j]),
                    "Model": fc["name"],
                })

        cdf_all = pd.DataFrame(chart_rows)

        # Hitung domain Y dengan margin 5%
        _yvals_all = cdf_all["Nilai"].dropna()
        _ymin_all, _ymax_all = float(_yvals_all.min()), float(_yvals_all.max())
        _ymrg_all = (_ymax_all - _ymin_all) * 0.05 if _ymax_all > _ymin_all else abs(_ymax_all) * 0.05
        _ydomain_all = [_ymin_all - _ymrg_all, _ymax_all + _ymrg_all]

        # Nearest-point selection for interactive tooltip
        nearest = alt.selection_point(
            nearest=True, on="pointerover",
            fields=["Periode"], empty=False,
        )

        base_all = alt.Chart(cdf_all).encode(
            x=alt.X("Periode:N", sort=None, title="Periode"),
        )

        # Lines
        line_all = base_all.mark_line().encode(
            y=alt.Y("Nilai:Q", title=ylabel_full,
                    scale=alt.Scale(zero=False, domain=_ydomain_all)),
            color=alt.Color("Model:N", title="Model"),
            strokeDash=alt.condition(
                alt.datum.Model == "Aktual",
                alt.value([0]),
                alt.value([5, 5]),
            ),
            strokeWidth=alt.condition(
                alt.datum.Model == "Aktual",
                alt.value(2.5),
                alt.value(1.5),
            ),
        )

        # Visible points on hover (show dots when cursor is near)
        points_all = base_all.mark_circle(size=60).encode(
            y=alt.Y("Nilai:Q", scale=alt.Scale(zero=False, domain=_ydomain_all)),
            color=alt.Color("Model:N", title="Model"),
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("Periode:N", title="Periode"),
                alt.Tooltip("Nilai:Q", title=ylabel_full, format=",.2f"),
            ],
        ).add_params(nearest)

        # Vertical rule at hover position
        rule_all = base_all.mark_rule(color="gray", strokeDash=[3, 3]).encode(
            opacity=alt.condition(nearest, alt.value(0.5), alt.value(0)),
        ).transform_filter(nearest)

        chart_all = alt.layer(line_all, points_all, rule_all).properties(height=450)
        st.altair_chart(chart_all, use_container_width=True)
        st.caption("Garis solid tebal = data aktual. Garis putus-putus = ramalan masing-masing model.")

        # Download grafik semua model
        dl_series_all = {"Aktual": list(hist_values) + [None] * h}
        dl_dashed_all = set()
        for fc in all_forecasts:
            if not fc["success"]:
                continue
            dl_dashed_all.add(fc["name"])
            fc_full = [None] * (hist_n - 1) + [float(hist_values[-1])]
            fc_full += [float(v) for v in fc["fc_values"]]
            dl_series_all[fc["name"]] = fc_full
        png_all_models = make_line_chart_png(
            periodes=all_labels,
            series_dict=dl_series_all,
            title="Grafik Ramalan Semua Model",
            ylabel=ylabel_full,
            dashed_series=dl_dashed_all,
        )
        st.download_button(
            "📥 Download Grafik Semua Model (PNG)", data=png_all_models,
            file_name="grafik_ramalan_semua_model.png",
            mime="image/png", key="dl_chart_all_models",
        )

        # ── Per-model detail in expanders ──
        st.markdown("#### 🔍 Detail Ramalan per Model")

        # Sort by MDAPE (best first)
        sorted_comp = comp_df.sort_values("MDAPE (%)", ascending=True)

        for _, comp_row in sorted_comp.iterrows():
            model_name = comp_row["Model"]
            fc = [f for f in all_forecasts if f["name"] == model_name][0]
            is_best = model_name == best_model_name

            with st.expander(f"{'🏆 ' if is_best else ''}{model_name} — MDAPE: {comp_row['MDAPE (%)']:.2f}% | MASE: {comp_row['MASE']:.4f}"):
                if not fc["success"]:
                    st.error(f"Gagal meramalkan: {fc['error']}")
                    continue

                fc_values = fc["fc_values"]
                lo95, hi95 = fc["lo95"], fc["hi95"]
                lo80, hi80 = fc["lo80"], fc["hi80"]

                # Forecast table
                tbl_data = {
                    "Periode": [f"+{i+1}" for i in range(h)],
                    f"Ramalan{unit_label}": [round(v, 2) for v in fc_values],
                }
                if show_ci and lo80 is not None:
                    tbl_data["Batas Bawah 80%"] = [round(v, 2) for v in lo80]
                    tbl_data["Batas Atas 80%"] = [round(v, 2) for v in hi80]
                if show_ci and lo95 is not None:
                    tbl_data["Batas Bawah 95%"] = [round(v, 2) for v in lo95]
                    tbl_data["Batas Atas 95%"] = [round(v, 2) for v in hi95]
                fc_tbl = pd.DataFrame(tbl_data)
                fc_tbl.index = range(1, h + 1)
                st.dataframe(fc_tbl, use_container_width=True)

                # Individual chart with CI
                chart_rows_i = []
                for i, lbl in enumerate(all_labels):
                    row = {"Periode": lbl, "order": i}
                    if i < hist_n:
                        row["Aktual"] = float(hist_values[i])
                    if i >= hist_n:
                        fi = i - hist_n
                        row["Ramalan"] = float(fc_values[fi])
                        if show_ci:
                            if lo95 is not None and fi < len(lo95):
                                row["CI95_lo"] = float(lo95[fi])
                                row["CI95_hi"] = float(hi95[fi])
                            if lo80 is not None and fi < len(lo80):
                                row["CI80_lo"] = float(lo80[fi])
                                row["CI80_hi"] = float(hi80[fi])
                    if i == hist_n - 1:
                        row["Ramalan"] = float(hist_values[-1])
                    chart_rows_i.append(row)

                cdf_i = pd.DataFrame(chart_rows_i)

                # Hitung domain Y per model dengan margin 5%
                _ycols_i = [c for c in ["Aktual", "Ramalan", "CI95_lo", "CI95_hi",
                                         "CI80_lo", "CI80_hi"] if c in cdf_i.columns]
                _yvals_i = pd.concat([cdf_i[c] for c in _ycols_i]).dropna()
                _ymin_i, _ymax_i = float(_yvals_i.min()), float(_yvals_i.max())
                _ymrg_i = (_ymax_i - _ymin_i) * 0.05 if _ymax_i > _ymin_i else abs(_ymax_i) * 0.05
                _ydomain_i = [_ymin_i - _ymrg_i, _ymax_i + _ymrg_i]

                base_i = alt.Chart(cdf_i).encode(x=alt.X("Periode:N", sort=None, title="Periode"))

                layers_i = []
                if show_ci and "CI95_lo" in cdf_i.columns:
                    band95 = base_i.mark_area(opacity=0.15, color="#e74c3c").encode(
                        y=alt.Y("CI95_lo:Q", title=ylabel_full,
                                scale=alt.Scale(zero=False, domain=_ydomain_i)),
                        y2="CI95_hi:Q",
                    )
                    layers_i.append(band95)
                if show_ci and "CI80_lo" in cdf_i.columns:
                    band80 = base_i.mark_area(opacity=0.25, color="#e74c3c").encode(
                        y=alt.Y("CI80_lo:Q", scale=alt.Scale(zero=False, domain=_ydomain_i)),
                        y2="CI80_hi:Q",
                    )
                    layers_i.append(band80)

                melted_i = cdf_i.melt(
                    id_vars=["Periode", "order"],
                    value_vars=[c for c in ["Aktual", "Ramalan"] if c in cdf_i.columns],
                    var_name="Series", value_name="Nilai",
                )
                melted_i = melted_i.dropna(subset=["Nilai"])
                line_i = alt.Chart(melted_i).mark_line(point=True).encode(
                    x=alt.X("Periode:N", sort=None),
                    y=alt.Y("Nilai:Q", title=ylabel_full,
                            scale=alt.Scale(zero=False, domain=_ydomain_i)),
                    color=alt.Color("Series:N", scale=alt.Scale(
                        domain=["Aktual", "Ramalan"], range=["#1f77b4", "#e74c3c"])),
                    strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(
                        domain=["Aktual", "Ramalan"], range=[[0], [5, 5]])),
                )
                layers_i.append(line_i)

                chart_i = alt.layer(*layers_i).properties(height=350)
                st.altair_chart(chart_i, use_container_width=True)

                # Download per-model chart
                dl_s_i = {}
                if "Aktual" in cdf_i.columns:
                    dl_s_i["Aktual"] = cdf_i["Aktual"].tolist()
                if "Ramalan" in cdf_i.columns:
                    dl_s_i["Ramalan"] = cdf_i["Ramalan"].tolist()
                safe_mn = re.sub(r'[^\w\-]', '_', model_name)
                png_i = make_line_chart_png(
                    periodes=cdf_i["Periode"].tolist(),
                    series_dict=dl_s_i,
                    title=f"Ramalan — {model_name}",
                    ylabel=ylabel_full,
                    dashed_series={"Ramalan"},
                )
                st.download_button(
                    "📥 Download Grafik (PNG)", data=png_i,
                    file_name=f"grafik_{safe_mn}.png",
                    mime="image/png", key=f"dl_chart_{safe_mn}",
                )

                # Model info
                # Try base model name for MODEL_INFO lookup
                base_name = model_name.split("(")[0].strip() if "(" in model_name else model_name
                if base_name == "ARIMAX":
                    base_name = "SARIMAX"
                elif base_name == "ARIMA":
                    base_name = "SARIMA"
                elif base_name == "ARFIMA":
                    base_name = "SARFIMA"
                info = MODEL_INFO.get(model_name) or MODEL_INFO.get(base_name)
                if info:
                    st.markdown(f"**ℹ️ Deskripsi:** {info['desc']}")

                # ── Kesimpulan variabel eksogen SARIMAX ──
                if base_name in ("SARIMAX", "ARIMAX") and "[" in model_name:
                    import re as _re
                    _m_exog = _re.search(r"\[([^\]]+)\]", model_name)
                    if _m_exog:
                        _exog_str = _m_exog.group(1)
                        _exog_list = [x.strip() for x in _exog_str.split(",")]
                        st.markdown("---")
                        st.markdown("**📋 Variabel Eksogen Terpilih:**")
                        st.markdown(
                            f"- **Variabel:** {', '.join(_exog_list)} "
                            f"({'semua variabel' if len(_exog_list) > 1 else 'variabel tunggal terbaik'})\n"
                            f"- **Metode seleksi:** perbandingan otomatis seluruh variabel vs. masing-masing individu "
                            f"(dipilih berdasarkan MDAPE terkecil)"
                        )

                # ── Kesimpulan hyperparameter MLP / MLPX ──
                if base_name in ("MLP", "MLPX") and "(" in model_name:
                    import re as _re
                    _m_lb = _re.search(r"lb=(\d+)", model_name)
                    _m_h = _re.search(r"h=([^,\)]+)", model_name)
                    _m_act = _re.search(r"(relu|tanh)", model_name)
                    _lb_val = _m_lb.group(1) if _m_lb else "?"
                    _h_val = _m_h.group(1) if _m_h else "?"
                    _act_val = _m_act.group(1) if _m_act else "?"
                    st.markdown("---")
                    st.markdown("**🧠 Kesimpulan Hyperparameter Terpilih:**")
                    st.markdown(
                        f"- **Lookback (lb):** {_lb_val} — model melihat {_lb_val} periode ke belakang\n"
                        f"- **Hidden neurons (h):** {_h_val} — jumlah neuron pada hidden layer\n"
                        f"- **Fungsi aktivasi:** `{_act_val}` — "
                        f"{'menangkap non-linearitas dengan output [-1, 1]' if _act_val == 'tanh' else 'sederhana dan efisien, output [0, ∞)'}"
                    )
                    if base_name == "MLPX":
                        _m_exog = _re.search(r"\[([^\]]+)\]", model_name)
                        if _m_exog:
                            _exog_str = _m_exog.group(1)
                            _exog_list = [x.strip() for x in _exog_str.split(",")]
                            st.markdown(
                                f"- **Variabel eksogen terpilih:** {', '.join(_exog_list)} "
                                f"({'semua variabel' if len(_exog_list) > 1 else 'variabel tunggal terbaik'})\n"
                                f"- **Metode seleksi:** perbandingan otomatis antara seluruh variabel vs. masing-masing individu"
                            )

        if show_ci:
            st.caption("Area merah muda: confidence interval 95% (luar) dan 80% (dalam). "
                       "Semakin jauh ramalan, semakin lebar interval — mencerminkan ketidakpastian yang meningkat.")

        # ── Download Excel ──
        st.markdown("#### 📥 Download Hasil")

        def create_excel_download():
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                # Sheet 1: Data awal + eksogen
                data_sheet = pd.DataFrame({"Periode": labels, "Nilai": values_stored})
                if st.session_state.get("use_sarimax") and st.session_state.get("exog_data") is not None:
                    exog_d = st.session_state["exog_data"]
                    exog_n = st.session_state.get("exog_col_names", [])
                    if len(exog_d) == len(labels):
                        for ci, cn in enumerate(exog_n):
                            if exog_d.ndim == 2 and ci < exog_d.shape[1]:
                                data_sheet[cn] = exog_d[:, ci]
                            elif exog_d.ndim == 1 and ci == 0:
                                data_sheet[cn] = exog_d
                data_sheet.to_excel(writer, sheet_name="Data", index=False)

                # Sheet 2: Data awal + ramalan semua model (sambung ke bawah)
                data_forecast_df = pd.DataFrame({"Periode": labels, "Nilai": values_stored})
                # Tambahkan kolom ramalan tiap model (NaN untuk baris data historis)
                for fc in all_forecasts:
                    if not fc["success"]:
                        continue
                    safe_n = fc["name"].replace('[', '(').replace(']', ')').replace(':', '-')
                    col_name = safe_n[:50]
                    hist_vals = [np.nan] * len(labels)
                    data_forecast_df[col_name] = hist_vals
                # Tambahkan baris ramalan
                for i_fc in range(h):
                    row = {"Periode": f"+{i_fc+1}", "Nilai": np.nan}
                    for fc in all_forecasts:
                        if not fc["success"]:
                            continue
                        safe_n = fc["name"].replace('[', '(').replace(']', ')').replace(':', '-')
                        col_name = safe_n[:50]
                        row[col_name] = round(fc["fc_values"][i_fc], 2) if i_fc < len(fc["fc_values"]) else np.nan
                    data_forecast_df = pd.concat([data_forecast_df, pd.DataFrame([row])], ignore_index=True)
                data_forecast_df.to_excel(writer, sheet_name="Data + Ramalan", index=False)

                comp_df.to_excel(writer, sheet_name="Perbandingan Model", index=False)
                # Combined forecast sheet
                combined_table.to_excel(writer, sheet_name="Ramalan Semua Model", index=False)
                # Individual forecast sheets with CI
                for fc in all_forecasts:
                    if not fc["success"]:
                        continue
                    safe_fc_name = fc['name'].replace('[', '(').replace(']', ')').replace(':', '-').replace('*', '').replace('?', '').replace('/', '-').replace('\\', '-')
                    sname_fc = f"FC_{safe_fc_name}"[:31]
                    fc_sheet_data = {
                        "Periode": [f"+{i+1}" for i in range(h)],
                        "Ramalan": [round(v, 2) for v in fc["fc_values"]],
                    }
                    if fc["lo80"] is not None:
                        fc_sheet_data["Batas Bawah 80%"] = [round(v, 2) for v in fc["lo80"]]
                        fc_sheet_data["Batas Atas 80%"] = [round(v, 2) for v in fc["hi80"]]
                    if fc["lo95"] is not None:
                        fc_sheet_data["Batas Bawah 95%"] = [round(v, 2) for v in fc["lo95"]]
                        fc_sheet_data["Batas Atas 95%"] = [round(v, 2) for v in fc["hi95"]]
                    pd.DataFrame(fc_sheet_data).to_excel(writer, sheet_name=sname_fc, index=False)
                # Out-of-sample detail per model
                for r in results:
                    # Hapus karakter yang tidak valid di nama sheet Excel
                    safe_name = r['name'].replace('[', '(').replace(']', ')').replace(':', '-').replace('*', '').replace('?', '').replace('/', '-').replace('\\', '-')
                    sname = f"OOS_{safe_name}"[:31]
                    oa, op = np.array(r["out_actual"]), np.array(r["out_pred"])
                    mn = min(len(oa), len(op))
                    pd.DataFrame({
                        "Aktual": oa[:mn], "Prediksi": op[:mn],
                        "Error": np.abs(oa[:mn] - op[:mn]),
                    }).to_excel(writer, sheet_name=sname, index=False)
            return output.getvalue()

        st.download_button(
            "📥 Download Hasil (Excel)",
            data=create_excel_download(),
            file_name=f"Hasil_Peramalan_{ds_name.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # ══════════════════════════════════════════════
    # TAB 4: Model Guide
    # ══════════════════════════════════════════════
    with tab_guide:
        st.markdown("#### 📖 Panduan Model Peramalan")
        st.caption("Klik pada setiap model untuk melihat deskripsi, kelebihan, dan kekurangannya.")

        sp = FREQ_SEASONAL.get(ds_freq)

        for i, (model_key, info) in enumerate(MODEL_INFO.items(), 1):
            # Check availability for current frequency
            available = True
            note = ""
            if model_key in ("SARIMA", "SARFIMA") and sp:
                note = f" — musiman {sp} periode"
            elif model_key in ("SARIMA", "SARFIMA") and not sp:
                note = " — ARIMA/ARFIMA (tanpa musiman)"
            elif model_key == "SARIMAX" and sp:
                note = f" — musiman {sp} periode + variabel eksogen (opsional)"
            elif model_key == "SARIMAX" and not sp:
                note = " — ARIMAX (tanpa musiman) + variabel eksogen (opsional)"

            icon = "✅" if available else "⛔"
            with st.expander(f"{icon} {i}. {model_key}{note}"):
                st.markdown(f"**Deskripsi:**\n\n{info['desc']}")
                st.markdown("")
                # ── Asumsi model ──
                if "assumptions" in info:
                    st.markdown("**📋 Asumsi Model:**")
                    for a in info["assumptions"]:
                        st.markdown(f"- {a}")
                    st.markdown("")
                # ── Rumus matematika ──
                if "formula" in info:
                    st.markdown("**📐 Rumus Matematika:**")
                    for eq in info["formula"]:
                        st.latex(eq)
                    if "symbols" in info:
                        st.markdown("**Keterangan simbol:**")
                        for sym, meaning in info["symbols"].items():
                            st.markdown(f"- {sym} = {meaning}")
                    st.markdown("")
                col_pro, col_con = st.columns(2)
                with col_pro:
                    st.markdown("**✅ Kelebihan:**")
                    for p in info["pros"]:
                        st.markdown(f"- {p}")
                with col_con:
                    st.markdown("**⚠️ Kekurangan:**")
                    for c in info["cons"]:
                        st.markdown(f"- {c}")

        st.divider()
        st.markdown("#### 📊 Ringkasan Ketersediaan Model")
        st.caption(f"Frekuensi dataset saat ini: **{ds_freq}** | "
                   f"Periode musiman: **{sp if sp else 'Tidak ada'}**")

        avail_data = []
        for model_key in MODEL_INFO:
            row = {"Model": model_key}
            for freq_name in ["Harian", "Mingguan", "Bulanan", "Triwulanan", "Semesteran", "Tahunan"]:
                freq_sp = FREQ_SEASONAL.get(freq_name)
                if model_key in ("SARIMA", "SARFIMA") and freq_sp is None:
                    row[freq_name] = "✓ (ARIMA/ARFIMA)"
                elif model_key == "SARIMAX" and freq_sp is None:
                    row[freq_name] = "✓ (ARIMAX)"
                elif model_key == "SARIMAX":
                    row[freq_name] = "✓*"
                else:
                    row[freq_name] = "✓"
            avail_data.append(row)
        avail_df = pd.DataFrame(avail_data)
        avail_df.index = range(1, len(avail_df) + 1)
        avail_df.index.name = "No"
        st.dataframe(avail_df, use_container_width=True)

        st.markdown("""
        **Catatan:**
        - ✓ = Model tersedia untuk frekuensi tersebut
        - ✓* = Model tersedia jika user mengaktifkan opsi SARIMAX dan menyediakan data eksogen
        - ✓ (ARIMA/ARFIMA) = Berjalan tanpa komponen musiman (P=D=Q=0)
        - ✓ (ARIMAX) = Berjalan tanpa komponen musiman + variabel eksogen
        - Ketersediaan juga bergantung pada jumlah data — model yang membutuhkan data lebih banyak akan dilewati otomatis
        - Model terbaik dipilih berdasarkan **MDAPE (Median Absolute Percentage Error)** terendah
        - MASE, MAPE, dan RMSE ditampilkan sebagai metrik pelengkap
        """)


if __name__ == "__main__":
    main()
