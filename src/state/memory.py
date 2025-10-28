"""
Shared Memory and Logging Persistence
Tracks decisions/outcomes, rolling metrics, and writes reports.
"""
import json
import os
from typing import Dict, Any, List
from datetime import datetime
import threading

DATA_DIR = os.path.join(os.getcwd(), 'data')
LOG_FILE = os.path.join(DATA_DIR, 'training_log.json')
REPORTS_DIR = os.path.join(os.getcwd(), 'reports')
SUMMARY_FILE = os.path.join(REPORTS_DIR, 'summary.json')

_lock = threading.Lock()


def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def append_decision(entry: Dict[str, Any]) -> None:
    """Append a decision/outcome to the training log."""
    _ensure_dirs()
    entry = dict(entry)
    entry['timestamp'] = entry.get('timestamp') or datetime.now().isoformat()
    with _lock:
        data: List[Dict[str, Any]] = []
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append(entry)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


def compute_metrics() -> Dict[str, Any]:
    """Compute rolling Sharpe, hit rate, and drawdown from log."""
    _ensure_dirs()
    with _lock:
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            data = []
    # Simple placeholders; can be replaced with proper finance math
    wins = sum(1 for d in data if d.get('outcome', 0.0) > 0)
    losses = sum(1 for d in data if d.get('outcome', 0.0) < 0)
    hit_rate = (wins / max(1, (wins + losses))) if (wins + losses) > 0 else 0.0
    total_pnl = sum(d.get('outcome', 0.0) for d in data)
    drawdown = min(0.0, min(total_pnl - sum(d.get('outcome', 0.0) for d in data[:i+1]) for i in range(len(data)))) if data else 0.0
    rolling_sharpe = 0.0  # compute properly later with returns stddev
    return {
        'rolling_sharpe': rolling_sharpe,
        'hit_rate': hit_rate,
        'drawdown': drawdown,
        'total_pnl': total_pnl,
        'count': len(data),
        'timestamp': datetime.now().isoformat(),
    }


def write_daily_summary(summary: Dict[str, Any]) -> None:
    """Write consolidated daily snapshot to reports/summary.json."""
    _ensure_dirs()
    with _lock:
        with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)