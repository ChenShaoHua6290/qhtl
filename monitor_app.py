#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPL 比值 K 线监控服务（基于 TqSDK）

功能概览：
1. Web 页面可动态配置天勤账号、企业微信 webhook、监控品种组。
2. 支持多组品种比值监控（例如 L 主连 / V 主连）。
3. 每组品种同时监控 15 分钟、30 分钟级别信号。
4. 出现“扩大/缩小”信号后，发送企业微信机器人消息。
5. 页面展示“可读名称 <-> Tq 代码”映射，前端可读，后台用真实代码。

运行：
    python monitor_app.py

依赖：
    pip install flask pandas requests tqsdk
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string, request
from tqsdk import TqApi, TqAuth

# ---------------------------
# 基础配置
# ---------------------------
CONFIG_FILE = "config.json"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080

# 预置的“可读名称 -> 天勤主连代码”映射（可按需继续扩展）
# 注意：代码样式示例为 KQ.m@交易所.小写品种
HUMAN_CODE_MAP: Dict[str, str] = {
    "聚乙烯主连(L)": "KQ.m@DCE.l",
    "聚氯乙烯主连(V)": "KQ.m@DCE.v",
    "螺纹钢主连(RB)": "KQ.m@SHFE.rb",
    "热卷主连(HC)": "KQ.m@SHFE.hc",
    "铁矿主连(I)": "KQ.m@DCE.i",
    "焦炭主连(J)": "KQ.m@DCE.j",
    "豆粕主连(M)": "KQ.m@DCE.m",
    "豆油主连(Y)": "KQ.m@DCE.y",
    "棕榈油主连(P)": "KQ.m@DCE.p",
    "PTA主连(TA)": "KQ.m@CZCE.TA",
}


# ---------------------------
# 数据结构
# ---------------------------
@dataclass
class PairConfig:
    """单组监控品种配置。"""

    pair_id: str
    left_name: str
    left_code: str
    right_name: str
    right_code: str
    enabled: bool = True


@dataclass
class AppConfig:
    """应用总配置。"""

    tq_user: str = ""
    tq_password: str = ""
    wecom_webhook: str = ""
    poll_seconds: int = 15
    pairs: List[PairConfig] = field(default_factory=list)


# ---------------------------
# 工具函数：配置读写
# ---------------------------
def load_config() -> AppConfig:
    """从本地 JSON 读取配置，不存在则返回默认配置。"""
    if not os.path.exists(CONFIG_FILE):
        return AppConfig(
            pairs=[
                PairConfig(
                    pair_id=str(uuid.uuid4()),
                    left_name="聚乙烯主连(L)",
                    left_code=HUMAN_CODE_MAP["聚乙烯主连(L)"],
                    right_name="聚氯乙烯主连(V)",
                    right_code=HUMAN_CODE_MAP["聚氯乙烯主连(V)"],
                    enabled=True,
                )
            ]
        )

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pairs = [PairConfig(**p) for p in raw.get("pairs", [])]
    return AppConfig(
        tq_user=raw.get("tq_user", ""),
        tq_password=raw.get("tq_password", ""),
        wecom_webhook=raw.get("wecom_webhook", ""),
        poll_seconds=int(raw.get("poll_seconds", 15)),
        pairs=pairs,
    )


def save_config(cfg: AppConfig) -> None:
    """保存配置到本地 JSON，便于重启后继续使用。"""
    payload = {
        "tq_user": cfg.tq_user,
        "tq_password": cfg.tq_password,
        "wecom_webhook": cfg.wecom_webhook,
        "poll_seconds": cfg.poll_seconds,
        "pairs": [asdict(p) for p in cfg.pairs],
    }
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------
# 指标计算（对应用户提供脚本）
# ---------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    """EMA 指标，等效于脚本中的 EMA(X, N)。"""
    return series.ewm(span=span, adjust=False).mean()


def cross(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """上穿：本根 a>b 且上一根 a<=b。"""
    return (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))


def crossdown(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """下穿：本根 a<b 且上一根 a>=b。"""
    return (series_a < series_b) & (series_a.shift(1) >= series_b.shift(1))


def calc_ratio_signal(df: pd.DataFrame) -> Tuple[Optional[str], Dict[str, float]]:
    """
    根据“价差比值”K线计算信号。

    输入 df 需包含列：open_l/high_l/low_l/close_l + open_r/high_r/low_r/close_r
    返回：
      - signal: None / "扩大" / "缩小"
      - debug: 关键信息，方便前端展示
    """
    if len(df) < 220:
        return None, {"reason": "k线数量不足（至少需要220根）"}

    # 对应原脚本中的价差开高低收
    ratio_open = df["open_l"] / df["open_r"]
    ratio_close = df["close_l"] / df["close_r"]
    ratio_high = df["high_l"] / df["high_r"]
    ratio_low = df["low_l"] / df["low_r"]

    # 第一组 MACD
    diff = ema(ratio_close, 13)
    difl = ema(ratio_close, 21)
    macd = diff - difl
    signal = ema(macd, 8)
    hist = diff - difl - signal

    # 第二组 MACD（用于金叉死叉）
    diff1 = ema(ratio_close, 8)
    difl1 = ema(ratio_close, 13)
    macd1 = diff1 - difl1
    signal1 = ema(macd1, 5)

    jc = cross(macd1, signal1)      # 金叉
    sc = crossdown(macd1, signal1)  # 死叉

    # 双重 EMA 平滑线（DE20/DE55/DE144/DE166），用于展示与调试
    e1 = ema(ratio_close, 144)
    e2 = ema(e1, 144)
    de144 = 2 * e1 - e2

    e3 = ema(ratio_close, 166)
    e4 = ema(e3, 166)
    de166 = 2 * e3 - e4

    e5 = ema(ratio_close, 20)
    e6 = ema(e5, 20)
    de20 = 2 * e5 - e6

    e7 = ema(ratio_close, 55)
    e8 = ema(e7, 55)
    de55 = 2 * e7 - e8

    # 使用“已完成 K 线”判断信号，避免当前未收盘 bar 抖动
    idx = -2
    a_signal = bool(jc.iloc[idx] and hist.iloc[idx] < 0)  # 扩大
    b_signal = bool(sc.iloc[idx] and hist.iloc[idx] > 0)  # 缩小

    out_signal: Optional[str] = None
    if a_signal:
        out_signal = "扩大"
    elif b_signal:
        out_signal = "缩小"

    debug = {
        "ratio_close": float(ratio_close.iloc[idx]),
        "ratio_high": float(ratio_high.iloc[idx]),
        "ratio_low": float(ratio_low.iloc[idx]),
        "hist": float(hist.iloc[idx]),
        "macd1": float(macd1.iloc[idx]),
        "signal1": float(signal1.iloc[idx]),
        "de20": float(de20.iloc[idx]),
        "de55": float(de55.iloc[idx]),
        "de144": float(de144.iloc[idx]),
        "de166": float(de166.iloc[idx]),
    }
    return out_signal, debug


# ---------------------------
# 企业微信通知
# ---------------------------
def send_wecom(webhook: str, text: str) -> Tuple[bool, str]:
    """发送企业微信机器人文本消息。"""
    if not webhook:
        return False, "未配置 webhook"

    try:
        resp = requests.post(
            webhook,
            json={"msgtype": "text", "text": {"content": text}},
            timeout=10,
        )
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        if data.get("errcode") != 0:
            return False, f"errcode={data.get('errcode')} errmsg={data.get('errmsg')}"
        return True, "ok"
    except Exception as e:
        return False, str(e)


# ---------------------------
# 监控服务
# ---------------------------
class MonitorService:
    """后台监控线程：周期性拉取 K 线并判断信号。"""

    def __init__(self, app_config: AppConfig):
        self._cfg = app_config
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self.last_status: Dict[str, Dict] = {}
        # 防止同一个 bar 重复推送
        self._last_sent: Dict[str, int] = {}

    def update_config(self, new_cfg: AppConfig) -> None:
        with self._lock:
            self._cfg = new_cfg

    def get_config(self) -> AppConfig:
        with self._lock:
            return self._cfg

    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def _fetch_ratio_df(self, api: TqApi, left_code: str, right_code: str, duration: int, data_len: int = 260) -> pd.DataFrame:
        """
        拉取左右两个品种相同周期 K 线，并按 datetime 对齐后计算比值。
        duration 单位：秒（15分钟=900，30分钟=1800）
        """
        k_l = api.get_kline_serial(left_code, duration, data_length=data_len)
        k_r = api.get_kline_serial(right_code, duration, data_length=data_len)

        # 等待数据加载；最多等待几次，防止网络抖动卡死
        for _ in range(8):
            api.wait_update(deadline=time.time() + 2)

        df_l = pd.DataFrame({
            "datetime": k_l["datetime"],
            "open_l": k_l["open"],
            "high_l": k_l["high"],
            "low_l": k_l["low"],
            "close_l": k_l["close"],
        })
        df_r = pd.DataFrame({
            "datetime": k_r["datetime"],
            "open_r": k_r["open"],
            "high_r": k_r["high"],
            "low_r": k_r["low"],
            "close_r": k_r["close"],
        })

        df = pd.merge(df_l, df_r, on="datetime", how="inner")
        df = df.dropna().reset_index(drop=True)
        return df

    def _run(self) -> None:
        """主循环：连接 Tq，监控所有 pair 的 15m/30m 信号。"""
        while not self._stop_event.is_set():
            cfg = self.get_config()
            if not cfg.tq_user or not cfg.tq_password:
                self.last_status = {"system": {"error": "请先在页面配置天勤账号密码"}}
                time.sleep(2)
                continue

            api = None
            try:
                api = TqApi(auth=TqAuth(cfg.tq_user, cfg.tq_password))
                current_status: Dict[str, Dict] = {}

                for pair in cfg.pairs:
                    if not pair.enabled:
                        continue

                    pair_key = f"{pair.left_code}/{pair.right_code}"
                    current_status[pair_key] = {}

                    for tf_name, duration in (("15m", 900), ("30m", 1800)):
                        try:
                            df = self._fetch_ratio_df(api, pair.left_code, pair.right_code, duration)
                            sig, debug = calc_ratio_signal(df)
                            bar_dt = int(df["datetime"].iloc[-2]) if len(df) >= 2 else -1

                            current_status[pair_key][tf_name] = {
                                "signal": sig,
                                "bar_datetime_ns": bar_dt,
                                "debug": debug,
                                "updated_at": datetime.now(timezone.utc).isoformat(),
                            }

                            # 出现信号时通知，并确保“每个 bar 每个信号只发一次”
                            if sig:
                                dedup_key = f"{pair.pair_id}|{tf_name}|{sig}"
                                last_bar_sent = self._last_sent.get(dedup_key)
                                if last_bar_sent != bar_dt:
                                    msg = (
                                        f"【PPL比值信号】\n"
                                        f"品种组: {pair.left_name} / {pair.right_name}\n"
                                        f"代码: {pair.left_code} / {pair.right_code}\n"
                                        f"周期: {tf_name}\n"
                                        f"信号: {sig}\n"
                                        f"比值收盘: {debug.get('ratio_close'):.6f}\n"
                                        f"UTC时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
                                    )
                                    ok, reason = send_wecom(cfg.wecom_webhook, msg)
                                    current_status[pair_key][tf_name]["notify"] = {
                                        "ok": ok,
                                        "reason": reason,
                                    }
                                    if ok:
                                        self._last_sent[dedup_key] = bar_dt

                        except Exception as e:
                            current_status[pair_key][tf_name] = {
                                "error": str(e),
                                "trace": traceback.format_exc(limit=2),
                            }

                self.last_status = current_status

            except Exception as e:
                self.last_status = {"system": {"error": str(e), "trace": traceback.format_exc(limit=2)}}
            finally:
                if api is not None:
                    api.close()

            # 轮询间隔，可在页面动态调整
            sleep_seconds = max(5, int(cfg.poll_seconds))
            for _ in range(sleep_seconds):
                if self._stop_event.is_set():
                    break
                time.sleep(1)


# ---------------------------
# Flask 应用
# ---------------------------
app = Flask(__name__)
app_config = load_config()
service = MonitorService(app_config)


HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PPL比值监控</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
    input, select, button { margin: 4px; padding: 6px 8px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; }
    th { background: #f7f7f7; }
    .ok { color: green; }
    .warn { color: #b8860b; }
    .err { color: red; }
    code { background: #f1f1f1; padding: 2px 4px; }
  </style>
</head>
<body>
  <h2>PPL比值K线监控（TqSDK）</h2>

  <div class="card">
    <h3>全局配置</h3>
    <label>天勤账号: <input id="tq_user"/></label>
    <label>天勤密码: <input id="tq_password" type="password"/></label>
    <label>企业微信Webhook: <input id="wecom_webhook" style="width:420px"/></label>
    <label>轮询秒数: <input id="poll_seconds" type="number" min="5" value="15"/></label>
    <button onclick="saveGlobalConfig()">保存配置</button>
    <button onclick="startMonitor()">启动监控</button>
    <button onclick="stopMonitor()">停止监控</button>
    <span id="run_status"></span>
  </div>

  <div class="card">
    <h3>新增监控品种组</h3>
    <div>
      <label>左品种(可读): <select id="left_name"></select></label>
      <label>右品种(可读): <select id="right_name"></select></label>
    </div>
    <div>
      <label>左代码(可编辑): <input id="left_code" style="width:220px"/></label>
      <label>右代码(可编辑): <input id="right_code" style="width:220px"/></label>
      <button onclick="addPair()">添加</button>
    </div>
    <small>说明：页面显示可读名称，后台实际使用代码。你也可以手动修改代码用于自定义。</small>
  </div>

  <div class="card">
    <h3>监控列表</h3>
    <table>
      <thead>
        <tr>
          <th>名称</th><th>代码</th><th>启用</th><th>操作</th>
        </tr>
      </thead>
      <tbody id="pairs_body"></tbody>
    </table>
  </div>

  <div class="card">
    <h3>实时状态（15m / 30m）</h3>
    <pre id="state_box">loading...</pre>
  </div>

<script>
let humanMap = {};

function fillPairSelect() {
  const left = document.getElementById('left_name');
  const right = document.getElementById('right_name');
  left.innerHTML = ''; right.innerHTML = '';
  Object.keys(humanMap).forEach(name => {
    const o1 = document.createElement('option'); o1.value = name; o1.textContent = name;
    const o2 = document.createElement('option'); o2.value = name; o2.textContent = name;
    left.appendChild(o1); right.appendChild(o2);
  });
  left.onchange = () => document.getElementById('left_code').value = humanMap[left.value] || '';
  right.onchange = () => document.getElementById('right_code').value = humanMap[right.value] || '';
  left.dispatchEvent(new Event('change'));
  right.dispatchEvent(new Event('change'));
}

async function loadConfig() {
  const r = await fetch('/api/config');
  const data = await r.json();
  humanMap = data.human_map;
  fillPairSelect();

  document.getElementById('tq_user').value = data.config.tq_user || '';
  document.getElementById('tq_password').value = data.config.tq_password || '';
  document.getElementById('wecom_webhook').value = data.config.wecom_webhook || '';
  document.getElementById('poll_seconds').value = data.config.poll_seconds || 15;

  const body = document.getElementById('pairs_body');
  body.innerHTML = '';
  data.config.pairs.forEach(p => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${p.left_name} / ${p.right_name}</td>
                    <td><code>${p.left_code}</code> / <code>${p.right_code}</code></td>
                    <td>${p.enabled ? '是' : '否'}</td>
                    <td>
                      <button onclick="togglePair('${p.pair_id}')">切换启用</button>
                      <button onclick="deletePair('${p.pair_id}')">删除</button>
                    </td>`;
    body.appendChild(tr);
  });

  document.getElementById('run_status').textContent = data.running ? '运行中' : '未运行';
}

async function saveGlobalConfig() {
  const payload = {
    tq_user: document.getElementById('tq_user').value,
    tq_password: document.getElementById('tq_password').value,
    wecom_webhook: document.getElementById('wecom_webhook').value,
    poll_seconds: parseInt(document.getElementById('poll_seconds').value || '15', 10)
  };
  await fetch('/api/config', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  await loadConfig();
}

async function addPair() {
  const payload = {
    left_name: document.getElementById('left_name').value,
    right_name: document.getElementById('right_name').value,
    left_code: document.getElementById('left_code').value,
    right_code: document.getElementById('right_code').value,
  };
  await fetch('/api/pairs', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  await loadConfig();
}

async function deletePair(id) {
  await fetch('/api/pairs/' + id, {method: 'DELETE'});
  await loadConfig();
}

async function togglePair(id) {
  await fetch('/api/pairs/' + id + '/toggle', {method: 'POST'});
  await loadConfig();
}

async function startMonitor() {
  await fetch('/api/start', {method: 'POST'});
  await loadConfig();
}

async function stopMonitor() {
  await fetch('/api/stop', {method: 'POST'});
  await loadConfig();
}

async function pollState() {
  try {
    const r = await fetch('/api/state');
    const data = await r.json();
    document.getElementById('state_box').textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    document.getElementById('state_box').textContent = '状态拉取失败: ' + e;
  }
}

setInterval(pollState, 5000);
setInterval(loadConfig, 10000);
loadConfig();
pollState();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/config", methods=["GET"])
def get_config_api():
    cfg = service.get_config()
    return jsonify({
        "config": {
            "tq_user": cfg.tq_user,
            "tq_password": cfg.tq_password,
            "wecom_webhook": cfg.wecom_webhook,
            "poll_seconds": cfg.poll_seconds,
            "pairs": [asdict(p) for p in cfg.pairs],
        },
        "human_map": HUMAN_CODE_MAP,
        "running": service.running,
    })


@app.route("/api/config", methods=["POST"])
def set_config_api():
    data = request.get_json(force=True, silent=True) or {}
    cfg = service.get_config()
    cfg.tq_user = data.get("tq_user", cfg.tq_user)
    cfg.tq_password = data.get("tq_password", cfg.tq_password)
    cfg.wecom_webhook = data.get("wecom_webhook", cfg.wecom_webhook)
    cfg.poll_seconds = int(data.get("poll_seconds", cfg.poll_seconds))
    save_config(cfg)
    service.update_config(cfg)
    return jsonify({"ok": True})


@app.route("/api/pairs", methods=["POST"])
def add_pair_api():
    data = request.get_json(force=True, silent=True) or {}
    cfg = service.get_config()
    pair = PairConfig(
        pair_id=str(uuid.uuid4()),
        left_name=data.get("left_name", "左品种"),
        left_code=data.get("left_code", ""),
        right_name=data.get("right_name", "右品种"),
        right_code=data.get("right_code", ""),
        enabled=True,
    )
    if not pair.left_code or not pair.right_code:
        return jsonify({"ok": False, "error": "代码不能为空"}), 400

    cfg.pairs.append(pair)
    save_config(cfg)
    service.update_config(cfg)
    return jsonify({"ok": True, "pair": asdict(pair)})


@app.route("/api/pairs/<pair_id>", methods=["DELETE"])
def delete_pair_api(pair_id: str):
    cfg = service.get_config()
    cfg.pairs = [p for p in cfg.pairs if p.pair_id != pair_id]
    save_config(cfg)
    service.update_config(cfg)
    return jsonify({"ok": True})


@app.route("/api/pairs/<pair_id>/toggle", methods=["POST"])
def toggle_pair_api(pair_id: str):
    cfg = service.get_config()
    for p in cfg.pairs:
        if p.pair_id == pair_id:
            p.enabled = not p.enabled
            break
    save_config(cfg)
    service.update_config(cfg)
    return jsonify({"ok": True})


@app.route("/api/start", methods=["POST"])
def start_api():
    service.start()
    return jsonify({"ok": True, "running": service.running})


@app.route("/api/stop", methods=["POST"])
def stop_api():
    service.stop()
    return jsonify({"ok": True, "running": service.running})


@app.route("/api/state", methods=["GET"])
def state_api():
    return jsonify({
        "running": service.running,
        "status": service.last_status,
        "ts": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    # 启动 Flask 服务；首次运行请先在页面填写账号等配置。
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=False)
