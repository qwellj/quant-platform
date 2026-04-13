# =============================================================================
# 量化投资分析平台 v3.0
# 新增: 基准对比、年度收益、PDF导出、基本面选股界面
# =============================================================================

import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib
import anthropic
import itertools
import platform
import os
import io
import time

if platform.system() == "Darwin":
    matplotlib.rcParams["font.family"] = "Arial Unicode MS"
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
os.makedirs("data", exist_ok=True)

# =============================================================================
# 工具函数
# =============================================================================

def _fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf


def get_stock_name(code):
    """获取股票名称"""
    try:
        df = ak.stock_individual_info_em(symbol=code)
        row = df[df.iloc[:,0] == "股票简称"]
        if len(row) > 0:
            return row.iloc[0, 1]
    except:
        pass
    return code


# =============================================================================
# 数据函数
# =============================================================================

def download_stock(code, start="20200101", end="20241231"):
    prefix = "sh" if code.startswith("6") else "sz"
    df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.to_datetime(start)) &
            (df["date"] <= pd.to_datetime(end))]
    df.set_index("date", inplace=True)
    df.to_csv(f"data/{code}.csv")
    return df


def get_benchmark(start="20200101", end="20241231"):
    """下载沪深300基准"""
    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= pd.to_datetime(start)) &
                (df["date"] <= pd.to_datetime(end))]
        df.set_index("date", inplace=True)
        return df["close"]
    except:
        return None


def calc_indicators(code):
    df = pd.read_csv(f"data/{code}.csv", index_col="date", parse_dates=True)
    df["ma5"]      = df["close"].rolling(5).mean()
    df["ma10"]     = df["close"].rolling(10).mean()
    df["ma20"]     = df["close"].rolling(20).mean()
    ema12          = df["close"].ewm(span=12).mean()
    ema26          = df["close"].ewm(span=26).mean()
    df["macd"]     = ema12 - ema26
    df["signal"]   = df["macd"].ewm(span=9).mean()
    df["hist"]     = df["macd"] - df["signal"]
    delta          = df["close"].diff()
    gain           = delta.clip(lower=0).rolling(14).mean()
    loss           = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]      = 100 - (100 / (1 + gain / loss))
    df["bb_mid"]   = df["close"].rolling(20).mean()
    df["bb_upper"] = df["bb_mid"] + 2 * df["close"].rolling(20).std()
    df["bb_lower"] = df["bb_mid"] - 2 * df["close"].rolling(20).std()
    df.to_csv(f"data/{code}_indicators.csv")
    return df


# =============================================================================
# 策略回测
# =============================================================================

def calc_stats(nav, trades, initial_cash):
    nav_s     = pd.Series(nav)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    total_ret = (nav_s.iloc[-1] / initial_cash - 1) * 100
    daily_ret = nav_s.pct_change().dropna()
    sharpe    = daily_ret.mean() / daily_ret.std() * (252 ** 0.5) if daily_ret.std() > 0 else 0
    max_dd    = ((nav_s - nav_s.cummax()) / nav_s.cummax()).min() * 100
    stats = {
        "total_ret": round(total_ret, 2),
        "sharpe":    round(sharpe, 3),
        "max_dd":    round(max_dd, 2),
        "n_trades":  len(trades_df),
        "n_stop":    len(trades_df[trades_df["action"]=="止损"])    if len(trades_df) > 0 else 0,
        "n_profit":  len(trades_df[trades_df["action"]=="止盈"])    if len(trades_df) > 0 else 0,
        "n_signal":  len(trades_df[trades_df["action"]=="信号卖出"]) if len(trades_df) > 0 else 0,
        "final_nav": round(nav_s.iloc[-1], 0),
    }
    return nav_s, trades_df, stats


def backtest_ma(code, fast=5, slow=40, initial_cash=1_000_000,
                stop_loss=0.10, take_profit=0.30,
                vol_filter=False, vol_mult=1.3,
                macd_filter=False):
    """
    双均线策略回测。

    新增过滤参数（两者可单独或同时开启）：
        vol_filter  : 成交量过滤。金叉买入时要求成交量 > 20日均量 × vol_mult
                      作用：排除缩量假突破，只在放量时入场
        macd_filter : MACD方向过滤。买入时要求 MACD柱状图(hist) > 0
                      作用：确认动能向上，避免"反弹中的死猫跳"
    """
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    df["maf"]    = df["close"].rolling(fast).mean()
    df["mas"]    = df["close"].rolling(slow).mean()
    df["vol_ma"] = df["volume"].rolling(20).mean()   # 20日均量
    df = df.dropna()
    cash, position, buy_price = initial_cash, 0, 0
    nav, trades = [], []
    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        if position > 0:
            pnl = (price - buy_price) / buy_price
            if pnl < -stop_loss:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "止损", "price": price, "pnl_pct": round(pnl*100,2)})
                position = 0
            elif pnl > take_profit:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "止盈", "price": price, "pnl_pct": round(pnl*100,2)})
                position = 0
            elif df["maf"].iloc[i-1] > df["mas"].iloc[i-1] and df["maf"].iloc[i] < df["mas"].iloc[i]:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "信号卖出", "price": price, "pnl_pct": round(pnl*100,2)})
                position = 0
        elif df["maf"].iloc[i-1] < df["mas"].iloc[i-1] and df["maf"].iloc[i] > df["mas"].iloc[i]:
            # ── 过滤条件判断 ──
            vol_ok  = (df["volume"].iloc[i] > df["vol_ma"].iloc[i] * vol_mult) if vol_filter  else True
            macd_ok = (df["hist"].iloc[i] > 0)                                  if macd_filter else True

            if vol_ok and macd_ok:
                shares = int(cash / price / 100) * 100
                if shares > 0:
                    cash -= shares * price * 1.001
                    position, buy_price = shares, price
                    # 记录触发了哪些过滤条件
                    filters = []
                    if vol_filter:  filters.append(f"放量×{vol_mult}")
                    if macd_filter: filters.append("MACD↑")
                    tag = "+".join(filters) if filters else "无过滤"
                    trades.append({"date": df.index[i], "action": "买入",
                                   "price": price, "pnl_pct": 0, "filter": tag})
        nav.append(cash + position * price)
    nav_s, trades_df, stats = calc_stats(nav, trades, initial_cash)
    nav_s.index = df.index[1:]
    return nav_s, trades_df, stats


def backtest_rsi(code, rsi_buy=30, rsi_sell=70, initial_cash=1_000_000,
                 stop_loss=0.10, take_profit=0.30):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    cash, position, buy_price = initial_cash, 0, 0
    nav, trades = [], []
    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        rsi   = df["rsi"].iloc[i]
        if position > 0:
            pnl = (price - buy_price) / buy_price
            if pnl < -stop_loss:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "止损", "price": price, "pnl_pct": round(pnl*100,2)})
                position = 0
            elif pnl > take_profit:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "止盈", "price": price, "pnl_pct": round(pnl*100,2)})
                position = 0
            elif rsi > rsi_sell:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "信号卖出", "price": price, "pnl_pct": round(pnl*100,2)})
                position = 0
        elif rsi < rsi_buy and position == 0:
            shares = int(cash / price / 100) * 100
            if shares > 0:
                cash -= shares * price * 1.001
                position, buy_price = shares, price
                trades.append({"date": df.index[i], "action": "买入", "price": price, "pnl_pct": 0})
        nav.append(cash + position * price)
    nav_s, trades_df, stats = calc_stats(nav, trades, initial_cash)
    nav_s.index = df.index[1:]
    return nav_s, trades_df, stats


def backtest_boll(code, initial_cash=1_000_000, stop_loss=0.10, take_profit=0.30):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    cash, position, buy_price = initial_cash, 0, 0
    nav, trades = [], []
    for i in range(1, len(df)):
        price      = df["close"].iloc[i]
        upper      = df["bb_upper"].iloc[i]
        mid        = df["bb_mid"].iloc[i]
        prev_close = df["close"].iloc[i-1]
        prev_upper = df["bb_upper"].iloc[i-1]
        if position > 0:
            pnl = (price - buy_price) / buy_price
            if pnl < -stop_loss:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "止损", "price": price, "pnl_pct": round(pnl*100,2)})
                position = 0
            elif pnl > take_profit:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "止盈", "price": price, "pnl_pct": round(pnl*100,2)})
                position = 0
            elif price < mid:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "信号卖出", "price": price, "pnl_pct": round(pnl*100,2)})
                position = 0
        elif prev_close < prev_upper and price > upper and position == 0:
            shares = int(cash / price / 100) * 100
            if shares > 0:
                cash -= shares * price * 1.001
                position, buy_price = shares, price
                trades.append({"date": df.index[i], "action": "买入", "price": price, "pnl_pct": 0})
        nav.append(cash + position * price)
    nav_s, trades_df, stats = calc_stats(nav, trades, initial_cash)
    nav_s.index = df.index[1:]
    return nav_s, trades_df, stats


def backtest_macd(code, initial_cash=1_000_000, stop_loss=0.10, take_profit=0.30):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    cash, position, buy_price = initial_cash, 0, 0
    nav, trades = [], []
    for i in range(1, len(df)):
        price       = df["close"].iloc[i]
        macd_prev   = df["macd"].iloc[i-1]
        signal_prev = df["signal"].iloc[i-1]
        macd_curr   = df["macd"].iloc[i]
        signal_curr = df["signal"].iloc[i]
        if position > 0:
            pnl = (price - buy_price) / buy_price
            if pnl < -stop_loss:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "止损", "price": price, "pnl_pct": round(pnl*100,2)})
                position = 0
            elif pnl > take_profit:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "止盈", "price": price, "pnl_pct": round(pnl*100,2)})
                position = 0
            elif macd_prev > signal_prev and macd_curr < signal_curr:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "信号卖出", "price": price, "pnl_pct": round(pnl*100,2)})
                position = 0
        elif macd_prev < signal_prev and macd_curr > signal_curr and position == 0:
            shares = int(cash / price / 100) * 100
            if shares > 0:
                cash -= shares * price * 1.001
                position, buy_price = shares, price
                trades.append({"date": df.index[i], "action": "买入", "price": price, "pnl_pct": 0})
        nav.append(cash + position * price)
    nav_s, trades_df, stats = calc_stats(nav, trades, initial_cash)
    nav_s.index = df.index[1:]
    return nav_s, trades_df, stats


def optimize_ma_params(code):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    results = []
    for fast, slow in itertools.product([5,10,15,20], [20,30,40,60]):
        if fast >= slow:
            continue
        maf   = df["close"].rolling(fast).mean()
        mas   = df["close"].rolling(slow).mean()
        valid = maf.notna() & mas.notna()
        maf, mas, price_s = maf[valid], mas[valid], df["close"][valid]
        cash, position, nav = 1_000_000, 0, []
        for i in range(1, len(price_s)):
            price = price_s.iloc[i]
            if maf.iloc[i-1] < mas.iloc[i-1] and maf.iloc[i] > mas.iloc[i] and position == 0:
                shares = int(cash / price / 100) * 100
                if shares > 0:
                    cash -= shares * price * 1.001
                    position = shares
            elif maf.iloc[i-1] > mas.iloc[i-1] and maf.iloc[i] < mas.iloc[i] and position > 0:
                cash += position * price * 0.999
                position = 0
            nav.append(cash + position * price)
        nav_s     = pd.Series(nav)
        daily_ret = nav_s.pct_change().dropna()
        sharpe    = daily_ret.mean() / daily_ret.std() * (252 ** 0.5) if daily_ret.std() > 0 else 0
        total_ret = (nav_s.iloc[-1] / 1_000_000 - 1) * 100
        max_dd    = ((nav_s - nav_s.cummax()) / nav_s.cummax()).min() * 100
        results.append({"fast": fast, "slow": slow,
                        "total_ret": round(total_ret, 2),
                        "sharpe": round(sharpe, 3),
                        "max_dd": round(max_dd, 2)})
    return pd.DataFrame(results).sort_values("sharpe", ascending=False)


# =============================================================================
# 基准对比 & 年度收益
# =============================================================================

def calc_annual_returns(nav_s):
    """按年计算收益率"""
    annual = {}
    for year in nav_s.index.year.unique():
        yr = nav_s[nav_s.index.year == year]
        if len(yr) < 2:
            continue
        annual[year] = round((yr.iloc[-1] / yr.iloc[0] - 1) * 100, 2)
    return pd.Series(annual)


def plot_benchmark_and_annual(code, nav_s, benchmark, strategy_name):
    """生成基准对比 + 年度收益两图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1：策略 vs 沪深300
    ax1 = axes[0]
    nav_norm = nav_s / nav_s.iloc[0]
    ax1.plot(nav_norm, label=f"Strategy ({strategy_name})",
             color="#4CAF50", linewidth=1.5)
    if benchmark is not None:
        bm_aligned = benchmark.reindex(nav_s.index, method="ffill").dropna()
        bm_norm    = bm_aligned / bm_aligned.iloc[0]
        ax1.plot(bm_norm, label="CSI 300 (Benchmark)",
                 color="#2196F3", linewidth=1.5, linestyle="--")
        excess = round((nav_norm.iloc[-1] - bm_norm.iloc[-1]) * 100, 2)
        ax1.set_title(f"{code} vs CSI300  (Excess: {excess:+.1f}%)", fontsize=12)
    else:
        ax1.set_title(f"{code} Strategy NAV", fontsize=12)
    ax1.axhline(1, color="gray", linestyle=":", linewidth=0.8)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # 图2：年度收益柱状图
    ax2 = axes[1]
    annual = calc_annual_returns(nav_s)
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in annual.values]
    bars   = ax2.bar(annual.index.astype(str), annual.values, color=colors)
    ax2.axhline(0, color="gray", linewidth=0.8)
    for bar, val in zip(bars, annual.values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (1 if val >= 0 else -3),
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_title(f"{code} Annual Returns (%)", fontsize=12)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    return fig, annual


# =============================================================================
# 可视化函数
# =============================================================================

def plot_ma(code, nav_s, fast, slow):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].plot(df["close"], label="Close", color="#2196F3", linewidth=1)
    axes[0].plot(df["close"].rolling(fast).mean(), label=f"MA{fast}", color="#FF9800", linewidth=1)
    axes[0].plot(df["close"].rolling(slow).mean(), label=f"MA{slow}", color="#F44336", linewidth=1)
    axes[0].set_title(f"{code} MA Crossover (MA{fast}/MA{slow})", fontsize=13)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot(nav_s / nav_s.iloc[0], color="#4CAF50", linewidth=1.5, label="NAV")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("NAV"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[2].plot(df["rsi"], color="#9C27B0", linewidth=1, label="RSI(14)")
    axes[2].axhline(70, color="red", linestyle="--", linewidth=0.8)
    axes[2].axhline(30, color="green", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("RSI"); axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_rsi(code, nav_s, rsi_buy, rsi_sell):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].plot(df["close"], label="Close", color="#2196F3", linewidth=1)
    axes[0].set_title(f"{code} RSI Strategy (Buy<{rsi_buy} / Sell>{rsi_sell})", fontsize=13)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot(nav_s / nav_s.iloc[0], color="#4CAF50", linewidth=1.5, label="NAV")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("NAV"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[2].plot(df["rsi"], color="#9C27B0", linewidth=1, label="RSI(14)")
    axes[2].axhline(rsi_sell, color="red",   linestyle="--", linewidth=1, label=f"OB {rsi_sell}")
    axes[2].axhline(rsi_buy,  color="green", linestyle="--", linewidth=1, label=f"OS {rsi_buy}")
    axes[2].fill_between(df.index, rsi_buy, df["rsi"].clip(upper=rsi_buy), alpha=0.2, color="green")
    axes[2].fill_between(df.index, rsi_sell, df["rsi"].clip(lower=rsi_sell), alpha=0.2, color="red")
    axes[2].set_ylabel("RSI"); axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_boll(code, nav_s):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(df["close"],    label="Close",  color="#2196F3", linewidth=1)
    axes[0].plot(df["bb_upper"], label="Upper",  color="#F44336", linewidth=1, linestyle="--")
    axes[0].plot(df["bb_mid"],   label="Middle", color="#FF9800", linewidth=1, linestyle="--")
    axes[0].plot(df["bb_lower"], label="Lower",  color="#4CAF50", linewidth=1, linestyle="--")
    axes[0].fill_between(df.index, df["bb_upper"], df["bb_lower"], alpha=0.08, color="gray")
    axes[0].set_title(f"{code} Bollinger Band Strategy", fontsize=13)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot(nav_s / nav_s.iloc[0], color="#4CAF50", linewidth=1.5, label="NAV")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("NAV"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_macd(code, nav_s):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].plot(df["close"], label="Close", color="#2196F3", linewidth=1)
    axes[0].set_title(f"{code} MACD Strategy", fontsize=13)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot(nav_s / nav_s.iloc[0], color="#4CAF50", linewidth=1.5, label="NAV")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("NAV"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[2].plot(df["macd"],   label="MACD",   color="#2196F3", linewidth=1)
    axes[2].plot(df["signal"], label="Signal", color="#F44336", linewidth=1)
    colors = ["#F44336" if v < 0 else "#4CAF50" for v in df["hist"]]
    axes[2].bar(df.index, df["hist"], color=colors, alpha=0.6)
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].set_ylabel("MACD"); axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    return fig


# =============================================================================
# PDF 报告生成
# =============================================================================

def generate_pdf(code, stock_name, strategy_name, stats, annual,
                 img_main_buf, img_bench_buf):
    """生成多页 PDF 分析报告（A4横版，统一尺寸）"""
    from PIL import Image
    PAGE_W, PAGE_H = 14, 9  # 统一页面尺寸（英寸）
    buf = io.BytesIO()
    with pdf_backend.PdfPages(buf) as pdf:

        # 封面
        fig0, ax0 = plt.subplots(figsize=(PAGE_W, PAGE_H))
        ax0.set_facecolor("#F8FAFF")
        fig0.patch.set_facecolor("#F8FAFF")
        ax0.axis("off")
        ax0.text(0.5, 0.80, "Quantitative Analysis Report",
                 ha="center", va="center", fontsize=32, fontweight="bold",
                 transform=ax0.transAxes, color="#1A237E")
        ax0.text(0.5, 0.65, f"Stock Code: {code}",
                 ha="center", va="center", fontsize=22,
                 transform=ax0.transAxes, color="#1565C0")
        ax0.text(0.5, 0.55, f"Strategy: {strategy_name}",
                 ha="center", va="center", fontsize=18,
                 transform=ax0.transAxes, color="#37474F")
        ax0.plot([0.15, 0.85], [0.48, 0.48], color="#BBDEFB", linewidth=1.5,
                 transform=ax0.transAxes)
        metrics = [
            ("Total Return",  f"{stats['total_ret']}%"),
            ("Sharpe Ratio",  f"{stats['sharpe']}"),
            ("Max Drawdown",  f"{stats['max_dd']}%"),
            ("Total Trades",  f"{stats['n_trades']}"),
            ("Final NAV",     f"{stats['final_nav']:,.0f}"),
        ]
        for idx, (label, value) in enumerate(metrics):
            x = 0.12 + idx * 0.19
            ax0.text(x, 0.36, value, ha="center", fontsize=17, fontweight="bold",
                     transform=ax0.transAxes, color="#1565C0")
            ax0.text(x, 0.28, label, ha="center", fontsize=11,
                     transform=ax0.transAxes, color="#78909C")
        ax0.text(0.5, 0.08, "Generated by Quant Platform v3.0  |  For Research Purposes Only",
                 ha="center", fontsize=10, color="#B0BEC5",
                 transform=ax0.transAxes)
        pdf.savefig(fig0, bbox_inches="tight")
        plt.close(fig0)

        # 图表页：统一尺寸嵌入
        for img_buf, title in [
            (img_main_buf, f"{code} - Strategy Chart"),
            (img_bench_buf, f"{code} - Benchmark Comparison & Annual Returns")
        ]:
            img_buf.seek(0)
            img = Image.open(img_buf)
            fig_img, ax_img = plt.subplots(figsize=(PAGE_W, PAGE_H))
            fig_img.patch.set_facecolor("white")
            ax_img.set_facecolor("white")
            # 留出标题空间，图片填满剩余区域
            ax_img.imshow(img, aspect="auto")
            ax_img.axis("off")
            fig_img.suptitle(title, fontsize=14, fontweight="bold",
                             color="#1A237E", y=0.97)
            fig_img.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig_img, bbox_inches="tight", facecolor="white")
            plt.close(fig_img)

        # 年度收益表格页
        fig_tbl, ax_tbl = plt.subplots(figsize=(PAGE_W, PAGE_H))
        fig_tbl.patch.set_facecolor("white")
        ax_tbl.set_facecolor("white")
        ax_tbl.axis("off")
        tbl_data = [[str(y), f"{v:.2f}%", "▲" if v >= 0 else "▼"]
                    for y, v in annual.items()]
        tbl = ax_tbl.table(
            cellText=tbl_data,
            colLabels=["Year", "Return (%)", "Direction"],
            loc="center", cellLoc="center",
            bbox=[0.2, 0.1, 0.6, 0.75]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(15)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#E0E0E0")
            if row == 0:
                cell.set_facecolor("#1565C0")
                cell.set_text_props(color="white", fontweight="bold")
            elif row > 0 and tbl_data[row-1][2] == "▲":
                cell.set_facecolor("#E8F5E9")
                cell.set_text_props(color="#2E7D32")
            else:
                cell.set_facecolor("#FFEBEE")
                cell.set_text_props(color="#C62828")
        fig_tbl.suptitle(f"{code} - Annual Returns Summary",
                         fontsize=14, fontweight="bold", color="#1A237E", y=0.95)
        pdf.savefig(fig_tbl, bbox_inches="tight", facecolor="white")
        plt.close(fig_tbl)

    buf.seek(0)
    return buf


# =============================================================================
# AI 分析
# =============================================================================

def ai_analysis(code, strategy_name, stats, api_key, extra_info=""):
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""
你是一位专业的量化投资分析师，请对以下A股股票回测结果进行深度分析，用中文回答。

股票代码: {code}
使用策略: {strategy_name}
回测区间: 2020-2024年（5年）
{extra_info}

回测绩效指标:
- 总收益率: {stats['total_ret']}%
- 夏普比率: {stats['sharpe']}
- 最大回撤: {stats['max_dd']}%
- 总交易次数: {stats['n_trades']} 笔
- 止损触发: {stats['n_stop']} 次
- 止盈触发: {stats['n_profit']} 次
- 信号卖出: {stats['n_signal']} 次
- 最终净值: ¥{stats['final_nav']:,.0f}（初始¥1,000,000）

请从以下几个维度进行分析（每个维度2-3句话）:
1. 整体表现评价（收益和风险是否匹配）
2. 策略适配性（这只股票适不适合该策略，为什么）
3. 风险提示（主要风险点在哪里）
4. 改进建议（如何进一步优化）
5. 一句话投资结论

请保持专业、客观，避免给出明确的买卖建议。
"""
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


# =============================================================================
# 月度因子回测函数（新增）
# =============================================================================

def batch_download_pool(codes: list, start: str, end: str) -> dict:
    """
    批量下载股票池历史价格，返回 {code: close_series}
    失败的股票自动跳过，每只间隔0.3秒防限流
    """
    price_data = {}
    for code in codes:
        try:
            prefix = "sh" if code.startswith("6") else "sz"
            df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= pd.to_datetime(start)) &
                    (df["date"] <= pd.to_datetime(end))]
            df.set_index("date", inplace=True)
            if len(df) > 120:           # 至少半年数据才入池
                price_data[code] = df["close"]
        except:
            pass
        time.sleep(0.3)
    return price_data


def score_stocks_at_date(price_data: dict, rebal_date: pd.Timestamp) -> pd.DataFrame:
    """
    在某个月末日期，对股票池里所有股票计算因子得分。

    因子说明：
        momentum_12_1：
            = 12个月前收盘价 → 1个月前收盘价的涨幅
            跳过最近1个月是为了规避"短期反转"效应
            （刚大涨的股票短期内往往有回调，跳过1月可以避开这个噪声）
            这是A股文献中最稳定的价格因子之一

        trend_filter：
            = 当前价格 > MA60（60日均线）
            只选择趋势向上的股票，过滤掉下跌趋势中的"价值陷阱"
            在均线因子失效的今天，这个过滤仍有一定的排除作用

    返回：
        DataFrame，columns=[momentum, trend, score]，index=股票代码
    """
    records = {}
    for code, ps in price_data.items():
        p = ps[ps.index <= rebal_date].dropna()
        if len(p) < 252:                # 至少1年历史
            continue

        # 动量：12个月前(约252日)到1个月前(约21日)
        p_12m = p.iloc[-252]
        p_1m  = p.iloc[-21]
        if p_12m <= 0:
            continue
        momentum = (p_1m / p_12m) - 1

        # 趋势过滤
        ma60  = p.rolling(60).mean().iloc[-1]
        trend = 1 if p.iloc[-1] > ma60 else 0

        records[code] = {"momentum": momentum, "trend": trend}

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).T
    df["mom_rank"] = df["momentum"].rank(pct=True)   # 百分位排名

    # 综合得分：动量排名 × 趋势过滤
    # trend=0 的股票得分为0，不会被选中
    df["score"]   = df["mom_rank"] * df["trend"]
    return df.sort_values("score", ascending=False)


def backtest_factor_monthly(price_data: dict,
                             start_date: str,
                             end_date:   str,
                             n_stocks:   int   = 20,
                             initial_cash: float = 1_000_000) -> tuple:
    """
    月度调仓因子回测引擎

    逻辑：
        每个月末 → 计算因子得分 → 选前n_stocks只
        → 卖掉不在新选股里的 → 买入新增股票（等权）
        → 持有到下月末 → 循环

    手续费：买入0.1%，卖出0.1%（含印花税）
    持仓：等权，即每只股票分配相同金额

    注意：月度换仓每年约12次，远少于均线策略的30-40次
    换手率低 → 手续费少 → 策略净收益更真实
    """
    # 生成月末再平衡日期序列
    all_month_ends = pd.date_range(start_date, end_date, freq="ME")
    if len(all_month_ends) < 3:
        return pd.Series(dtype=float), [], pd.DataFrame()

    # 合并所有股票价格为宽表（日期 x 股票）
    price_df = pd.DataFrame(price_data).sort_index()

    cash        = initial_cash
    holdings    = {}        # {code: shares}
    nav_records = []
    rebal_log   = []

    for i in range(len(all_month_ends) - 1):
        rebal_date = all_month_ends[i]
        next_date  = all_month_ends[i + 1]

        # ── 当月末：计算得分，决定新持仓 ──
        scores = score_stocks_at_date(price_data, rebal_date)
        if scores.empty:
            continue

        new_selection = scores.head(n_stocks).index.tolist()

        # 卖出不在新选股里的（月末收盘价成交）
        to_sell = [c for c in list(holdings.keys()) if c not in new_selection]
        for code in to_sell:
            if code in price_df.columns:
                idx = price_df.index.get_indexer([rebal_date], method="ffill")[0]
                if idx >= 0:
                    sell_px = price_df.iloc[idx][code]
                    if not pd.isna(sell_px) and sell_px > 0:
                        cash += holdings[code] * sell_px * 0.999
            del holdings[code]

        # 买入新增的股票（等权分配剩余现金）
        to_buy   = [c for c in new_selection if c not in holdings]
        n_buy    = len(to_buy)
        if n_buy > 0 and cash > 1000:
            per_amt = cash * 0.99 / n_buy     # 留1%现金缓冲
            for code in to_buy:
                if code not in price_df.columns:
                    continue
                idx = price_df.index.get_indexer([rebal_date], method="ffill")[0]
                if idx < 0:
                    continue
                buy_px = price_df.iloc[idx][code]
                if pd.isna(buy_px) or buy_px <= 0:
                    continue
                shares = int(per_amt / buy_px / 100) * 100
                if shares > 0:
                    cost = shares * buy_px * 1.001
                    if cost <= cash:
                        cash -= cost
                        holdings[code] = shares

        # 记录换仓情况
        rebal_log.append({
            "换仓日":   rebal_date.strftime("%Y-%m"),
            "新选股":   ", ".join(new_selection[:5]) + ("..." if len(new_selection) > 5 else ""),
            "持仓只数": len(holdings),
            "现金余额": round(cash, 0),
        })

        # ── 当月：逐日计算净值 ──
        period_idx = price_df[(price_df.index > rebal_date) &
                               (price_df.index <= next_date)].index
        for d in period_idx:
            pv = cash
            for code, sh in holdings.items():
                if code in price_df.columns:
                    px = price_df.loc[d, code]
                    if not pd.isna(px):
                        pv += sh * px
            nav_records.append({"date": d, "nav": pv})

    if not nav_records:
        return pd.Series(dtype=float), rebal_log, pd.DataFrame()

    nav_s   = pd.DataFrame(nav_records).set_index("date")["nav"]
    rebal_df = pd.DataFrame(rebal_log)
    return nav_s, rebal_log, rebal_df


def plot_factor_result(nav_s: pd.Series, benchmark: pd.Series,
                       rebal_df: pd.DataFrame, n_stocks: int) -> bytes:
    """生成因子回测三合一图表"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 图1：策略净值 vs 沪深300
    ax1 = axes[0]
    nav_norm = nav_s / nav_s.iloc[0]
    ax1.plot(nav_norm, color="#1D9E75", linewidth=1.5,
             label=f"因子策略(Top{n_stocks})")
    ax1.fill_between(nav_norm.index, nav_norm, 1,
                     where=(nav_norm < 1), alpha=0.15, color="#E24B4A")
    if benchmark is not None:
        bm = benchmark.reindex(nav_s.index, method="ffill").dropna()
        if len(bm) > 0:
            bm_norm = bm / bm.iloc[0]
            ax1.plot(bm_norm, color="#378ADD", linewidth=1.2,
                     linestyle="--", label="沪深300")
    ax1.axhline(1, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_title("策略净值 vs 基准", fontsize=11)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # 图2：年度收益柱状图
    ax2 = axes[1]
    annual = {}
    for yr in nav_s.index.year.unique():
        yr_nav = nav_s[nav_s.index.year == yr]
        if len(yr_nav) > 2:
            annual[yr] = round((yr_nav.iloc[-1] / yr_nav.iloc[0] - 1) * 100, 1)
    if annual:
        yrs    = list(annual.keys())
        vals   = list(annual.values())
        colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in vals]
        bars   = ax2.bar([str(y) for y in yrs], vals, color=colors, alpha=0.8)
        ax2.axhline(0, color="gray", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (0.5 if v >= 0 else -2),
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_title("年度收益", fontsize=11)
    ax2.grid(alpha=0.3, axis="y")

    # 图3：回撤曲线
    ax3 = axes[2]
    nav_norm2 = nav_s / nav_s.iloc[0]
    drawdown  = (nav_norm2 - nav_norm2.cummax()) / nav_norm2.cummax() * 100
    ax3.fill_between(drawdown.index, drawdown, 0,
                     alpha=0.4, color="#E24B4A", label="回撤")
    ax3.plot(drawdown, color="#E24B4A", linewidth=0.8)
    ax3.set_title("回撤曲线", fontsize=11)
    ax3.set_ylabel("%"); ax3.grid(alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    buf.seek(0)
    plt.close(fig)
    return buf.read()


# =============================================================================
# Walk-Forward 验证函数（新增）
# =============================================================================

def _wfv_run(price_s: pd.Series, fast: int, slow: int,
             vol_s: pd.Series = None, vol_filter: bool = False, vol_mult: float = 1.3,
             hist_s: pd.Series = None, macd_filter: bool = False) -> dict:
    """WFV内部回测（支持与backtest_ma相同的过滤条件）"""
    maf = price_s.rolling(fast).mean()
    mas = price_s.rolling(slow).mean()
    ok  = maf.notna() & mas.notna()
    maf, mas, ps = maf[ok], mas[ok], price_s[ok]

    # 对齐可选序列
    vol_ma = vol_s.rolling(20).mean()[ok] if (vol_filter and vol_s is not None) else None
    vol_a  = vol_s[ok]                     if (vol_filter and vol_s is not None) else None
    hist_a = hist_s[ok]                    if (macd_filter and hist_s is not None) else None

    cash, pos, nav = 1_000_000, 0, []
    for i in range(1, len(ps)):
        p = ps.iloc[i]
        if maf.iloc[i-1] < mas.iloc[i-1] and maf.iloc[i] >= mas.iloc[i] and pos == 0:
            vol_ok  = (vol_a.iloc[i] > vol_ma.iloc[i] * vol_mult) if vol_a  is not None else True
            macd_ok = (hist_a.iloc[i] > 0)                         if hist_a is not None else True
            if vol_ok and macd_ok:
                sh = int(cash / p / 100) * 100
                if sh > 0:
                    cash -= sh * p * 1.001; pos = sh
        elif maf.iloc[i-1] > mas.iloc[i-1] and maf.iloc[i] <= mas.iloc[i] and pos > 0:
            cash += pos * p * 0.999; pos = 0
        nav.append(cash + pos * p)
    if not nav:
        return {"sharpe": 0, "total_ret": 0, "nav": pd.Series(dtype=float)}
    nav_s = pd.Series(nav, index=ps.index[1:])
    dr    = nav_s.pct_change().dropna()
    sh    = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    return {
        "sharpe":    round(float(sh), 4),
        "total_ret": round((nav_s.iloc[-1] / 1_000_000 - 1) * 100, 2),
        "nav":       nav_s,
    }


def _wfv_optimize(price_s: pd.Series,
                  vol_s: pd.Series = None, vol_filter: bool = False, vol_mult: float = 1.3,
                  hist_s: pd.Series = None, macd_filter: bool = False) -> tuple:
    """在给定序列上网格搜索最优MA参数（支持过滤）"""
    best = (-999, 5, 40)
    for f, s in itertools.product([5, 10, 15, 20], [20, 30, 40, 60]):
        if f >= s:
            continue
        r = _wfv_run(price_s, f, s, vol_s, vol_filter, vol_mult, hist_s, macd_filter)
        if r["sharpe"] > best[0]:
            best = (r["sharpe"], f, s)
    return best[1], best[2], best[0]


def run_simple_split(code: str, train_end: str, test_start: str,
                     vol_filter=False, vol_mult=1.3, macd_filter=False) -> dict:
    """简单切割验证：训练集优化参数，测试集跑结果"""
    df_all = pd.read_csv(f"data/{code}_indicators.csv",
                         index_col="date", parse_dates=True).dropna()
    ps    = df_all["close"]
    vol_s = df_all["volume"]
    hist_s= df_all["hist"]
    train_ps  = ps[ps.index <= train_end]
    test_ps   = ps[ps.index >= test_start]
    train_vol = vol_s[vol_s.index <= train_end]
    test_vol  = vol_s[vol_s.index >= test_start]
    train_h   = hist_s[hist_s.index <= train_end]
    test_h    = hist_s[hist_s.index >= test_start]
    if len(train_ps) < 60 or len(test_ps) < 20:
        return {}
    bf, bs, is_sh = _wfv_optimize(train_ps, train_vol, vol_filter, vol_mult, train_h, macd_filter)
    oos           = _wfv_run(test_ps, bf, bs, test_vol, vol_filter, vol_mult, test_h, macd_filter)
    decay         = oos["sharpe"] / is_sh if is_sh > 0 else 0
    return {
        "best_fast": bf, "best_slow": bs,
        "is_sharpe": round(is_sh, 3), "oos_sharpe": round(oos["sharpe"], 3),
        "oos_ret":   oos["total_ret"], "decay": round(decay, 3),
        "oos_nav":   oos["nav"],
        "train_len": len(train_ps), "test_len": len(test_ps),
    }


def run_walk_forward(code: str, start_year: int, end_year: int,
                     train_yrs: int = 3,
                     vol_filter=False, vol_mult=1.3, macd_filter=False) -> dict:
    """滚动WFV：每年重新优化参数，拼接样本外净值"""
    df_all = pd.read_csv(f"data/{code}_indicators.csv",
                         index_col="date", parse_dates=True).dropna()
    ps     = df_all["close"]
    vol_s  = df_all["volume"]
    hist_s = df_all["hist"]
    rows, oos_pieces, is_sh_list, oos_sh_list = [], [], [], []

    yr = start_year
    while yr + train_yrs <= end_year:
        t_end = f"{yr + train_yrs - 1}1231"
        v_st  = f"{yr + train_yrs}0101"
        v_end = f"{yr + train_yrs}1231"
        train_ps  = ps[(ps.index >= f"{yr}0101") & (ps.index <= t_end)]
        test_ps   = ps[(ps.index >= v_st) & (ps.index <= v_end)]
        train_vol = vol_s[(vol_s.index >= f"{yr}0101") & (vol_s.index <= t_end)]
        test_vol  = vol_s[(vol_s.index >= v_st) & (vol_s.index <= v_end)]
        train_h   = hist_s[(hist_s.index >= f"{yr}0101") & (hist_s.index <= t_end)]
        test_h    = hist_s[(hist_s.index >= v_st) & (hist_s.index <= v_end)]
        if len(train_ps) < 60 or len(test_ps) < 20:
            yr += 1; continue

        bf, bs, is_sh = _wfv_optimize(train_ps, train_vol, vol_filter, vol_mult, train_h, macd_filter)
        oos           = _wfv_run(test_ps, bf, bs, test_vol, vol_filter, vol_mult, test_h, macd_filter)
        oos_sh        = oos["sharpe"]

        nav_piece = oos["nav"].copy()
        if oos_pieces:
            scale = oos_pieces[-1].iloc[-1] / 1_000_000
            nav_piece = nav_piece * scale
        oos_pieces.append(nav_piece)

        is_sh_list.append(is_sh); oos_sh_list.append(oos_sh)
        rows.append({
            "训练区间": f"{yr}~{yr+train_yrs-1}",
            "测试年份": yr + train_yrs,
            "参数":    f"MA{bf}/MA{bs}",
            "样本内夏普": round(is_sh, 3),
            "样本外夏普": round(oos_sh, 3),
            "样本外收益": f"{oos['total_ret']:.1f}%",
            "判断": "✅" if oos_sh > 0.3 else ("⚠️" if oos_sh > 0 else "❌"),
        })
        yr += 1

    if not rows:
        return {}

    oos_combined = pd.concat(oos_pieces)
    oos_combined = oos_combined[~oos_combined.index.duplicated()]
    mean_is  = float(np.mean(is_sh_list))
    mean_oos = float(np.mean(oos_sh_list))
    decay    = mean_oos / mean_is if mean_is > 0 else 0
    win_rate = sum(s > 0 for s in oos_sh_list) / len(oos_sh_list)

    return {
        "table":        pd.DataFrame(rows),
        "oos_combined": oos_combined,
        "mean_is":      round(mean_is, 3),
        "mean_oos":     round(mean_oos, 3),
        "decay":        round(decay, 3),
        "win_rate":     win_rate,
        "oos_sharpes":  oos_sh_list,
        "is_sharpes":   is_sh_list,
        "test_years":   [r["测试年份"] for r in rows],
    }


def calc_ic_series(code: str, signal_col: str = "macd",
                   fwd_days: int = 20) -> dict:
    """计算单个因子的IC（信息系数）"""
    df  = pd.read_csv(f"data/{code}_indicators.csv",
                      index_col="date", parse_dates=True).dropna()
    if signal_col not in df.columns:
        return {}
    sig = df[signal_col]
    fwd = df["close"].pct_change(fwd_days).shift(-fwd_days)

    monthly_ic = []
    for period, grp in sig.groupby(sig.index.to_period("M")):
        idx  = grp.index.intersection(fwd.dropna().index)
        if len(idx) < 5:
            continue
        ic = grp[idx].corr(fwd[idx], method="spearman")
        if not np.isnan(ic):
            monthly_ic.append({"period": str(period), "ic": ic})

    if not monthly_ic:
        return {}
    ic_s    = pd.Series([x["ic"] for x in monthly_ic])
    ic_mean = ic_s.mean(); ic_std = ic_s.std()
    return {
        "ic_mean":    round(float(ic_mean), 4),
        "ic_ir":      round(float(ic_mean / ic_std) if ic_std > 0 else 0, 3),
        "ic_pos_pct": round(float((ic_s > 0).mean()), 3),
        "ic_series":  ic_s,
        "n_months":   len(ic_s),
    }


def plot_wfv_charts(code: str, wfv_result: dict, split_result: dict) -> bytes:
    """生成WFV可视化图（三合一），返回PNG bytes"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 图1：拼接样本外净值
    ax1 = axes[0]
    oos = wfv_result["oos_combined"] / 1_000_000
    ax1.plot(oos, color="#1D9E75", linewidth=1.5, label="样本外净值（WFV）")
    ax1.fill_between(oos.index, oos, 1, where=(oos < 1),
                     alpha=0.18, color="#E24B4A", label="回撤")
    ax1.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_title(f"{code}  WFV拼接净值", fontsize=11)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # 图2：各年样本内 vs 样本外夏普对比
    ax2 = axes[1]
    yrs  = wfv_result["test_years"]
    is_s = wfv_result["is_sharpes"]
    oo_s = wfv_result["oos_sharpes"]
    x = np.arange(len(yrs)); w = 0.35
    ax2.bar(x - w/2, is_s,  w, label="样本内夏普", color="#378ADD", alpha=0.75)
    ax2.bar(x + w/2, oo_s, w, label="样本外夏普", color="#1D9E75", alpha=0.75)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels([str(y) for y in yrs])
    ax2.set_title("各年：样本内 vs 样本外夏普", fontsize=11)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")

    # 图3：简单切割净值对比
    ax3 = axes[2]
    if split_result.get("oos_nav") is not None and len(split_result["oos_nav"]) > 0:
        oos_norm = split_result["oos_nav"] / 1_000_000
        ax3.plot(oos_norm, color="#EF9F27", linewidth=1.5, label="测试集净值")
        ax3.fill_between(oos_norm.index, oos_norm, 1,
                         where=(oos_norm < 1), alpha=0.15, color="#E24B4A")
        ax3.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_title(f"简单切割  测试集净值", fontsize=11)
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    buf.seek(0)
    plt.close(fig)
    return buf.read()


# =============================================================================
# 基本面选股函数
# =============================================================================

def get_fundamental_data():
    df = ak.stock_zh_a_spot_em()
    df = df[["代码","名称","最新价","市盈率-动态","市净率",
             "总市值","换手率","60日涨跌幅"]].copy()
    df.columns = ["code","name","price","pe","pb","market_cap","turnover","ret_60d"]
    for col in ["pe","pb","market_cap","turnover","ret_60d"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def screen_stocks(df, pe_max=35, pb_min=1, pb_max=8,
                  cap_min=200, cap_max=3000, ret_60d_min=0):
    exclude = ["银行","保险","证券","地产","建设","铁建","中铁"]
    mask    = df["name"].apply(lambda x: any(k in str(x) for k in exclude))
    result  = df[
        (~mask) &
        (df["pe"] > 0) & (df["pe"] < pe_max) &
        (df["pb"] > pb_min) & (df["pb"] < pb_max) &
        (df["market_cap"] > cap_min * 1e8) &
        (df["market_cap"] < cap_max * 1e8) &
        (df["ret_60d"] > ret_60d_min)
    ].copy()
    result = result.sort_values("pe")
    result["市值(亿)"] = (result["market_cap"] / 1e8).round(1)
    return result[["code","name","price","pe","pb","市值(亿)","ret_60d"]]


def technical_screen(code, window=20):
    """
    技术面筛选：计算价量信号
    返回包含各信号的字典
    """
    try:
        prefix = "sh" if code.startswith("6") else "sz"
        df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
        df["date"] = pd.to_datetime(df["date"])
        df = df.tail(60).copy()  # 只取最近60天
        df.set_index("date", inplace=True)
        if len(df) < window + 5:
            return None

        close  = df["close"]
        volume = df["volume"]

        # 均线多头排列
        ma5  = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma_bullish = (ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1])

        # 放量突破：成交量 > N日均量的1.5倍且价格创N日新高
        vol_avg      = volume.rolling(window).mean()
        vol_breakout = (volume.iloc[-1] > vol_avg.iloc[-1] * 1.5) and \
                       (close.iloc[-1] == close.rolling(window).max().iloc[-1])

        # 价量背离（顶背离）：价格创新高但量能萎缩
        price_new_high  = close.iloc[-1] == close.rolling(window).max().iloc[-1]
        vol_below_avg   = volume.iloc[-1] < vol_avg.iloc[-1] * 0.7
        top_divergence  = price_new_high and vol_below_avg

        # 缩量整理后放量（金针探底型）：
        # 前5日成交量低迷，最新一日放量
        recent_vol_low  = volume.iloc[-6:-1].mean()
        vol_surge       = volume.iloc[-1] > recent_vol_low * 2.0

        # MACD 金叉
        ema12      = close.ewm(span=12).mean()
        ema26      = close.ewm(span=26).mean()
        macd       = ema12 - ema26
        signal     = macd.ewm(span=9).mean()
        macd_cross = (macd.iloc[-2] < signal.iloc[-2]) and \
                     (macd.iloc[-1] > signal.iloc[-1])

        # RSI 超卖反弹
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = (100 - (100 / (1 + gain / loss))).iloc[-1]
        rsi_oversold_bounce = (rsi > 30) and (rsi < 50) and \
                              (close.iloc[-1] > close.iloc[-3])

        # 综合打分（满分5分）
        score = sum([
            ma_bullish,
            vol_breakout,
            not top_divergence,   # 无顶背离加分
            macd_cross or vol_surge,
            rsi_oversold_bounce
        ])

        return {
            "code":                code,
            "均线多头":            "✅" if ma_bullish        else "❌",
            "放量突破":            "✅" if vol_breakout       else "❌",
            "顶背离预警":          "⚠️" if top_divergence    else "✅",
            "MACD金叉/缩量放量":   "✅" if (macd_cross or vol_surge) else "❌",
            "RSI超卖反弹":         "✅" if rsi_oversold_bounce else "❌",
            "技术评分":            f"{score}/5",
            "RSI当前值":           round(rsi, 1),
            "_score":              score,
        }
    except:
        return None


# =============================================================================
# Streamlit 界面
# =============================================================================

st.set_page_config(page_title="量化投资分析平台", page_icon="📈", layout="wide")
st.title("📈 量化投资分析平台 v6.0")
st.caption("策略回测 · Walk-Forward验证 · 月度因子回测 · 基本面选股 · PDF报告导出")

# 顶部页面导航
page = st.radio("", ["📊 策略回测", "🔬 策略验证(WFV)", "📅 因子回测", "🔍 基本面选股"], horizontal=True, label_visibility="collapsed")
st.divider()

# =========================================================================
# 页面一：策略回测
# =========================================================================
if page == "📊 策略回测":

    with st.sidebar:
        st.header("⚙️ 基础设置")
        mode = st.radio("分析模式", ["📊 单股分析", "📈 多股对比"], horizontal=True)
        if mode == "📊 单股分析":
            code       = st.text_input("股票代码", value="600519", help="沪市6开头，深市0/3开头")
            codes_list = [code.strip()] if code.strip() else []
        else:
            codes_input = st.text_area("股票代码（每行一个，最多6只）",
                                        value="600519\n000858\n601899")
            codes_list  = [c.strip() for c in codes_input.strip().split("\n") if c.strip()][:6]
            code        = codes_list[0] if codes_list else ""
        start_date = st.text_input("开始日期", value="20200101")
        end_date   = st.text_input("结束日期", value="20241231")

        st.divider()
        st.header("📐 策略选择")
        strategy = st.selectbox("选择策略", [
            "🔀 双均线（MA金叉死叉）",
            "📉 RSI超买超卖",
            "📊 布林带突破",
            "⚡ MACD金叉死叉",
            "🏆 四策略对比"
        ])

        st.divider()
        st.header("🛡️ 风险控制")
        stop_loss   = st.slider("止损比例", 5, 20, 10, 1, help="（%）") / 100
        take_profit = st.slider("止盈比例", 10, 50, 30, 1, help="（%）") / 100

        if "双均线" in strategy:
            st.divider()
            st.header("📏 均线参数")
            use_auto = st.checkbox("自动优化参数", value=True)
            if not use_auto:
                fast_ma = st.slider("快线周期", 5, 20, 5, 1)
                slow_ma = st.slider("慢线周期", 20, 60, 40, 5)
            else:
                fast_ma, slow_ma = 5, 40

            st.divider()
            st.header("🔍 信号过滤")
            vol_filter  = st.checkbox("成交量过滤", value=False,
                                      help="金叉时成交量须大于20日均量N倍，排除缩量假突破")
            vol_mult    = st.slider("成交量倍数", 1.0, 3.0, 1.3, 0.1,
                                    disabled=not vol_filter)
            macd_filter = st.checkbox("MACD方向过滤", value=False,
                                      help="买入时要求MACD柱状图>0，确认动能向上")
        else:
            use_auto, fast_ma, slow_ma = True, 5, 40
            vol_filter, vol_mult, macd_filter = False, 1.3, False

        if "RSI" in strategy:
            st.divider()
            st.header("📏 RSI参数")
            rsi_buy  = st.slider("超卖线（买入）", 20, 40, 30, 1)
            rsi_sell = st.slider("超买线（卖出）", 60, 80, 70, 1)
        else:
            rsi_buy, rsi_sell = 30, 70

        st.divider()
        st.header("🤖 AI 分析")
        api_key = st.text_input("Anthropic API Key", type="password")
        run_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

    if run_btn:
        if not codes_list:
            st.error("请输入股票代码")
            st.stop()

        # 多股对比模式
        if mode == "📈 多股对比":
            all_results, all_navs = [], {}
            for c in codes_list:
                with st.status(f"⏳ 处理 {c}...", expanded=False) as status:
                    try:
                        download_stock(c, start_date, end_date)
                        calc_indicators(c)
                        if "双均线" in strategy:
                            if use_auto:
                                opt = optimize_ma_params(c)
                                bf  = int(opt.iloc[0]["fast"])
                                bs  = int(opt.iloc[0]["slow"])
                            else:
                                bf, bs = fast_ma, slow_ma
                            nav_s, _, stats = backtest_ma(c, bf, bs, stop_loss=stop_loss, take_profit=take_profit, vol_filter=vol_filter, vol_mult=vol_mult, macd_filter=macd_filter)
                            sname = f"MA{bf}/MA{bs}"
                        elif "RSI" in strategy:
                            nav_s, _, stats = backtest_rsi(c, rsi_buy, rsi_sell, stop_loss=stop_loss, take_profit=take_profit)
                            sname = f"RSI({rsi_buy}/{rsi_sell})"
                        elif "布林带" in strategy:
                            nav_s, _, stats = backtest_boll(c, stop_loss=stop_loss, take_profit=take_profit)
                            sname = "Bollinger"
                        else:
                            nav_s, _, stats = backtest_macd(c, stop_loss=stop_loss, take_profit=take_profit)
                            sname = "MACD"
                        all_navs[c] = nav_s
                        all_results.append({"股票代码": c, "策略": sname, **stats})
                        status.update(label=f"✅ {c} 完成", state="complete")
                    except Exception as e:
                        status.update(label=f"❌ {c} 失败", state="error")

            if not all_results:
                st.error("所有股票处理失败")
                st.stop()

            st.divider()
            st.subheader(f"📈 多股对比（{sname}策略）")
            compare_df = pd.DataFrame(all_results)[
                ["股票代码","策略","total_ret","sharpe","max_dd","n_trades","final_nav"]]
            compare_df.columns = ["股票代码","策略","总收益率(%)","夏普比率","最大回撤(%)","交易次数","最终净值"]
            best_idx = compare_df["夏普比率"].idxmax()
            st.dataframe(compare_df, hide_index=True, width="stretch")
            st.success(f"🥇 综合最优：**{compare_df.iloc[best_idx]['股票代码']}**（夏普比率最高）")

            fig, ax = plt.subplots(figsize=(14, 5))
            for c, nav_s in all_navs.items():
                ax.plot(nav_s / nav_s.iloc[0], label=c, linewidth=1.5)
            ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title(f"Multi-Stock NAV Comparison ({sname})", fontsize=13)
            ax.legend(); ax.grid(alpha=0.3)
            plt.tight_layout()
            st.image(_fig_to_buf(fig), width="stretch")

            fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
            codes_show = compare_df["股票代码"].tolist()
            axes2[0].bar(codes_show, compare_df["总收益率(%)"],
                         color=["#4CAF50" if v>0 else "#F44336" for v in compare_df["总收益率(%)"]])
            axes2[0].set_title("Total Return (%)"); axes2[0].grid(alpha=0.3, axis="y")
            axes2[1].bar(codes_show, compare_df["夏普比率"], color="#2196F3")
            axes2[1].set_title("Sharpe Ratio"); axes2[1].grid(alpha=0.3, axis="y")
            axes2[2].bar(codes_show, compare_df["最大回撤(%)"], color="#FF9800")
            axes2[2].set_title("Max Drawdown (%)"); axes2[2].grid(alpha=0.3, axis="y")
            plt.tight_layout()
            st.image(_fig_to_buf(fig2), width="stretch")

            # 多股多策略矩阵
            st.subheader("🔢 多股 × 多策略矩阵对比")
            matrix_results = []
            strategy_funcs = {
                "MA":        lambda c: backtest_ma(c, stop_loss=stop_loss, take_profit=take_profit, vol_filter=vol_filter, vol_mult=vol_mult, macd_filter=macd_filter),
                "RSI":       lambda c: backtest_rsi(c, stop_loss=stop_loss, take_profit=take_profit),
                "Bollinger": lambda c: backtest_boll(c, stop_loss=stop_loss, take_profit=take_profit),
                "MACD":      lambda c: backtest_macd(c, stop_loss=stop_loss, take_profit=take_profit),
            }
            with st.spinner("运行矩阵对比中..."):
                for c in codes_list:
                    row = {"股票": c}
                    for skey, sfunc in strategy_funcs.items():
                        try:
                            _, _, st_ = sfunc(c)
                            row[f"{skey}_收益"] = st_["total_ret"]
                            row[f"{skey}_夏普"] = st_["sharpe"]
                        except:
                            row[f"{skey}_收益"] = None
                            row[f"{skey}_夏普"] = None
                    matrix_results.append(row)

            matrix_df  = pd.DataFrame(matrix_results).set_index("股票")
            sharpe_cols = [c for c in matrix_df.columns if "夏普" in c]
            sharpe_mat  = matrix_df[sharpe_cols].copy()
            sharpe_mat.columns = ["MA", "RSI", "Bollinger", "MACD"]

            fig3, ax3 = plt.subplots(figsize=(8, max(3, len(codes_list))))
            im = ax3.imshow(sharpe_mat.values.astype(float), cmap="RdYlGn", aspect="auto")
            ax3.set_xticks(range(4)); ax3.set_xticklabels(["MA","RSI","Bollinger","MACD"])
            ax3.set_yticks(range(len(codes_list))); ax3.set_yticklabels(codes_list)
            for i in range(len(codes_list)):
                for j in range(4):
                    val = sharpe_mat.values[i, j]
                    if val is not None and not np.isnan(float(val)):
                        ax3.text(j, i, f"{float(val):.2f}", ha="center", va="center",
                                 fontsize=11, fontweight="bold",
                                 color="white" if abs(float(val)) > 0.6 else "black")
            plt.colorbar(im, ax=ax3, label="Sharpe Ratio")
            ax3.set_title("Sharpe Ratio Heatmap: Stock x Strategy", fontsize=13)
            plt.tight_layout()
            st.image(_fig_to_buf(fig3), width="stretch")

            best_val, best_stock, best_strat = -999, "", ""
            for c in codes_list:
                row = matrix_df.loc[c]
                for skey in ["MA","RSI","Bollinger","MACD"]:
                    val = row.get(f"{skey}_夏普")
                    if val is not None and not np.isnan(float(val)) and float(val) > best_val:
                        best_val   = float(val)
                        best_stock = c
                        best_strat = skey
            st.success(f"🥇 全局最优组合：**{best_stock}** × **{best_strat}**（夏普比率 {best_val:.3f}）")

            if api_key:
                with st.spinner("Claude 正在分析中..."):
                    try:
                        best_stats = all_results[best_idx]
                        best_name  = compare_df.iloc[best_idx]["股票代码"]
                        summary    = compare_df.to_string(index=False)
                        analysis   = ai_analysis(best_name, f"多股对比（{sname}策略）",
                                                 best_stats, api_key, f"对比汇总:\n{summary}")
                        st.subheader("🤖 AI 智能分析")
                        st.markdown(analysis)
                    except Exception as e:
                        st.error(f"AI分析失败: {e}")
            st.stop()

        # 单股分析模式
        with st.status("📥 下载数据中...", expanded=True) as status:
            try:
                df_raw = download_stock(code, start_date, end_date)
                st.write(f"✅ 下载完成，共 {len(df_raw)} 条数据")
                status.update(label="数据下载完成", state="complete")
            except Exception as e:
                status.update(label="下载失败", state="error")
                st.error(f"下载失败: {e}")
                st.stop()

        with st.status("📊 计算指标...", expanded=True) as status:
            calc_indicators(code)
            st.write("✅ MA、MACD、RSI、布林带计算完成")
            status.update(label="指标计算完成", state="complete")

        with st.status("📡 获取沪深300基准...", expanded=True) as status:
            benchmark = get_benchmark(start_date, end_date)
            if benchmark is not None:
                st.write("✅ 沪深300数据获取成功")
                status.update(label="基准获取完成", state="complete")
            else:
                st.write("⚠️ 基准获取失败，跳过对比")
                status.update(label="基准获取失败（跳过）", state="error")

        with st.status("⚡ 执行回测...", expanded=True) as status:
            if "双均线" in strategy:
                if use_auto:
                    opt_df    = optimize_ma_params(code)
                    best_fast = int(opt_df.iloc[0]["fast"])
                    best_slow = int(opt_df.iloc[0]["slow"])
                    st.write(f"✅ 最优参数: MA{best_fast}/MA{best_slow}")
                else:
                    best_fast, best_slow = fast_ma, slow_ma
                nav_s, trades_df, stats = backtest_ma(code, best_fast, best_slow, stop_loss=stop_loss, take_profit=take_profit, vol_filter=vol_filter, vol_mult=vol_mult, macd_filter=macd_filter)
                fig_main      = plot_ma(code, nav_s, best_fast, best_slow)
                strategy_name = f"MA Crossover MA{best_fast}/MA{best_slow}"
                extra_info    = f"Best MA params: MA{best_fast}/MA{best_slow}"

            elif "RSI" in strategy:
                nav_s, trades_df, stats = backtest_rsi(code, rsi_buy, rsi_sell, stop_loss=stop_loss, take_profit=take_profit)
                fig_main      = plot_rsi(code, nav_s, rsi_buy, rsi_sell)
                strategy_name = f"RSI Strategy (Buy<{rsi_buy}/Sell>{rsi_sell})"
                extra_info    = f"RSI buy: {rsi_buy}, sell: {rsi_sell}"

            elif "布林带" in strategy:
                nav_s, trades_df, stats = backtest_boll(code, stop_loss=stop_loss, take_profit=take_profit)
                fig_main      = plot_boll(code, nav_s)
                strategy_name = "Bollinger Band Strategy"
                extra_info    = "Bollinger: 20-day MA, 2 std"

            elif "MACD" in strategy:
                nav_s, trades_df, stats = backtest_macd(code, stop_loss=stop_loss, take_profit=take_profit)
                fig_main      = plot_macd(code, nav_s)
                strategy_name = "MACD Crossover Strategy"
                extra_info    = "MACD: EMA12/EMA26/Signal9"

            elif "四策略" in strategy:
                opt_df    = optimize_ma_params(code)
                best_fast = int(opt_df.iloc[0]["fast"])
                best_slow = int(opt_df.iloc[0]["slow"])
                nav_ma,   _, stats_ma   = backtest_ma(code, best_fast, best_slow, stop_loss=stop_loss, take_profit=take_profit, vol_filter=vol_filter, vol_mult=vol_mult, macd_filter=macd_filter)
                nav_rsi,  _, stats_rsi  = backtest_rsi(code, rsi_buy, rsi_sell, stop_loss=stop_loss, take_profit=take_profit)
                nav_boll, _, stats_boll = backtest_boll(code, stop_loss=stop_loss, take_profit=take_profit)
                nav_macd, _, stats_macd = backtest_macd(code, stop_loss=stop_loss, take_profit=take_profit)
                all_s = [stats_ma, stats_rsi, stats_boll, stats_macd]
                names = [f"MA{best_fast}/MA{best_slow}", "RSI", "Bollinger", "MACD"]
                best_idx   = max(range(4), key=lambda i: all_s[i]["sharpe"])
                stats      = all_s[best_idx]
                nav_s      = [nav_ma, nav_rsi, nav_boll, nav_macd][best_idx]
                strategy_name = names[best_idx]
                extra_info    = ""
                fig_main = plt.figure(figsize=(14, 5))
                ax = fig_main.add_subplot(111)
                for n, lb in zip([nav_ma, nav_rsi, nav_boll, nav_macd], names):
                    ax.plot(n / n.iloc[0], label=lb, linewidth=1.5)
                ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
                ax.set_title(f"{code} Four-Strategy NAV Comparison", fontsize=13)
                ax.legend(); ax.grid(alpha=0.3)
                fig_main.tight_layout()

            st.write(f"✅ 回测完成，共 {stats['n_trades']} 笔交易")
            status.update(label="回测完成", state="complete")

        # 基准对比 + 年度收益
        fig_bench, annual = plot_benchmark_and_annual(code, nav_s, benchmark, strategy_name)
        stock_name = get_stock_name(code)

        # 展示结果
        st.divider()
        st.subheader(f"📊 {code} {stock_name} · {strategy_name}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总收益率", f"{stats['total_ret']}%",
                    delta=f"{'↑' if stats['total_ret']>0 else '↓'}")
        col2.metric("夏普比率", f"{stats['sharpe']}")
        col3.metric("最大回撤", f"{stats['max_dd']}%")
        col4.metric("最终净值", f"¥{stats['final_nav']:,.0f}")

        col5, col6, col7 = st.columns(3)
        col5.metric("止损触发", f"{stats['n_stop']} 次")
        col6.metric("止盈触发", f"{stats['n_profit']} 次")
        col7.metric("信号卖出", f"{stats['n_signal']} 次")

        st.subheader("📈 策略图表")
        img_main_buf = _fig_to_buf(fig_main)
        img_main_buf.seek(0)
        st.image(img_main_buf, width="stretch")
        st.session_state["img_main"] = img_main_buf.getvalue()

        st.subheader("📊 基准对比 & 年度收益")
        img_bench_buf = _fig_to_buf(fig_bench)
        img_bench_buf.seek(0)
        st.image(img_bench_buf, width="stretch")
        st.session_state["img_bench"] = img_bench_buf.getvalue()
        st.session_state["pdf_meta"]  = {
            "code": code, "stock_name": stock_name,
            "strategy_name": strategy_name,
            "stats": stats, "annual": annual.to_dict()
        }

        if len(trades_df) > 0:
            st.subheader("📋 交易记录（最近20笔）")
            st.dataframe(trades_df.tail(20), hide_index=True, width="stretch")

        # AI 分析
        st.subheader("🤖 AI 智能分析")
        annual_str = "  ".join([f"{y}: {v:.1f}%" for y, v in annual.items()])
        full_extra = f"{extra_info}\n年度收益: {annual_str}"
        if api_key:
            with st.spinner("Claude 正在分析中..."):
                try:
                    analysis = ai_analysis(code, strategy_name, stats, api_key, full_extra)
                    st.markdown(analysis)
                    ai_text = analysis
                except Exception as e:
                    st.error(f"AI分析失败: {e}")
                    ai_text = ""
        else:
            st.info("填入 Anthropic API Key 即可获得 AI 智能分析报告")
            ai_text = ""

        # PDF 导出
        st.divider()
        st.subheader("📄 导出 PDF 报告")
        st.session_state["show_pdf"] = True

    else:
        st.info("👈 在左侧设置参数，点击「开始分析」")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**🔀 双均线**\n- 金叉买入/死叉卖出\n- 自动参数优化")
        with col2:
            st.markdown("**📉 RSI策略**\n- 超卖买入/超买卖出\n- 可调买卖阈值")
        with col3:
            st.markdown("**📊 布林带**\n- 突破上轨买入\n- 跌破中轨卖出")
        with col4:
            st.markdown("**🏆 四策略对比**\n- 同时运行四个策略\n- AI推荐最优策略")

# PDF 导出区域（始终可见，分析完成后激活）
if st.session_state.get("show_pdf") or st.session_state.get("pdf_meta"):
    st.divider()
    st.subheader("📄 导出 PDF 报告")
    if "pdf_meta" in st.session_state:
        if st.button("生成 PDF 报告", type="secondary"):
            with st.spinner("正在生成 PDF..."):
                try:
                    meta    = st.session_state["pdf_meta"]
                    pdf_buf = generate_pdf(
                        meta["code"], meta["stock_name"],
                        meta["strategy_name"], meta["stats"],
                        pd.Series(meta["annual"]),
                        io.BytesIO(st.session_state["img_main"]),
                        io.BytesIO(st.session_state["img_bench"])
                    )
                    st.download_button(
                        label="⬇️ 下载 PDF 报告",
                        data=pdf_buf,
                        file_name=f"{meta['code']}_{meta['strategy_name']}_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"PDF生成失败: {e}")

# =========================================================================
# 页面：月度因子回测
# =========================================================================
elif page == "📅 因子回测":
    st.subheader("📅 月度调仓因子回测")
    st.caption("基于动量因子 + 趋势过滤，每月末自动换仓，等权持有前N只股票")

    with st.sidebar:
        st.header("⚙️ 股票池设置")
        st.caption("先用基本面条件圈定股票池，再用动量因子每月选股")
        fac_pe_max   = st.slider("PE上限",   10, 60,  35, 1)
        fac_pb_max   = st.slider("PB上限",    2, 15,   8, 1)
        fac_cap_min  = st.slider("市值下限(亿)", 50, 500, 100, 50)
        fac_cap_max  = st.slider("市值上限(亿)", 500, 10000, 3000, 500)
        fac_max_pool = st.slider("最多下载只数", 20, 80, 40, 10,
                                  help="只数越多越准确，但下载越慢")

        st.divider()
        st.header("📅 回测参数")
        fac_start    = st.text_input("开始日期", value="20190101")
        fac_end      = st.text_input("结束日期", value="20241231")
        fac_n        = st.slider("每月持仓只数", 5, 40, 20, 5)

        st.divider()
        st.header("🔬 WFV 验证")
        fac_wfv      = st.checkbox("同时做样本外验证", value=True)
        fac_train_end = st.text_input("训练集截止", value="20221231")
        fac_test_start = st.text_input("测试集起始", value="20230101")

        run_fac = st.button("🚀 开始回测", type="primary", use_container_width=True)

    if run_fac:

        # ── Step1：获取股票池 ──
        with st.status("📥 获取全市场基本面数据...", expanded=True) as sts:
            try:
                df_all   = get_fundamental_data()
                df_screen = screen_stocks(df_all,
                                          pe_max=fac_pe_max, pb_min=0.5,
                                          pb_max=fac_pb_max,
                                          cap_min=fac_cap_min,
                                          cap_max=fac_cap_max,
                                          ret_60d_min=-50)
                pool_codes = df_screen["code"].tolist()[:fac_max_pool]
                st.write(f"✅ 筛出 {len(pool_codes)} 只股票（取前{fac_max_pool}只）")
                sts.update(label=f"股票池：{len(pool_codes)} 只", state="complete")
            except Exception as e:
                sts.update(label=f"获取失败：{e}", state="error")
                st.stop()

        # ── Step2：批量下载历史价格 ──
        progress_bar = st.progress(0, text="下载历史价格中...")
        price_data = {}
        for idx, code in enumerate(pool_codes):
            try:
                prefix = "sh" if code.startswith("6") else "sz"
                df_tmp = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
                df_tmp["date"] = pd.to_datetime(df_tmp["date"])
                df_tmp = df_tmp[(df_tmp["date"] >= pd.to_datetime(fac_start)) &
                                (df_tmp["date"] <= pd.to_datetime(fac_end))]
                df_tmp.set_index("date", inplace=True)
                if len(df_tmp) > 120:
                    price_data[code] = df_tmp["close"]
            except:
                pass
            time.sleep(0.3)
            progress_bar.progress((idx + 1) / len(pool_codes),
                                   text=f"下载中 {idx+1}/{len(pool_codes)}：{code}")

        progress_bar.empty()
        st.success(f"✅ 成功下载 {len(price_data)} 只股票数据")

        if len(price_data) < 5:
            st.error("有效股票数量过少，请调整筛选条件或扩大股票池")
            st.stop()

        # ── Step3：全样本回测 ──
        with st.status("📊 运行月度因子回测...", expanded=True) as sts:
            nav_full, rebal_log, rebal_df = backtest_factor_monthly(
                price_data, fac_start, fac_end, fac_n
            )
            if nav_full.empty:
                sts.update(label="回测失败，数据不足", state="error")
                st.stop()
            sts.update(label="回测完成", state="complete")

        # ── 绩效指标 ──
        total_ret = (nav_full.iloc[-1] / 1_000_000 - 1) * 100
        dr        = nav_full.pct_change().dropna()
        sharpe    = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        max_dd    = ((nav_full - nav_full.cummax()) / nav_full.cummax()).min() * 100
        n_rebal   = len(rebal_log)

        st.subheader("📊 全样本回测结果")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总收益率",   f"{total_ret:.1f}%")
        c2.metric("夏普比率",   f"{sharpe:.3f}")
        c3.metric("最大回撤",   f"{max_dd:.1f}%")
        c4.metric("换仓次数",   f"{n_rebal} 次")

        # ── 图表 ──
        benchmark = get_benchmark(fac_start, fac_end)
        img_bytes = plot_factor_result(nav_full, benchmark, rebal_df, fac_n)
        st.image(img_bytes, use_container_width=True)

        # ── 换仓记录 ──
        with st.expander("📋 换仓记录（最近10次）"):
            if not rebal_df.empty:
                st.dataframe(rebal_df.tail(10), hide_index=True,
                             use_container_width=True)

        # ── WFV 验证 ──
        if fac_wfv:
            st.divider()
            st.subheader("🔬 样本外验证（简单切割）")

            train_data = {c: ps[ps.index <= fac_train_end]
                          for c, ps in price_data.items()}
            test_data  = {c: ps[ps.index >= fac_test_start]
                          for c, ps in price_data.items()}

            with st.spinner("运行训练集回测..."):
                nav_train, _, _ = backtest_factor_monthly(
                    train_data, fac_start, fac_train_end, fac_n)
            with st.spinner("运行测试集回测..."):
                nav_test, _, _ = backtest_factor_monthly(
                    test_data, fac_test_start, fac_end, fac_n)

            if not nav_train.empty and not nav_test.empty:
                dr_train  = nav_train.pct_change().dropna()
                dr_test   = nav_test.pct_change().dropna()
                sh_train  = (dr_train.mean() / dr_train.std() * np.sqrt(252)
                             if dr_train.std() > 0 else 0)
                sh_test   = (dr_test.mean()  / dr_test.std()  * np.sqrt(252)
                             if dr_test.std()  > 0 else 0)
                ret_train = (nav_train.iloc[-1] / 1_000_000 - 1) * 100
                ret_test  = (nav_test.iloc[-1]  / 1_000_000 - 1) * 100
                decay     = sh_test / sh_train if sh_train > 0 else 0

                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("训练集夏普", f"{sh_train:.3f}")
                cc2.metric("测试集夏普", f"{sh_test:.3f}",
                            delta=f"{sh_test-sh_train:+.3f}")
                cc3.metric("训练集收益", f"{ret_train:.1f}%")
                cc4.metric("测试集收益", f"{ret_test:.1f}%",
                            delta=f"{ret_test-ret_train:+.1f}%")

                if decay > 0.7:
                    st.success(f"夏普衰减比 {decay:.2f}  ✅ 健壮——因子在样本外依然有效")
                elif decay > 0.4:
                    st.warning(f"夏普衰减比 {decay:.2f}  ⚠️ 一般——有一定样本外有效性")
                elif sh_test > 0:
                    st.warning(f"夏普衰减比 {decay:.2f}  ⚠️ 衰减较大，但样本外仍为正")
                else:
                    st.error(f"夏普衰减比 {decay:.2f}  ❌ 样本外为负——需进一步改进")

                st.caption("💡 与双均线相比，月度因子策略的样本外衰减通常更小，"
                           "因为换手率低、信号来自价格动量（有学术依据）")

    else:
        st.info("👈 在左侧设置参数，点击「开始回测」")
        st.markdown("""
**策略逻辑说明：**

每个月末，对股票池里所有股票计算一个综合得分：

```
动量得分 = 12个月前价格 → 1个月前价格的涨幅（跳过最近1月）
趋势过滤 = 当前价格 > 60日均线（否则得分清零）
最终得分 = 动量百分位排名 × 趋势过滤
```

选出得分前N只，等权持有到下月末，循环。

**与双均线策略的核心区别：**
- 换手率：每年约12次 vs 30-40次，手续费大幅降低
- 分散性：20只等权 vs 单股，风险更分散
- 信号来源：12个月动量在A股有学术支持，IC约0.04~0.08
""")

# =========================================================================
# 页面二：策略验证（Walk-Forward）
# =========================================================================
elif page == "🔬 策略验证(WFV)":
    st.subheader("🔬 策略验证：Walk-Forward + IC分析")
    st.caption("用样本外数据客观评估策略是否有真实优势，而不是过拟合历史数据")

    with st.sidebar:
        st.header("⚙️ 验证设置")
        wfv_code  = st.text_input("股票代码", value="600519", key="wfv_code")
        wfv_start = st.text_input("数据起始日期", value="20180101", key="wfv_start")
        wfv_end   = st.text_input("数据截止日期", value="20241231", key="wfv_end")

        st.divider()
        st.header("✂️ 简单切割")
        train_end  = st.text_input("训练集截止", value="20221231")
        test_start = st.text_input("测试集起始", value="20230101")

        st.divider()
        st.header("🔄 Walk-Forward")
        wfv_train_yrs = st.slider("训练窗口（年）", 2, 4, 3, 1)
        start_yr      = st.number_input("起始年份", value=2018, step=1)
        end_yr        = st.number_input("终止年份", value=2024, step=1)

        st.divider()
        st.header("📐 IC分析因子")
        ic_col = st.selectbox("选择信号列", ["macd", "rsi", "hist"])
        ic_fwd = st.slider("预测未来N日收益", 5, 60, 20, 5)

        st.divider()
        st.header("🔍 信号过滤（验证用）")
        wfv_vol_filter  = st.checkbox("成交量过滤", value=False, key="wfv_vf")
        wfv_vol_mult    = st.slider("成交量倍数", 1.0, 3.0, 1.3, 0.1,
                                    disabled=not wfv_vol_filter, key="wfv_vm")
        wfv_macd_filter = st.checkbox("MACD方向过滤", value=False, key="wfv_mf")

        run_wfv = st.button("🚀 开始验证", type="primary", use_container_width=True)

    if run_wfv:
        code = wfv_code.strip()

        # ── 下载 & 计算指标 ──
        with st.status("📥 下载数据并计算指标...", expanded=True) as sts:
            try:
                download_stock(code, wfv_start, wfv_end)
                calc_indicators(code)
                stock_name = get_stock_name(code)
                st.write(f"✅ {code} {stock_name} 数据就绪")
                sts.update(label="数据准备完成", state="complete")
            except Exception as e:
                sts.update(label=f"失败：{e}", state="error")
                st.stop()

        # ── 分三列展示结果 ──
        col_s, col_w, col_i = st.columns([1, 1.4, 1])

        # ── 简单切割 ──
        with col_s:
            st.markdown("#### ✂️ 简单切割")
            with st.spinner("计算中..."):
                spl = run_simple_split(code, train_end, test_start,
                                       wfv_vol_filter, wfv_vol_mult, wfv_macd_filter)
            if not spl:
                st.warning("数据不足")
            else:
                decay = spl["decay"]
                if decay > 0.7:
                    badge, color = "✅ 健壮", "normal"
                elif decay > 0.5:
                    badge, color = "⚠️ 一般", "normal"
                else:
                    badge, color = "❌ 过拟合", "normal"

                st.metric("最优参数", f"MA{spl['best_fast']}/MA{spl['best_slow']}")
                st.metric("训练集夏普", spl["is_sharpe"])
                st.metric("测试集夏普", spl["oos_sharpe"],
                           delta=f"{spl['oos_sharpe']-spl['is_sharpe']:+.3f}")
                st.metric("测试集收益", f"{spl['oos_ret']:.1f}%")

                if decay > 0.7:
                    st.success(f"夏普衰减比 {decay:.2f}  ✅ 健壮")
                elif decay > 0.5:
                    st.warning(f"夏普衰减比 {decay:.2f}  ⚠️ 一般")
                else:
                    st.error(f"夏普衰减比 {decay:.2f}  ❌ 过拟合")

                st.caption(f"训练：{spl['train_len']}日 | 测试：{spl['test_len']}日")

        # ── Walk-Forward ──
        with col_w:
            st.markdown("#### 🔄 Walk-Forward")
            with st.spinner("滚动验证中（较慢，请稍候）..."):
                wfv = run_walk_forward(code, int(start_yr), int(end_yr), wfv_train_yrs,
                                       wfv_vol_filter, wfv_vol_mult, wfv_macd_filter)

            if not wfv:
                st.warning("数据不足，无法进行WFV")
            else:
                # 明细表
                st.dataframe(wfv["table"], hide_index=True, use_container_width=True)

                # 汇总指标
                decay_w = wfv["decay"]
                c1, c2 = st.columns(2)
                c1.metric("平均样本内夏普", wfv["mean_is"])
                c2.metric("平均样本外夏普", wfv["mean_oos"],
                           delta=f"{wfv['mean_oos']-wfv['mean_is']:+.3f}")
                c1.metric("夏普衰减比", f"{decay_w:.2f}")
                c2.metric("样本外正收益率", f"{wfv['win_rate']:.0%}")

                if decay_w > 0.7:
                    st.success("✅ WFV结论：策略在多个时间窗口均稳健")
                elif decay_w > 0.5:
                    st.warning("⚠️ WFV结论：有一定优势，建议进一步改进信号")
                else:
                    st.error("❌ WFV结论：过拟合较严重，建议重新设计策略")

        # ── IC分析 ──
        with col_i:
            st.markdown(f"#### 📐 IC分析（{ic_col}）")
            with st.spinner("计算IC..."):
                ic = calc_ic_series(code, ic_col, ic_fwd)

            if not ic:
                st.warning("无法计算IC，检查因子列名")
            else:
                ic_mean = ic["ic_mean"]; ic_ir = ic["ic_ir"]
                ic_pos  = ic["ic_pos_pct"]

                st.metric("IC均值", f"{ic_mean:.4f}",
                           help="|IC|>0.05 有效，接近0无效")
                st.metric("ICIR", f"{ic_ir:.3f}",
                           help=">0.5 稳定，<0.5 不稳定")
                st.metric("IC>0占比", f"{ic_pos:.0%}",
                           help=">55% 方向稳定")
                st.metric("月份数", ic["n_months"])

                if abs(ic_mean) > 0.05 and abs(ic_ir) > 0.5:
                    st.success("✅ 因子有效且稳定")
                elif abs(ic_mean) > 0.02:
                    st.warning("⚠️ 因子有微弱预测力")
                else:
                    st.error("❌ 因子无预测力（接近随机）")

        # ── 可视化 ──
        if wfv and spl:
            st.divider()
            st.subheader("📊 验证图表")
            with st.spinner("生成图表..."):
                img_bytes = plot_wfv_charts(code, wfv, spl)
            st.image(img_bytes, use_container_width=True)

        # ── 综合结论 ──
        if wfv and spl and ic:
            st.divider()
            st.subheader("🏁 综合诊断结论")
            score = 0
            checks = []
            if spl.get("decay", 0) > 0.5:
                score += 1; checks.append("✅ 简单切割夏普衰减可接受")
            else:
                checks.append("❌ 简单切割夏普严重衰减")
            if wfv.get("decay", 0) > 0.5:
                score += 1; checks.append("✅ Walk-Forward多窗口衰减可接受")
            else:
                checks.append("❌ Walk-Forward衰减严重")
            if wfv.get("win_rate", 0) > 0.6:
                score += 1; checks.append(f"✅ 样本外正收益年份占比 {wfv['win_rate']:.0%}")
            else:
                checks.append(f"❌ 样本外正收益年份占比 {wfv.get('win_rate',0):.0%}")
            if abs(ic.get("ic_mean", 0)) > 0.02:
                score += 1; checks.append(f"✅ 信号IC均值 {ic['ic_mean']:.4f}（有微弱预测力）")
            else:
                checks.append(f"❌ 信号IC均值 {ic.get('ic_mean',0):.4f}（无预测力）")

            for c in checks:
                st.write(c)

            st.divider()
            if score >= 3:
                st.success(f"**综合评分 {score}/4 → 策略有一定真实优势，可进入6个月模拟盘验证**")
            elif score >= 2:
                st.warning(f"**综合评分 {score}/4 → 优势不明显，建议改进信号再测试**")
            else:
                st.error(f"**综合评分 {score}/4 → 确认过拟合，当前策略不建议实盘**")

    else:
        # 未运行时的说明
        st.info("👈 在左侧设置股票代码和时间范围，点击「开始验证」")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""**✂️ 简单切割**
- 训练集优化参数
- 测试集验证效果
- 计算夏普衰减比""")
        with c2:
            st.markdown("""**🔄 Walk-Forward**
- 多窗口滚动测试
- 拼接真实样本外净值
- 逐年重新优化参数""")
        with c3:
            st.markdown("""**📐 IC分析**
- 衡量信号预测力
- 计算月度IC均值/ICIR
- 判断因子是否有效""")

        st.divider()
        st.markdown("""
**判断标准速查：**

| 指标 | 优秀 | 一般 | 过拟合 |
|------|------|------|--------|
| 夏普衰减比 | > 0.7 | 0.5~0.7 | < 0.5 |
| 样本外正收益年份 | > 60% | 40~60% | < 40% |
| IC均值绝对值 | > 0.05 | 0.02~0.05 | < 0.02 |
| ICIR | > 0.5 | 0.3~0.5 | < 0.3 |
""")

# =========================================================================
# 页面三：基本面选股
# =========================================================================
elif page == "🔍 基本面选股":
    st.subheader("🔍 基本面 + 技术面双重选股")
    st.caption("第一层：基本面筛选估值合理的股票  →  第二层：技术面筛选价量信号良好的标的")

    # 两栏布局：基本面 + 技术面参数
    tab1, tab2 = st.tabs(["📊 第一层：基本面筛选", "📈 第二层：技术面筛选"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            pe_max      = st.slider("市盈率上限 (PE)", 10, 60, 35, 1)
            pb_min      = st.slider("市净率下限 (PB)", 0, 3, 1, 1)
        with col2:
            pb_max      = st.slider("市净率上限 (PB)", 3, 20, 8, 1)
            cap_min     = st.slider("市值下限 (亿)", 50, 500, 200, 50)
        with col3:
            cap_max     = st.slider("市值上限 (亿)", 500, 10000, 3000, 500)
            ret_60d_min = st.slider("近60日涨幅下限 (%)", -20, 20, 0, 1)

    with tab2:
        st.markdown("对基本面筛出的股票进行技术面二次过滤，每只股票计算5个信号，满分5分：")
        col1, col2 = st.columns(2)
        with col1:
            use_tech    = st.checkbox("启用技术面筛选", value=True)
            min_score   = st.slider("技术评分下限（分）", 1, 5, 3, 1,
                                     help="只保留技术评分达到此分数的股票")
            require_ma  = st.checkbox("必须满足均线多头排列", value=True)
        with col2:
            require_vol = st.checkbox("必须满足放量突破", value=False)
            no_diverge  = st.checkbox("排除顶背离预警股票", value=True)
            max_stocks  = st.slider("最多分析前N只股票", 10, 100, 30, 10,
                                     help="技术面分析耗时较长，建议不超过50只")

        st.info("""
        **5个技术信号说明：**
        - 🔵 **均线多头排列**：MA5 > MA10 > MA20，趋势向上
        - 🔵 **放量突破**：成交量 > 20日均量1.5倍，且价格创20日新高
        - 🔵 **无顶背离**：价格创新高时成交量未萎缩（健康上涨）
        - 🔵 **MACD金叉 / 缩量放量**：动能信号出现
        - 🔵 **RSI超卖反弹**：RSI从低位回升，有反弹动能
        """)

    screen_btn = st.button("🔍 开始筛选", type="primary")

    if screen_btn:
        # 第一层：基本面筛选
        with st.status("📥 获取全市场数据（约30秒）...", expanded=True) as status:
            try:
                df_all = get_fundamental_data()
                st.write(f"✅ 共获取 {len(df_all)} 只股票")
                status.update(label="数据获取完成", state="complete")
            except Exception as e:
                status.update(label=f"获取失败: {e}", state="error")
                st.error(f"获取失败，请检查网络: {e}")
                st.stop()

        with st.status("🔍 基本面筛选中...", expanded=True) as status:
            result = screen_stocks(df_all, pe_max=pe_max, pb_min=pb_min,
                                   pb_max=pb_max, cap_min=cap_min,
                                   cap_max=cap_max, ret_60d_min=ret_60d_min)
            st.write(f"✅ 基本面筛选完成，共 {len(result)} 只")
            status.update(label=f"基本面筛选：{len(result)} 只", state="complete")

        st.divider()
        st.subheader(f"📋 第一层结果：基本面筛选（{len(result)} 只）")
        st.dataframe(result, hide_index=True, width="stretch")

        # 基本面分布图
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        pe_bins = [0, 10, 15, 20, 25, 30, 35]
        axes[0].hist(result["pe"].dropna(), bins=pe_bins,
                     color="#2196F3", edgecolor="white")
        axes[0].set_title("PE Distribution", fontsize=12)
        axes[0].set_xlabel("PE Ratio"); axes[0].grid(alpha=0.3, axis="y")
        axes[1].scatter(result["pe"], result["ret_60d"],
                        alpha=0.6, color="#4CAF50", s=30)
        axes[1].set_xlabel("PE Ratio")
        axes[1].set_ylabel("60-Day Return (%)")
        axes[1].set_title("PE vs 60-Day Return", fontsize=12)
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        st.image(_fig_to_buf(fig), width="stretch")

        # 第二层：技术面筛选
        if use_tech and len(result) > 0:
            st.divider()
            st.subheader("📈 第二层：技术面筛选")

            candidates = result["code"].head(max_stocks).tolist()
            tech_results = []

            progress = st.progress(0, text="技术面分析中...")
            for idx, code in enumerate(candidates):
                res = technical_screen(code)
                if res:
                    tech_results.append(res)
                progress.progress((idx + 1) / len(candidates),
                                  text=f"分析中 {idx+1}/{len(candidates)}: {code}")
                time.sleep(0.5)
            progress.empty()

            if not tech_results:
                st.warning("技术面分析失败，请检查网络")
            else:
                tech_df = pd.DataFrame(tech_results)

                # 按条件过滤
                filtered = tech_df[tech_df["_score"] >= min_score].copy()
                if require_ma:
                    filtered = filtered[filtered["均线多头"] == "✅"]
                if require_vol:
                    filtered = filtered[filtered["放量突破"] == "✅"]
                if no_diverge:
                    filtered = filtered[filtered["顶背离预警"] == "✅"]
                filtered = filtered.sort_values("_score", ascending=False)

                # 合并基本面信息
                final = filtered.merge(
                    result[["code","name","price","pe","pb","ret_60d"]],
                    on="code", how="left"
                )
                display_cols = ["code","name","price","pe","pb","ret_60d",
                                "均线多头","放量突破","顶背离预警",
                                "MACD金叉/缩量放量","RSI超卖反弹",
                                "技术评分","RSI当前值"]
                final_display = final[[c for c in display_cols if c in final.columns]]

                st.success(f"🎯 双重筛选完成！基本面{len(result)}只 → 技术面分析{len(candidates)}只 → 最终入选 **{len(final_display)}** 只")
                st.dataframe(final_display, hide_index=True, width="stretch")

                # 技术评分分布图
                if len(tech_df) > 0:
                    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
                    score_counts = tech_df["_score"].value_counts().sort_index()
                    axes2[0].bar(score_counts.index.astype(str),
                                 score_counts.values, color="#9C27B0")
                    axes2[0].set_title("Technical Score Distribution", fontsize=12)
                    axes2[0].set_xlabel("Score (out of 5)")
                    axes2[0].grid(alpha=0.3, axis="y")

                    axes2[1].scatter(tech_df["RSI当前值"],
                                     tech_df["_score"],
                                     alpha=0.6, color="#FF9800", s=50)
                    axes2[1].axvline(30, color="green", linestyle="--",
                                     linewidth=1, label="Oversold 30")
                    axes2[1].axvline(70, color="red", linestyle="--",
                                     linewidth=1, label="Overbought 70")
                    axes2[1].set_xlabel("RSI Value")
                    axes2[1].set_ylabel("Technical Score")
                    axes2[1].set_title("RSI vs Technical Score", fontsize=12)
                    axes2[1].legend(fontsize=9); axes2[1].grid(alpha=0.3)
                    plt.tight_layout()
                    st.image(_fig_to_buf(fig2), width="stretch")

                # 快捷操作
                if len(final_display) > 0:
                    codes_str = "\n".join(final_display["code"].head(6).tolist())
                    st.info(f"💡 双重筛选TOP6，可复制到「策略回测」多股对比：\n```\n{codes_str}\n```")
        elif not use_tech:
            codes_str = "\n".join(result["code"].head(6).tolist())
            st.info(f"💡 基本面TOP6，可复制到「策略回测」多股对比：\n```\n{codes_str}\n```")
