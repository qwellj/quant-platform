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
                stop_loss=0.10, take_profit=0.30):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    df["maf"] = df["close"].rolling(fast).mean()
    df["mas"] = df["close"].rolling(slow).mean()
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
            shares = int(cash / price / 100) * 100
            if shares > 0:
                cash -= shares * price * 1.001
                position, buy_price = shares, price
                trades.append({"date": df.index[i], "action": "买入", "price": price, "pnl_pct": 0})
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
    """生成多页 PDF 分析报告（接收PNG buffer）"""
    from PIL import Image
    buf = io.BytesIO()
    with pdf_backend.PdfPages(buf) as pdf:

        # 封面
        fig0, ax0 = plt.subplots(figsize=(11, 8.5))
        ax0.axis("off")
        ax0.text(0.5, 0.72, "Quantitative Analysis Report",
                 ha="center", va="center", fontsize=28, fontweight="bold",
                 transform=ax0.transAxes)
        ax0.text(0.5, 0.60, f"{code}  {stock_name}",
                 ha="center", va="center", fontsize=20,
                 transform=ax0.transAxes, color="#2196F3")
        ax0.text(0.5, 0.50, f"Strategy: {strategy_name}",
                 ha="center", va="center", fontsize=16,
                 transform=ax0.transAxes)
        summary = (
            f"Total Return: {stats['total_ret']}%     "
            f"Sharpe: {stats['sharpe']}     "
            f"Max Drawdown: {stats['max_dd']}%\n"
            f"Trades: {stats['n_trades']}     "
            f"Final NAV: {stats['final_nav']:,.0f}"
        )
        ax0.text(0.5, 0.36, summary,
                 ha="center", va="center", fontsize=13,
                 transform=ax0.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", alpha=0.8))
        ax0.text(0.5, 0.08, "Generated by Quant Platform v3.0",
                 ha="center", va="center", fontsize=10, color="gray",
                 transform=ax0.transAxes)
        pdf.savefig(fig0, bbox_inches="tight")
        plt.close(fig0)

        # 策略图表页（从buffer读取图片嵌入）
        for img_buf, title in [(img_main_buf, "Strategy Chart"),
                               (img_bench_buf, "Benchmark & Annual Returns")]:
            img_buf.seek(0)
            img = Image.open(img_buf)
            fig_img, ax_img = plt.subplots(figsize=(11, 7))
            ax_img.imshow(img)
            ax_img.axis("off")
            ax_img.set_title(title, fontsize=14, pad=10)
            pdf.savefig(fig_img, bbox_inches="tight")
            plt.close(fig_img)

        # 年度收益表格页
        fig_tbl, ax_tbl = plt.subplots(figsize=(11, 4))
        ax_tbl.axis("off")
        tbl_data = [[str(y), f"{v:.2f}%", "▲" if v >= 0 else "▼"]
                    for y, v in annual.items()]
        tbl = ax_tbl.table(
            cellText=tbl_data,
            colLabels=["Year", "Return", "Direction"],
            loc="center", cellLoc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(13)
        tbl.scale(1.5, 2.2)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2196F3")
                cell.set_text_props(color="white", fontweight="bold")
            elif row > 0 and tbl_data[row-1][2] == "▲":
                cell.set_facecolor("#E8F5E9")
            else:
                cell.set_facecolor("#FFEBEE")
        ax_tbl.set_title("Annual Returns Summary", fontsize=14, pad=20)
        pdf.savefig(fig_tbl, bbox_inches="tight")
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


# =============================================================================
# Streamlit 界面
# =============================================================================

st.set_page_config(page_title="量化投资分析平台", page_icon="📈", layout="wide")
st.title("📈 量化投资分析平台 v3.0")
st.caption("策略回测 · 基准对比 · 年度收益 · 基本面选股 · PDF报告导出")

# 顶部页面导航
page = st.radio("", ["📊 策略回测", "🔍 基本面选股"], horizontal=True, label_visibility="collapsed")
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
        else:
            use_auto, fast_ma, slow_ma = True, 5, 40

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
                            nav_s, _, stats = backtest_ma(c, bf, bs, stop_loss=stop_loss, take_profit=take_profit)
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
                "MA":        lambda c: backtest_ma(c, stop_loss=stop_loss, take_profit=take_profit),
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
                nav_s, trades_df, stats = backtest_ma(code, best_fast, best_slow, stop_loss=stop_loss, take_profit=take_profit)
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
                nav_ma,   _, stats_ma   = backtest_ma(code, best_fast, best_slow, stop_loss=stop_loss, take_profit=take_profit)
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
        if st.button("生成 PDF 报告", type="secondary"):
            if "img_main" not in st.session_state:
                st.error("请先完成回测分析，再生成PDF")
            else:
                with st.spinner("正在生成 PDF..."):
                    meta = st.session_state["pdf_meta"]
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

# =========================================================================
# 页面二：基本面选股
# =========================================================================
elif page == "🔍 基本面选股":
    st.subheader("🔍 基本面选股")
    st.caption("从全市场5800只股票中筛选估值合理、趋势向上的优质标的")

    col1, col2, col3 = st.columns(3)
    with col1:
        pe_max    = st.slider("市盈率上限 (PE)", 10, 60, 35, 1)
        pb_min    = st.slider("市净率下限 (PB)", 0, 3, 1, 1)
    with col2:
        pb_max    = st.slider("市净率上限 (PB)", 3, 20, 8, 1)
        cap_min   = st.slider("市值下限 (亿)", 50, 500, 200, 50)
    with col3:
        cap_max   = st.slider("市值上限 (亿)", 500, 10000, 3000, 500)
        ret_60d_min = st.slider("近60日涨幅下限 (%)", -20, 20, 0, 1)

    screen_btn = st.button("🔍 开始筛选", type="primary")

    if screen_btn:
        with st.status("📥 获取全市场数据（约30秒）...", expanded=True) as status:
            try:
                df_all = get_fundamental_data()
                st.write(f"✅ 共获取 {len(df_all)} 只股票")
                status.update(label="数据获取完成", state="complete")
            except Exception as e:
                status.update(label=f"获取失败: {e}", state="error")
                st.error(f"获取失败，请检查网络: {e}")
                st.stop()

        with st.status("🔍 筛选中...", expanded=True) as status:
            result = screen_stocks(df_all, pe_max=pe_max, pb_min=pb_min,
                                   pb_max=pb_max, cap_min=cap_min,
                                   cap_max=cap_max, ret_60d_min=ret_60d_min)
            st.write(f"✅ 筛选完成，共 {len(result)} 只")
            status.update(label=f"筛选完成：{len(result)} 只", state="complete")

        st.divider()
        st.subheader(f"📋 筛选结果（共 {len(result)} 只）")
        st.dataframe(result, hide_index=True, width="stretch")

        # 快捷操作：一键复制股票代码到回测
        codes_str = "\n".join(result["code"].head(6).tolist())
        st.info(f"💡 可将以下代码复制到「策略回测」的多股对比输入框：\n```\n{codes_str}\n```")

        # 行业分布图
        if len(result) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            pe_bins = [0, 10, 15, 20, 25, 30, 35]
            axes[0].hist(result["pe"].dropna(), bins=pe_bins, color="#2196F3", edgecolor="white")
            axes[0].set_title("PE Distribution", fontsize=12)
            axes[0].set_xlabel("PE Ratio"); axes[0].grid(alpha=0.3, axis="y")
            axes[1].scatter(result["pe"], result["ret_60d"],
                            alpha=0.6, color="#4CAF50", s=30)
            axes[1].set_xlabel("PE Ratio"); axes[1].set_ylabel("60-Day Return (%)")
            axes[1].set_title("PE vs 60-Day Return", fontsize=12)
            axes[1].grid(alpha=0.3)
            plt.tight_layout()
            st.image(_fig_to_buf(fig), width="stretch")
