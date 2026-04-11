# =============================================================================
# 量化投资分析平台 v2.0 - 多策略版
# 运行方式: streamlit run quant_app.py
# 新增策略: RSI超买超卖、布林带突破、MACD金叉死叉
# =============================================================================

import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import anthropic
import time
import os
import io

matplotlib.rcParams["font.family"] = "Arial Unicode MS"
os.makedirs("data", exist_ok=True)

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
# 策略回测函数
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
    """策略一：双均线金叉死叉"""
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
    """策略二：RSI超买超卖"""
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


def backtest_boll(code, initial_cash=1_000_000,
                  stop_loss=0.10, take_profit=0.30):
    """策略三：布林带突破"""
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


def backtest_macd(code, initial_cash=1_000_000,
                  stop_loss=0.10, take_profit=0.30):
    """策略四：MACD金叉死叉"""
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
    import itertools
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
# 可视化函数
# =============================================================================

def _fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf


def plot_ma(code, nav_s, fast, slow):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].plot(df["close"], label="收盘价", color="#2196F3", linewidth=1)
    axes[0].plot(df["close"].rolling(fast).mean(), label=f"MA{fast}", color="#FF9800", linewidth=1)
    axes[0].plot(df["close"].rolling(slow).mean(), label=f"MA{slow}", color="#F44336", linewidth=1)
    axes[0].set_title(f"{code} 双均线策略（MA{fast}/MA{slow}）", fontsize=13)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot(nav_s / nav_s.iloc[0], color="#4CAF50", linewidth=1.5, label="策略净值")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("净值"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[2].plot(df["rsi"], color="#9C27B0", linewidth=1, label="RSI(14)")
    axes[2].axhline(70, color="red", linestyle="--", linewidth=0.8)
    axes[2].axhline(30, color="green", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("RSI"); axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


def plot_rsi(code, nav_s, rsi_buy, rsi_sell):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].plot(df["close"], label="收盘价", color="#2196F3", linewidth=1)
    axes[0].set_title(f"{code} RSI策略（买入<{rsi_buy} / 卖出>{rsi_sell}）", fontsize=13)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot(nav_s / nav_s.iloc[0], color="#4CAF50", linewidth=1.5, label="策略净值")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("净值"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[2].plot(df["rsi"], color="#9C27B0", linewidth=1, label="RSI(14)")
    axes[2].axhline(rsi_sell, color="red",   linestyle="--", linewidth=1, label=f"超买{rsi_sell}")
    axes[2].axhline(rsi_buy,  color="green", linestyle="--", linewidth=1, label=f"超卖{rsi_buy}")
    axes[2].fill_between(df.index, rsi_buy, df["rsi"].clip(upper=rsi_buy), alpha=0.2, color="green")
    axes[2].fill_between(df.index, rsi_sell, df["rsi"].clip(lower=rsi_sell), alpha=0.2, color="red")
    axes[2].set_ylabel("RSI"); axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


def plot_boll(code, nav_s):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(df["close"],    label="收盘价", color="#2196F3", linewidth=1)
    axes[0].plot(df["bb_upper"], label="上轨",   color="#F44336", linewidth=1, linestyle="--")
    axes[0].plot(df["bb_mid"],   label="中轨",   color="#FF9800", linewidth=1, linestyle="--")
    axes[0].plot(df["bb_lower"], label="下轨",   color="#4CAF50", linewidth=1, linestyle="--")
    axes[0].fill_between(df.index, df["bb_upper"], df["bb_lower"], alpha=0.08, color="gray")
    axes[0].set_title(f"{code} 布林带策略", fontsize=13)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot(nav_s / nav_s.iloc[0], color="#4CAF50", linewidth=1.5, label="策略净值")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("净值"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


def plot_macd(code, nav_s):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].plot(df["close"], label="收盘价", color="#2196F3", linewidth=1)
    axes[0].set_title(f"{code} MACD策略", fontsize=13)
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot(nav_s / nav_s.iloc[0], color="#4CAF50", linewidth=1.5, label="策略净值")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("净值"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[2].plot(df["macd"],   label="MACD",   color="#2196F3", linewidth=1)
    axes[2].plot(df["signal"], label="Signal", color="#F44336", linewidth=1)
    colors = ["#F44336" if v < 0 else "#4CAF50" for v in df["hist"]]
    axes[2].bar(df.index, df["hist"], color=colors, alpha=0.6, label="柱状图")
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].set_ylabel("MACD"); axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


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
# Streamlit 界面
# =============================================================================

st.set_page_config(page_title="量化投资分析平台", page_icon="📈", layout="wide")
st.title("📈 量化投资分析平台 v2.0")
st.caption("支持双均线、RSI、布林带、MACD四大策略，输入股票代码一键分析")

with st.sidebar:
    st.header("⚙️ 基础设置")
    mode = st.radio("分析模式", ["📊 单股分析", "📈 多股对比"], horizontal=True)
    if mode == "📊 单股分析":
        code       = st.text_input("股票代码", value="600519", help="沪市6开头，深市0/3开头")
        codes_list = [code.strip()] if code.strip() else []
    else:
        codes_input = st.text_area("股票代码（每行一个，最多6只）",
                                    value="600519\n000858\n601899",
                                    help="每行输入一个股票代码")
        codes_list = [c.strip() for c in codes_input.strip().split("\n") if c.strip()][:6]
        code = codes_list[0] if codes_list else ""
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
    stop_loss   = st.slider("止损比例", 5, 20, 10, 1, help="亏损超过此比例自动卖出（%）") / 100
    take_profit = st.slider("止盈比例", 10, 50, 30, 1, help="盈利超过此比例自动卖出（%）") / 100

    if "双均线" in strategy:
        st.divider()
        st.header("📏 均线参数")
        use_auto = st.checkbox("自动优化参数", value=True)
        if not use_auto:
            fast_ma = st.slider("快线周期", 5, 20, 5, 1)
            slow_ma = st.slider("慢线周期", 20, 60, 40, 5)
        else:
            fast_ma, slow_ma = 5, 40

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

    # =========================================================================
    # 多股对比模式
    # =========================================================================
    if mode == "📈 多股对比":
        all_results = []
        all_navs    = {}

        for c in codes_list:
            with st.status(f"⏳ 处理 {c} ...", expanded=False) as status:
                try:
                    download_stock(c, start_date, end_date)
                    calc_indicators(c)

                    if "双均线" in strategy:
                        if use_auto:
                            opt   = optimize_ma_params(c)
                            bf    = int(opt.iloc[0]["fast"])
                            bs    = int(opt.iloc[0]["slow"])
                        else:
                            bf, bs = fast_ma, slow_ma
                        nav_s, _, stats = backtest_ma(c, bf, bs, stop_loss=stop_loss, take_profit=take_profit)
                        sname = f"双均线 MA{bf}/MA{bs}"
                    elif "RSI" in strategy:
                        nav_s, _, stats = backtest_rsi(c, rsi_buy, rsi_sell, stop_loss=stop_loss, take_profit=take_profit)
                        sname = f"RSI({rsi_buy}/{rsi_sell})"
                    elif "布林带" in strategy:
                        nav_s, _, stats = backtest_boll(c, stop_loss=stop_loss, take_profit=take_profit)
                        sname = "布林带"
                    elif "MACD" in strategy:
                        nav_s, _, stats = backtest_macd(c, stop_loss=stop_loss, take_profit=take_profit)
                        sname = "MACD"
                    else:
                        opt   = optimize_ma_params(c)
                        bf    = int(opt.iloc[0]["fast"])
                        bs    = int(opt.iloc[0]["slow"])
                        nav_s, _, stats = backtest_ma(c, bf, bs, stop_loss=stop_loss, take_profit=take_profit)
                        sname = f"双均线 MA{bf}/MA{bs}"

                    all_navs[c]    = nav_s
                    all_results.append({"股票代码": c, "策略": sname, **stats})
                    status.update(label=f"✅ {c} 完成", state="complete")
                except Exception as e:
                    status.update(label=f"❌ {c} 失败: {e}", state="error")

        if not all_results:
            st.error("所有股票处理失败，请检查网络")
            st.stop()

        st.divider()
        st.subheader(f"📈 多股对比结果（{sname}策略）")

        # 对比表格
        compare_df = pd.DataFrame(all_results)[
            ["股票代码","策略","total_ret","sharpe","max_dd","n_trades","final_nav"]
        ]
        compare_df.columns = ["股票代码","策略","总收益率(%)","夏普比率","最大回撤(%)","交易次数","最终净值"]
        best_idx = compare_df["夏普比率"].idxmax()
        st.dataframe(compare_df, hide_index=True, width="stretch")
        st.success(f"🥇 综合表现最优：**{compare_df.iloc[best_idx]['股票代码']}**（夏普比率最高）")

        # 净值曲线对比图
        fig, ax = plt.subplots(figsize=(14, 5))
        for c, nav_s in all_navs.items():
            ax.plot(nav_s / nav_s.iloc[0], label=c, linewidth=1.5)
        ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"多股净值曲线对比（{sname}策略）", fontsize=13)
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        st.image(_fig_to_buf(fig), width="stretch")

        # 柱状图：收益率对比
        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
        codes_show = compare_df["股票代码"].tolist()
        axes2[0].bar(codes_show, compare_df["总收益率(%)"],
                     color=["#4CAF50" if v>0 else "#F44336" for v in compare_df["总收益率(%)"]])
        axes2[0].set_title("总收益率(%)"); axes2[0].grid(alpha=0.3, axis="y")
        axes2[1].bar(codes_show, compare_df["夏普比率"], color="#2196F3")
        axes2[1].set_title("夏普比率"); axes2[1].grid(alpha=0.3, axis="y")
        axes2[2].bar(codes_show, compare_df["最大回撤(%)"], color="#FF9800")
        axes2[2].set_title("最大回撤(%)"); axes2[2].grid(alpha=0.3, axis="y")
        plt.tight_layout()
        st.image(_fig_to_buf(fig2), width="stretch")

        st.subheader("🤖 AI 智能分析")
        if api_key:
            with st.spinner("Claude 正在分析中..."):
                try:
                    best_stats = all_results[best_idx]
                    best_code  = compare_df.iloc[best_idx]["股票代码"]
                    summary    = compare_df.to_string(index=False)
                    analysis   = ai_analysis(
                        best_code,
                        f"多股对比（{sname}策略），对比股票：{', '.join(codes_list)}",
                        best_stats, api_key,
                        f"多股对比汇总:\n{summary}"
                    )
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"AI分析失败: {e}")
        else:
            st.info("填入 Anthropic API Key 即可获得 AI 智能分析报告")

        st.stop()

    # =========================================================================
    # 单股分析模式（原有逻辑）
    # =========================================================================
    with st.status("📥 下载数据中...", expanded=True) as status:
        try:
            df = download_stock(code, start_date, end_date)
            st.write(f"✅ 下载完成，共 {len(df)} 条数据")
            status.update(label="数据下载完成", state="complete")
        except Exception as e:
            status.update(label="下载失败", state="error")
            st.error(f"下载失败，请检查网络或股票代码: {e}")
            st.stop()

    with st.status("📊 计算技术指标...", expanded=True) as status:
        calc_indicators(code)
        st.write("✅ MA、MACD、RSI、布林带计算完成")
        status.update(label="指标计算完成", state="complete")

    if "四策略对比" in strategy:
        with st.status("⚡ 四策略回测中...", expanded=True) as status:
            opt_df    = optimize_ma_params(code)
            best_fast = int(opt_df.iloc[0]["fast"])
            best_slow = int(opt_df.iloc[0]["slow"])
            nav_ma,   _, stats_ma   = backtest_ma(code, best_fast, best_slow, stop_loss=stop_loss, take_profit=take_profit)
            nav_rsi,  _, stats_rsi  = backtest_rsi(code, rsi_buy, rsi_sell, stop_loss=stop_loss, take_profit=take_profit)
            nav_boll, _, stats_boll = backtest_boll(code, stop_loss=stop_loss, take_profit=take_profit)
            nav_macd, _, stats_macd = backtest_macd(code, stop_loss=stop_loss, take_profit=take_profit)
            st.write("✅ 四策略回测完成")
            status.update(label="回测完成", state="complete")

        st.divider()
        st.subheader(f"🏆 {code} 四策略对比结果")

        compare_df = pd.DataFrame([
            {"策略": f"双均线 MA{best_fast}/MA{best_slow}", **stats_ma},
            {"策略": "RSI超买超卖",  **stats_rsi},
            {"策略": "布林带突破",   **stats_boll},
            {"策略": "MACD金叉死叉", **stats_macd},
        ])[["策略","total_ret","sharpe","max_dd","n_trades","final_nav"]]
        compare_df.columns = ["策略","总收益率(%)","夏普比率","最大回撤(%)","交易次数","最终净值"]

        best_idx = compare_df["夏普比率"].idxmax()
        st.dataframe(compare_df, hide_index=True, width="stretch")
        st.success(f"🥇 最优策略：**{compare_df.iloc[best_idx]['策略']}**（夏普比率最高）")

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(nav_ma   / nav_ma.iloc[0],   label=f"双均线 MA{best_fast}/MA{best_slow}", linewidth=1.5)
        ax.plot(nav_rsi  / nav_rsi.iloc[0],  label="RSI超买超卖",  linewidth=1.5)
        ax.plot(nav_boll / nav_boll.iloc[0], label="布林带突破",   linewidth=1.5)
        ax.plot(nav_macd / nav_macd.iloc[0], label="MACD金叉死叉", linewidth=1.5)
        ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"{code} 四策略净值曲线对比", fontsize=13)
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        st.image(_fig_to_buf(fig), width="stretch")

        st.subheader("🤖 AI 智能分析")
        if api_key:
            with st.spinner("Claude 正在分析中..."):
                try:
                    best_stats = [stats_ma, stats_rsi, stats_boll, stats_macd][best_idx]
                    best_name  = compare_df.iloc[best_idx]["策略"]
                    analysis   = ai_analysis(code, f"四策略对比，最优为{best_name}", best_stats, api_key)
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"AI分析失败: {e}")
        else:
            st.info("填入 Anthropic API Key 即可获得 AI 智能分析报告")

    else:
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
                chart_buf     = plot_ma(code, nav_s, best_fast, best_slow)
                strategy_name = f"双均线 MA{best_fast}/MA{best_slow}"
                extra_info    = f"最优均线参数: MA{best_fast}/MA{best_slow}"

            elif "RSI" in strategy:
                nav_s, trades_df, stats = backtest_rsi(code, rsi_buy, rsi_sell, stop_loss=stop_loss, take_profit=take_profit)
                chart_buf     = plot_rsi(code, nav_s, rsi_buy, rsi_sell)
                strategy_name = f"RSI策略（买入<{rsi_buy} / 卖出>{rsi_sell}）"
                extra_info    = f"RSI买入阈值: {rsi_buy}，卖出阈值: {rsi_sell}"

            elif "布林带" in strategy:
                nav_s, trades_df, stats = backtest_boll(code, stop_loss=stop_loss, take_profit=take_profit)
                chart_buf     = plot_boll(code, nav_s)
                strategy_name = "布林带突破策略"
                extra_info    = "布林带参数: 20日均线，2倍标准差"

            elif "MACD" in strategy:
                nav_s, trades_df, stats = backtest_macd(code, stop_loss=stop_loss, take_profit=take_profit)
                chart_buf     = plot_macd(code, nav_s)
                strategy_name = "MACD金叉死叉策略"
                extra_info    = "MACD参数: EMA12/EMA26/Signal9"

            st.write(f"✅ 回测完成，共 {stats['n_trades']} 笔交易")
            status.update(label="回测完成", state="complete")

        st.divider()
        st.subheader(f"📊 {code} · {strategy_name}")

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
        st.image(chart_buf, width="stretch")

        if len(trades_df) > 0:
            st.subheader("📋 交易记录（最近20笔）")
            st.dataframe(trades_df.tail(20), hide_index=True, width="stretch")

        st.subheader("🤖 AI 智能分析")
        if api_key:
            with st.spinner("Claude 正在分析中..."):
                try:
                    analysis = ai_analysis(code, strategy_name, stats, api_key, extra_info)
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"AI分析失败: {e}")
        else:
            st.info("填入 Anthropic API Key 即可获得 AI 智能分析报告")

else:
    st.info("👈 在左侧输入股票代码，选择策略，点击「开始分析」")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        **🔀 双均线策略**
        - 金叉买入/死叉卖出
        - 自动参数优化
        - 适合趋势型股票
        """)
    with col2:
        st.markdown("""
        **📉 RSI策略**
        - 超卖买入/超买卖出
        - 可调买卖阈值
        - 适合震荡型股票
        """)
    with col3:
        st.markdown("""
        **📊 布林带策略**
        - 突破上轨买入
        - 跌破中轨卖出
        - 捕捉强势突破
        """)
    with col4:
        st.markdown("""
        **🏆 四策略对比**
        - 同时运行四个策略
        - 净值曲线对比图
        - AI推荐最优策略
        """)
