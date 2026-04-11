# =============================================================================
# 量化投资分析平台 - Streamlit Web 应用
# 运行方式: streamlit run quant_app.py
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
# 数据与分析函数
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


def run_backtest(code, fast, slow, initial_cash=1_000_000,
                 stop_loss=0.10, take_profit=0.30):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    df[f"ma{fast}"] = df["close"].rolling(fast).mean()
    df[f"ma{slow}"] = df["close"].rolling(slow).mean()
    df = df.dropna()

    cash, position, buy_price = initial_cash, 0, 0
    nav, trades = [], []

    for i in range(1, len(df)):
        price     = df["close"].iloc[i]
        fast_prev = df[f"ma{fast}"].iloc[i-1]
        slow_prev = df[f"ma{slow}"].iloc[i-1]
        fast_curr = df[f"ma{fast}"].iloc[i]
        slow_curr = df[f"ma{slow}"].iloc[i]

        if position > 0:
            pnl = (price - buy_price) / buy_price
            if pnl < -stop_loss:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "止损",
                               "price": price, "pnl_pct": round(pnl*100, 2)})
                position = 0
            elif pnl > take_profit:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "止盈",
                               "price": price, "pnl_pct": round(pnl*100, 2)})
                position = 0
            elif fast_prev > slow_prev and fast_curr < slow_curr:
                cash += position * price * 0.999
                trades.append({"date": df.index[i], "action": "信号卖出",
                               "price": price, "pnl_pct": round(pnl*100, 2)})
                position = 0
        elif fast_prev < slow_prev and fast_curr > slow_curr:
            shares = int(cash / price / 100) * 100
            if shares > 0:
                cash -= shares * price * 1.001
                position  = shares
                buy_price = price
                trades.append({"date": df.index[i], "action": "买入",
                               "price": price, "pnl_pct": 0})
        nav.append(cash + position * price)

    nav_s     = pd.Series(nav, index=df.index[1:])
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    total_ret = (nav_s.iloc[-1] / initial_cash - 1) * 100
    daily_ret = nav_s.pct_change().dropna()
    sharpe    = daily_ret.mean() / daily_ret.std() * (252 ** 0.5)
    max_dd    = ((nav_s - nav_s.cummax()) / nav_s.cummax()).min() * 100

    stats = {
        "total_ret": round(total_ret, 2),
        "sharpe":    round(sharpe, 3),
        "max_dd":    round(max_dd, 2),
        "n_trades":  len(trades_df),
        "n_stop":    len(trades_df[trades_df["action"]=="止损"])   if len(trades_df) > 0 else 0,
        "n_profit":  len(trades_df[trades_df["action"]=="止盈"])   if len(trades_df) > 0 else 0,
        "n_signal":  len(trades_df[trades_df["action"]=="信号卖出"]) if len(trades_df) > 0 else 0,
        "final_nav": round(nav_s.iloc[-1], 0),
    }
    return nav_s, trades_df, stats


def optimize_params(code):
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
        sharpe    = daily_ret.mean() / daily_ret.std() * (252 ** 0.5)
        total_ret = (nav_s.iloc[-1] / 1_000_000 - 1) * 100
        max_dd    = ((nav_s - nav_s.cummax()) / nav_s.cummax()).min() * 100
        results.append({"fast": fast, "slow": slow,
                        "total_ret": round(total_ret, 2),
                        "sharpe": round(sharpe, 3),
                        "max_dd": round(max_dd, 2)})
    return pd.DataFrame(results).sort_values("sharpe", ascending=False)


def plot_analysis(code, nav_s):
    df = pd.read_csv(f"data/{code}_indicators.csv",
                     index_col="date", parse_dates=True).dropna()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    ax1 = axes[0]
    ax1.plot(df["close"],    label="收盘价", color="#2196F3", linewidth=1)
    ax1.plot(df["ma5"],      label="MA5",    color="#FF9800", linewidth=1)
    ax1.plot(df["ma20"],     label="MA20",   color="#F44336", linewidth=1)
    ax1.fill_between(df.index, df["bb_upper"], df["bb_lower"],
                     alpha=0.1, color="gray", label="布林带")
    ax1.set_title(f"{code} 技术分析", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(nav_s / nav_s.iloc[0], color="#4CAF50", linewidth=1.5, label="策略净值")
    ax2.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("净值")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    ax3.plot(df["rsi"], color="#9C27B0", linewidth=1, label="RSI(14)")
    ax3.axhline(70, color="red",   linestyle="--", linewidth=0.8)
    ax3.axhline(30, color="green", linestyle="--", linewidth=0.8)
    ax3.set_ylabel("RSI")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close()
    return buf


def ai_analysis(code, stats, opt_df, api_key):
    """调用 Claude API 对回测结果进行智能分析"""
    client = anthropic.Anthropic(api_key=api_key)
    best   = opt_df.iloc[0]

    prompt = f"""
你是一位专业的量化投资分析师，请对以下A股股票回测结果进行深度分析，用中文回答。

股票代码: {code}
回测区间: 2020-2024年（5年）
最优参数: MA{int(best['fast'])} / MA{int(best['slow'])}

回测绩效指标:
- 总收益率: {stats['total_ret']}%
- 夏普比率: {stats['sharpe']}
- 最大回撤: {stats['max_dd']}%
- 总交易次数: {stats['n_trades']} 笔
- 止损触发: {stats['n_stop']} 次
- 止盈触发: {stats['n_profit']} 次
- 信号卖出: {stats['n_signal']} 次
- 最终净值: ¥{stats['final_nav']:,.0f}（初始¥1,000,000）

参数优化TOP5结果:
{opt_df.head().to_string(index=False)}

请从以下几个维度进行分析（每个维度2-3句话）:
1. 整体表现评价（收益和风险是否匹配）
2. 策略适配性（这只股票适不适合均线策略，为什么）
3. 风险提示（主要风险点在哪里）
4. 改进建议（如何进一步优化这个策略）
5. 一句话投资结论

请保持专业、客观，避免给出明确的买卖建议。
"""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


# =============================================================================
# Streamlit 界面
# =============================================================================

st.set_page_config(page_title="量化投资分析平台", page_icon="📈", layout="wide")

st.title("📈 量化投资分析平台")
st.caption("输入股票代码，自动完成数据下载、技术分析、回测优化与AI解读")

# 侧边栏参数设置
with st.sidebar:
    st.header("⚙️ 参数设置")
    code        = st.text_input("股票代码", value="600519", 
                                 help="沪市6开头，深市0/3开头")
    start_date  = st.text_input("开始日期", value="20200101")
    end_date    = st.text_input("结束日期", value="20241231")
    stop_loss   = st.slider("止损比例", 5, 20, 10, 1,
                         help="亏损超过此比例自动卖出（%）") / 100
    take_profit = st.slider("止盈比例", 10, 50, 30, 1,
                         help="盈利超过此比例自动卖出（%）") / 100
    st.divider()
    st.header("🤖 AI 分析")
    api_key     = st.text_input("Anthropic API Key", type="password",
                                 help="填入后自动生成AI分析报告")
    run_btn     = st.button("🚀 开始分析", type="primary", use_container_width=True)

# 主界面
if run_btn:
    if not code:
        st.error("请输入股票代码")
        st.stop()

    # Step 1: 下载数据
    with st.status("📥 正在下载数据...", expanded=True) as status:
        try:
            df = download_stock(code, start_date, end_date)
            st.write(f"✅ 下载完成，共 {len(df)} 条数据")
            status.update(label="数据下载完成", state="complete")
        except Exception as e:
            status.update(label=f"下载失败: {e}", state="error")
            st.error(f"数据下载失败，请检查网络或股票代码: {e}")
            st.stop()

    # Step 2: 计算指标
    with st.status("📊 计算技术指标...", expanded=True) as status:
        df_ind = calc_indicators(code)
        st.write("✅ MA、MACD、RSI、布林带计算完成")
        status.update(label="指标计算完成", state="complete")

    # Step 3: 参数优化
    with st.status("🔍 参数优化中（16种组合）...", expanded=True) as status:
        opt_df    = optimize_params(code)
        best_fast = int(opt_df.iloc[0]["fast"])
        best_slow = int(opt_df.iloc[0]["slow"])
        st.write(f"✅ 最优参数: MA{best_fast} / MA{best_slow}")
        status.update(label=f"参数优化完成：最优 MA{best_fast}/MA{best_slow}", state="complete")

    # Step 4: 回测
    with st.status("⚡ 执行回测...", expanded=True) as status:
        nav_s, trades_df, stats = run_backtest(
            code, best_fast, best_slow,
            stop_loss=stop_loss, take_profit=take_profit
        )
        st.write(f"✅ 回测完成，共 {stats['n_trades']} 笔交易")
        status.update(label="回测完成", state="complete")

    # ---- 展示结果 ----
    st.divider()
    st.subheader(f"📊 {code} 分析结果")

    # 核心指标卡片
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总收益率",  f"{stats['total_ret']}%",
                delta=f"{'↑' if stats['total_ret']>0 else '↓'}")
    col2.metric("夏普比率",  f"{stats['sharpe']}")
    col3.metric("最大回撤",  f"{stats['max_dd']}%")
    col4.metric("最终净值",  f"¥{stats['final_nav']:,.0f}")

    col5, col6, col7 = st.columns(3)
    col5.metric("止损触发", f"{stats['n_stop']} 次")
    col6.metric("止盈触发", f"{stats['n_profit']} 次")
    col7.metric("信号卖出", f"{stats['n_signal']} 次")

    # 图表
    st.subheader("📈 技术分析图")
    chart_buf = plot_analysis(code, nav_s)
    st.image(chart_buf, use_container_width=True)

    # 参数优化结果
    st.subheader("🔧 参数优化结果")
    st.dataframe(opt_df, use_container_width=True, hide_index=True)

    # 交易记录
    if len(trades_df) > 0:
        st.subheader("📋 交易记录")
        st.dataframe(trades_df.tail(20), use_container_width=True, hide_index=True)

    # AI 分析
    st.subheader("🤖 AI 智能分析")
    if api_key:
        with st.spinner("Claude 正在分析中..."):
            try:
                analysis = ai_analysis(code, stats, opt_df, api_key)
                st.markdown(analysis)
            except Exception as e:
                st.error(f"AI 分析失败: {e}")
    else:
        st.info("在左侧填入 Anthropic API Key 即可获得 AI 智能分析报告")

else:
    # 默认首页提示
    st.info("👈 在左侧输入股票代码，点击「开始分析」")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📥 数据模块**
        - 自动下载日K线
        - 支持全A股
        - 前复权处理
        """)
    with col2:
        st.markdown("""
        **⚡ 回测模块**
        - 双均线策略
        - 自动参数优化
        - 止损止盈控制
        """)
    with col3:
        st.markdown("""
        **🤖 AI分析模块**
        - Claude智能解读
        - 多维度评价
        - 风险提示
        """)
