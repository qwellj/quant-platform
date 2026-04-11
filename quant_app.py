# =============================================================================
# 页面二：选股中心（将下面这段替换 quant_app.py 中原来的 elif page == "🔍 基本面选股": 部分）
# =============================================================================

elif page == "🔍 基本面选股":

    # ── 顶部：模式选择 ────────────────────────────────────────────────────────
    st.subheader("🔍 选股中心")

    screen_mode = st.radio(
        "选股模式",
        ["📊 仅基本面选股", "📈 仅技术面选股", "🔀 基本面 + 技术面联合选股"],
        horizontal=True,
        help=(
            "仅基本面：从全市场按估值指标筛选  |  "
            "仅技术面：对指定股票池分析价量信号  |  "
            "联合选股：先基本面缩圈，再技术面精选"
        )
    )
    st.divider()

    # ── 参数面板（按模式显示不同组合）────────────────────────────────────────

    # ---------- 基本面参数 ----------
    if screen_mode in ["📊 仅基本面选股", "🔀 基本面 + 技术面联合选股"]:
        st.markdown("#### 📊 基本面筛选条件")
        col1, col2, col3 = st.columns(3)
        with col1:
            pe_max      = st.slider("市盈率上限 (PE)",  10,  60,  35, 1)
            pb_min      = st.slider("市净率下限 (PB)",   0,   3,   1, 1)
        with col2:
            pb_max      = st.slider("市净率上限 (PB)",   3,  20,   8, 1)
            cap_min     = st.slider("市值下限 (亿)",    50, 500, 200, 50)
        with col3:
            cap_max     = st.slider("市值上限 (亿)",   500, 10000, 3000, 500)
            ret_60d_min = st.slider("近60日涨幅下限 (%)", -20, 20, 0, 1)
    else:
        # 仅技术面模式：给默认值（后面不会用到，但避免 NameError）
        pe_max = 35; pb_min = 1; pb_max = 8
        cap_min = 200; cap_max = 3000; ret_60d_min = 0

    # ---------- 仅技术面：自定义股票池 ----------
    if screen_mode == "📈 仅技术面选股":
        st.divider()
        st.markdown("#### 📋 待分析股票池")
        st.caption("输入你想做技术面筛选的股票代码，每行一个（最多100只）")
        manual_input = st.text_area(
            "股票代码（每行一个）",
            value="600519\n000858\n601899\n000333\n600036",
            height=160,
            label_visibility="collapsed"
        )
        manual_codes = [c.strip() for c in manual_input.strip().split("\n")
                        if c.strip() and c.strip().isdigit()][:100]
        st.caption(f"已输入 **{len(manual_codes)}** 只股票")
    else:
        manual_codes = []

    # ---------- 技术面参数 ----------
    if screen_mode in ["📈 仅技术面选股", "🔀 基本面 + 技术面联合选股"]:
        st.divider()
        st.markdown("#### 📈 技术面筛选条件")
        st.caption("每只股票计算5个价量信号，满分5分")

        col1, col2 = st.columns(2)
        with col1:
            min_score  = st.slider("技术评分下限（分）", 1, 5, 3, 1,
                                   help="只保留达到此评分的股票")
            require_ma = st.checkbox("必须满足均线多头排列 (MA5>MA10>MA20)", value=True)
            require_vol= st.checkbox("必须满足放量突破信号", value=False)
        with col2:
            no_diverge = st.checkbox("排除顶背离预警股票", value=True)
            if screen_mode == "🔀 基本面 + 技术面联合选股":
                max_stocks = st.slider("基本面结果中最多分析前N只", 10, 100, 30, 10,
                                       help="技术面分析较耗时，建议不超过50只")
            else:
                max_stocks = len(manual_codes) if manual_codes else 50

        with st.expander("📖 5个技术信号说明"):
            st.markdown("""
| 信号 | 触发条件 |
|------|----------|
| 🔵 均线多头排列 | MA5 > MA10 > MA20，短中期趋势均向上 |
| 🔵 放量突破 | 成交量 > 20日均量×1.5，且价格创20日新高 |
| 🔵 无顶背离 | 价格创新高时成交量未萎缩（上涨健康） |
| 🔵 MACD金叉 / 缩量放量 | MACD上穿Signal线，或短期缩量后突然放量 |
| 🔵 RSI超卖反弹 | RSI在30~50区间且价格近3日回升 |
            """)
    else:
        # 仅基本面模式：给默认值
        min_score = 3; require_ma = True; require_vol = False
        no_diverge = True; max_stocks = 30

    # ── 开始筛选按钮 ──────────────────────────────────────────────────────────
    st.divider()
    label_map = {
        "📊 仅基本面选股":          "📊 开始基本面筛选",
        "📈 仅技术面选股":          "📈 开始技术面筛选",
        "🔀 基本面 + 技术面联合选股": "🔀 开始联合选股",
    }
    screen_btn = st.button(label_map[screen_mode], type="primary")

    # ── 执行逻辑 ──────────────────────────────────────────────────────────────
    if screen_btn:

        # ====================================================================
        # 模式A：仅基本面
        # ====================================================================
        if screen_mode == "📊 仅基本面选股":

            with st.status("📥 获取全市场数据（约30秒）...", expanded=True) as status:
                try:
                    df_all = get_fundamental_data()
                    st.write(f"✅ 共获取 {len(df_all)} 只股票")
                    status.update(label="数据获取完成", state="complete")
                except Exception as e:
                    status.update(label=f"获取失败: {e}", state="error")
                    st.error(f"获取失败，请检查网络: {e}"); st.stop()

            with st.status("🔍 基本面筛选中...", expanded=True) as status:
                result = screen_stocks(df_all, pe_max=pe_max, pb_min=pb_min,
                                       pb_max=pb_max, cap_min=cap_min,
                                       cap_max=cap_max, ret_60d_min=ret_60d_min)
                st.write(f"✅ 筛选完成，共 {len(result)} 只")
                status.update(label=f"基本面筛选：{len(result)} 只", state="complete")

            st.divider()
            st.subheader(f"📋 基本面筛选结果（{len(result)} 只）")
            st.dataframe(result, hide_index=True, use_container_width=True)

            # 分布图
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].hist(result["pe"].dropna(), bins=[0,10,15,20,25,30,35],
                         color="#2196F3", edgecolor="white")
            axes[0].set_title("PE Distribution"); axes[0].set_xlabel("PE Ratio")
            axes[0].grid(alpha=0.3, axis="y")
            axes[1].scatter(result["pe"], result["ret_60d"],
                            alpha=0.6, color="#4CAF50", s=30)
            axes[1].set_xlabel("PE Ratio"); axes[1].set_ylabel("60-Day Return (%)")
            axes[1].set_title("PE vs 60-Day Return"); axes[1].grid(alpha=0.3)
            plt.tight_layout()
            st.image(_fig_to_buf(fig), use_container_width=True)

            if len(result) > 0:
                codes_str = "\n".join(result["code"].head(6).tolist())
                st.info(f"💡 基本面 TOP6，可复制到「策略回测」进行多股对比：\n```\n{codes_str}\n```")

        # ====================================================================
        # 模式B：仅技术面
        # ====================================================================
        elif screen_mode == "📈 仅技术面选股":

            if not manual_codes:
                st.error("请先输入至少一只股票代码"); st.stop()

            st.subheader(f"📈 对 {len(manual_codes)} 只股票进行技术面分析")

            tech_results = []
            progress = st.progress(0, text="技术面分析中...")
            for idx, code in enumerate(manual_codes):
                res = technical_screen(code)
                if res:
                    tech_results.append(res)
                progress.progress(
                    (idx + 1) / len(manual_codes),
                    text=f"分析中 {idx+1}/{len(manual_codes)}: {code}"
                )
                time.sleep(0.3)
            progress.empty()

            if not tech_results:
                st.warning("所有股票技术面分析均失败，请检查网络或股票代码")
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

                display_cols = ["code","均线多头","放量突破","顶背离预警",
                                "MACD金叉/缩量放量","RSI超卖反弹","技术评分","RSI当前值"]
                final_display = filtered[[c for c in display_cols if c in filtered.columns]]

                st.success(
                    f"📈 技术面筛选完成！分析 {len(manual_codes)} 只 "
                    f"→ 通过 **{len(final_display)}** 只"
                )
                st.dataframe(final_display, hide_index=True, use_container_width=True)

                # 评分分布图
                if len(tech_df) > 0:
                    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
                    score_counts = tech_df["_score"].value_counts().sort_index()
                    axes2[0].bar(score_counts.index.astype(str), score_counts.values,
                                 color="#9C27B0")
                    axes2[0].set_title("Technical Score Distribution")
                    axes2[0].set_xlabel("Score (out of 5)"); axes2[0].grid(alpha=0.3, axis="y")
                    axes2[1].scatter(tech_df["RSI当前值"], tech_df["_score"],
                                     alpha=0.6, color="#FF9800", s=50)
                    axes2[1].axvline(30, color="green", linestyle="--", linewidth=1, label="Oversold 30")
                    axes2[1].axvline(70, color="red",   linestyle="--", linewidth=1, label="Overbought 70")
                    axes2[1].set_xlabel("RSI Value"); axes2[1].set_ylabel("Score")
                    axes2[1].set_title("RSI vs Technical Score")
                    axes2[1].legend(fontsize=9); axes2[1].grid(alpha=0.3)
                    plt.tight_layout()
                    st.image(_fig_to_buf(fig2), use_container_width=True)

                if len(final_display) > 0:
                    codes_str = "\n".join(final_display["code"].head(6).tolist())
                    st.info(f"💡 技术面 TOP6，可复制到「策略回测」进行多股对比：\n```\n{codes_str}\n```")

        # ====================================================================
        # 模式C：基本面 + 技术面联合选股
        # ====================================================================
        elif screen_mode == "🔀 基本面 + 技术面联合选股":

            # 第一层：基本面
            with st.status("📥 获取全市场数据（约30秒）...", expanded=True) as status:
                try:
                    df_all = get_fundamental_data()
                    st.write(f"✅ 共获取 {len(df_all)} 只股票")
                    status.update(label="数据获取完成", state="complete")
                except Exception as e:
                    status.update(label=f"获取失败: {e}", state="error")
                    st.error(f"获取失败，请检查网络: {e}"); st.stop()

            with st.status("🔍 第一层：基本面筛选...", expanded=True) as status:
                result = screen_stocks(df_all, pe_max=pe_max, pb_min=pb_min,
                                       pb_max=pb_max, cap_min=cap_min,
                                       cap_max=cap_max, ret_60d_min=ret_60d_min)
                st.write(f"✅ 基本面筛选完成，共 {len(result)} 只")
                status.update(label=f"第一层完成：{len(result)} 只", state="complete")

            st.divider()
            st.subheader(f"📋 第一层结果：基本面筛选（{len(result)} 只）")
            st.dataframe(result, hide_index=True, use_container_width=True)

            # 基本面分布图
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].hist(result["pe"].dropna(), bins=[0,10,15,20,25,30,35],
                         color="#2196F3", edgecolor="white")
            axes[0].set_title("PE Distribution"); axes[0].set_xlabel("PE Ratio")
            axes[0].grid(alpha=0.3, axis="y")
            axes[1].scatter(result["pe"], result["ret_60d"],
                            alpha=0.6, color="#4CAF50", s=30)
            axes[1].set_xlabel("PE Ratio"); axes[1].set_ylabel("60-Day Return (%)")
            axes[1].set_title("PE vs 60-Day Return"); axes[1].grid(alpha=0.3)
            plt.tight_layout()
            st.image(_fig_to_buf(fig), use_container_width=True)

            # 第二层：技术面
            if len(result) == 0:
                st.warning("基本面筛选结果为空，请放宽筛选条件")
            else:
                st.divider()
                st.subheader("📈 第二层：技术面精选")

                candidates = result["code"].head(max_stocks).tolist()
                tech_results = []
                progress = st.progress(0, text="技术面分析中...")
                for idx, code in enumerate(candidates):
                    res = technical_screen(code)
                    if res:
                        tech_results.append(res)
                    progress.progress(
                        (idx + 1) / len(candidates),
                        text=f"分析中 {idx+1}/{len(candidates)}: {code}"
                    )
                    time.sleep(0.5)
                progress.empty()

                if not tech_results:
                    st.warning("技术面分析全部失败，请检查网络")
                else:
                    tech_df = pd.DataFrame(tech_results)

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
                                    "MACD金叉/缩量放量","RSI超卖反弹","技术评分","RSI当前值"]
                    final_display = final[[c for c in display_cols if c in final.columns]]

                    st.success(
                        f"🎯 联合筛选完成！基本面 {len(result)} 只 "
                        f"→ 技术面分析 {len(candidates)} 只 "
                        f"→ 最终入选 **{len(final_display)}** 只"
                    )
                    st.dataframe(final_display, hide_index=True, use_container_width=True)

                    # 技术评分分布图
                    if len(tech_df) > 0:
                        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
                        score_counts = tech_df["_score"].value_counts().sort_index()
                        axes2[0].bar(score_counts.index.astype(str), score_counts.values,
                                     color="#9C27B0")
                        axes2[0].set_title("Technical Score Distribution")
                        axes2[0].set_xlabel("Score (out of 5)"); axes2[0].grid(alpha=0.3, axis="y")
                        axes2[1].scatter(tech_df["RSI当前值"], tech_df["_score"],
                                         alpha=0.6, color="#FF9800", s=50)
                        axes2[1].axvline(30, color="green", linestyle="--", linewidth=1, label="Oversold 30")
                        axes2[1].axvline(70, color="red",   linestyle="--", linewidth=1, label="Overbought 70")
                        axes2[1].set_xlabel("RSI Value"); axes2[1].set_ylabel("Score")
                        axes2[1].set_title("RSI vs Technical Score")
                        axes2[1].legend(fontsize=9); axes2[1].grid(alpha=0.3)
                        plt.tight_layout()
                        st.image(_fig_to_buf(fig2), use_container_width=True)

                    if len(final_display) > 0:
                        codes_str = "\n".join(final_display["code"].head(6).tolist())
                        st.info(
                            f"💡 联合选股 TOP6，可复制到「策略回测」多股对比：\n```\n{codes_str}\n```"
                        )
