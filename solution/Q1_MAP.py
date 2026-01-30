"""
MCM 2026 Problem C - Question 1 (MAP/约束优化版)
================================================
思路：
1) 每周每季对选手投票份额 v 做约束优化：
   - v >= 0, sum(v)=1
   - 淘汰约束：淘汰者组合得分需最低（线性化近似）
2) 目标：让 v 尽量接近先验（由评委得分软最大得到）
3) 输出票数估计与一致性评估，并生成论文风格图表
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

warnings.filterwarnings("ignore")

# -----------------------------
# 全局配置 & 可视化风格（与 data_visualization.py 对齐）
# -----------------------------
np.random.seed(42)

COLORS = {
    "primary": "#7BADDF",      # 浅蓝
    "secondary": "#B581B4",    # 薰衣草紫
    "accent": "#EAB170",       # 暖橙
    "success": "#DA8176",      # 珊瑚粉
    "neutral": "#B1A8D3",      # 淡紫
    "light": "#BADDF3"         # 极浅蓝
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.6,
})

OUTPUT_DIR = Path("plots/q1_map")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 辅助函数
# -----------------------------
def softmax(x, temp=1.0):
    x = np.asarray(x) / max(temp, 1e-8)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def build_truth_map(df_clean: pd.DataFrame):
    df_truth = df_clean[["season", "celebrity_name", "elimination_week"]].copy()
    df_truth["elim_week"] = np.where(
        df_truth["elimination_week"].notna() & (df_truth["elimination_week"] > 0),
        df_truth["elimination_week"],
        np.nan
    )
    truth_map = (
        df_truth.dropna(subset=["elim_week"])
        .groupby(["season", "elim_week"])["celebrity_name"]
        .apply(list)
        .to_dict()
    )
    return truth_map


def get_rule_sets(max_season: int):
    rank_seasons = set([1, 2]) | set(range(28, max_season + 1))
    pct_seasons = set(range(3, 28))
    return rank_seasons, pct_seasons


def solve_week_vote_share(
    judge_total: np.ndarray,
    prior: np.ndarray,
    elim_mask: np.ndarray,
    alpha_score: float = 0.5,
    margin: float = 1e-4
):
    """
    约束优化求解投票份额 v:
      - v >= 0, sum v = 1
      - 淘汰约束：淘汰者组合得分最低（线性近似）
    组合得分：combined = alpha_score * judge_norm + (1-alpha_score) * v
    """
    n = len(judge_total)
    if n == 0:
        return np.array([])

    # 评委分归一化
    jt = np.asarray(judge_total, dtype=float)
    jt_min, jt_max = np.min(jt), np.max(jt)
    if jt_max - jt_min > 1e-9:
        judge_norm = (jt - jt_min) / (jt_max - jt_min)
    else:
        judge_norm = np.ones_like(jt) * 0.5

    prior = np.asarray(prior, dtype=float)
    prior = np.clip(prior, 1e-8, None)
    prior = prior / prior.sum()

    if minimize is None:
        return prior

    # 目标函数：L2贴近先验
    def objective(v):
        return 0.5 * np.sum((v - prior) ** 2)

    # 约束：sum(v)=1
    cons = [{"type": "eq", "fun": lambda v: np.sum(v) - 1.0}]

    # 约束：淘汰者组合得分最低
    # combined = alpha_score * judge_norm + (1 - alpha_score) * v
    elim_idx = np.where(elim_mask)[0].tolist()
    keep_idx = np.where(~elim_mask)[0].tolist()
    if len(elim_idx) > 0 and len(keep_idx) > 0:
        for e in elim_idx:
            for k in keep_idx:
                # combined_k - combined_e >= margin
                cons.append({
                    "type": "ineq",
                    "fun": lambda v, e=e, k=k: (
                        alpha_score * (judge_norm[k] - judge_norm[e]) +
                        (1 - alpha_score) * (v[k] - v[e]) - margin
                    )
                })

    bounds = [(1e-8, 1.0) for _ in range(n)]
    x0 = prior.copy()

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 200})
    if not res.success:
        return prior

    v = np.clip(res.x, 1e-8, None)
    v = v / v.sum()
    return v


# -----------------------------
# 主流程
# -----------------------------
def main():
    print("=" * 70)
    print("Q1_MAP: 约束优化求解粉丝投票")
    print("=" * 70)

    # 1) 读取清洗数据
    df_clean = pd.read_csv("dwts_cleaned.csv")
    df_long = pd.read_csv("dwts_long_format.csv")

    # season weeks
    season_weeks = df_clean.groupby("season")["weeks_participated"].max().to_dict()
    df_long["season_weeks"] = df_long["season"].map(season_weeks)

    # 真实淘汰
    truth_map = build_truth_map(df_clean)

    # 赛季规则
    max_season = int(df_long["season"].max())
    rank_seasons, pct_seasons = get_rule_sets(max_season)

    # 2) 逐周求解投票份额
    records = []
    for (season, week), g in df_long.groupby(["season", "week"]):
        season = int(season)
        week = int(week)

        # 仅保留当周有评分的选手
        g = g[g["judge_total"].notna() & (g["judge_total"] > 0)].copy()
        if len(g) == 0:
            continue

        # 先验：softmax(judge_total)
        prior = softmax(g["judge_total"].values, temp=1.0)

        # 淘汰集合
        true_list = truth_map.get((season, week), truth_map.get((season, float(week)), []))
        elim_mask = g["celebrity_name"].isin(true_list).values

        # 约束优化求解
        vote_share = solve_week_vote_share(
            judge_total=g["judge_total"].values,
            prior=prior,
            elim_mask=elim_mask,
            alpha_score=0.5,
            margin=1e-4
        )

        g["vote_share_hat"] = vote_share

        # 票池：周次递增 + 赛季强度
        BASE_TOTAL_VOTES = 1_000_000
        W = int(season_weeks.get(season, g["week"].max()))
        factor = 0.8 + 0.6 * (week - 1) / (W - 1) if W > 1 else 1.0
        season_strength = df_long[df_long["season"] == season]["judge_total"].mean() / df_long["judge_total"].mean()
        total_votes = BASE_TOTAL_VOTES * factor * season_strength

        g["total_votes_hat"] = total_votes
        g["votes_hat"] = g["vote_share_hat"] * g["total_votes_hat"]

        records.append(g)

    result_df = pd.concat(records, ignore_index=True)
    result_df = result_df.sort_values(["season", "week", "votes_hat"], ascending=[True, True, False])
    result_df.to_csv("q1_map_vote_estimates.csv", index=False)

    print(f"[输出] q1_map_vote_estimates.csv 保存完成，行数: {len(result_df)}")

    # 3) 一致性评估（按规则复原淘汰）
    eval_rows = []
    for (season, week), g in result_df.groupby(["season", "week"]):
        season = int(season)
        week = int(week)
        true_list = truth_map.get((season, week), truth_map.get((season, float(week)), []))
        true_set = set(true_list)
        k = len(true_list)

        if k == 0:
            eval_rows.append({"season": season, "week": week, "true_k": 0, "exact_match": 1, "bottom2_cover": 1})
            continue

        if season in rank_seasons:
            g = g.copy()
            g["judge_rank_w"] = g["judge_total"].rank(ascending=False, method="min")
            g["vote_rank_w"] = g["votes_hat"].rank(ascending=False, method="min")
            g["combined_rank"] = g["judge_rank_w"] + g["vote_rank_w"]
            g = g.sort_values(["combined_rank", "judge_total", "votes_hat"], ascending=[False, True, True])
        else:
            g = g.copy()
            judge_sum = g["judge_total"].sum()
            vote_sum = g["votes_hat"].sum()
            g["judge_pct"] = g["judge_total"] / judge_sum if judge_sum > 0 else 0.0
            g["vote_pct"] = g["votes_hat"] / vote_sum if vote_sum > 0 else 0.0
            g["combined_pct"] = g["judge_pct"] + g["vote_pct"]
            g = g.sort_values(["combined_pct", "judge_total", "votes_hat"], ascending=[True, True, True])

        pred_list = g["celebrity_name"].head(k).tolist()
        bottom2 = g["celebrity_name"].head(2).tolist()

        exact_match = int(set(pred_list) == true_set)
        bottom2_cover = int(len(true_set.intersection(set(bottom2))) > 0)

        eval_rows.append({
            "season": season,
            "week": week,
            "true_k": k,
            "exact_match": exact_match,
            "bottom2_cover": bottom2_cover
        })

    eval_df = pd.DataFrame(eval_rows)
    mask_elim = eval_df["true_k"] > 0
    exact_elim = eval_df.loc[mask_elim, "exact_match"].mean()
    bottom2 = eval_df.loc[mask_elim, "bottom2_cover"].mean()
    print(f"[一致性] 淘汰周精确匹配率: {exact_elim:.4f}")
    print(f"[一致性] Bottom-2覆盖率: {bottom2:.4f}")

    # 4) 不确定性分析（Conformal Prediction：不使用 Bootstrap）
    #    用“先验份额”作为弱标签构造非一致性分数并做split conformal
    eps = 1e-6
    result_df = result_df.copy()
    result_df["prior_share"] = result_df.groupby(["season", "week"])["judge_total"].transform(
        lambda s: softmax(s.values, temp=1.0)
    )

    # 按赛季划分校准集/目标集
    seasons = np.sort(result_df["season"].unique())
    rng = np.random.default_rng(2026)
    rng.shuffle(seasons)
    split = int(len(seasons) * 0.8)
    calib_seasons = set(seasons[:split])

    calib = result_df[result_df["season"].isin(calib_seasons)].copy()
    # 非一致性分数：相对偏差
    calib["score"] = np.abs(calib["vote_share_hat"] - calib["prior_share"]) / (calib["prior_share"] + eps)

    # 80%区间（alpha=0.2）
    alpha = 0.2
    qhat = np.quantile(calib["score"].values, 1 - alpha)

    # 生成区间
    result_df["share_q10"] = np.clip(result_df["vote_share_hat"] - qhat * (result_df["prior_share"] + eps), 0, 1)
    result_df["share_q90"] = np.clip(result_df["vote_share_hat"] + qhat * (result_df["prior_share"] + eps), 0, 1)
    result_df["share_q50"] = result_df["vote_share_hat"]
    # 直接使用份额区间宽度，避免小份额导致比例爆炸
    result_df["rel_ci80"] = result_df["share_q90"] - result_df["share_q10"]

    unc_df = result_df[["season", "week", "celebrity_name", "share_q10", "share_q50", "share_q90", "rel_ci80"]].copy()
    unc_df.to_csv("q1_map_uncertainty.csv", index=False)
    print(f"[输出] q1_map_uncertainty.csv 保存完成，行数: {len(unc_df)}")

    # 5) 绘图
    # 4.1 赛季级淘汰一致率
    season_summary = (
        eval_df.groupby("season")
        .apply(lambda x: pd.Series({
            "exact_elim": x.loc[x["true_k"] > 0, "exact_match"].mean() if (x["true_k"] > 0).any() else np.nan,
            "bottom2": x.loc[x["true_k"] > 0, "bottom2_cover"].mean() if (x["true_k"] > 0).any() else np.nan
        }))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.bar(season_summary["season"].astype(str), season_summary["exact_elim"],
           color=COLORS["primary"], edgecolor="navy", alpha=0.85)
    ax.axhline(y=season_summary["exact_elim"].mean(), color="red", linestyle="--", linewidth=2)
    ax.set_title("Season-level Elimination Exact Match Rate", fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Exact Match Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_season_consistency.png")
    plt.close()

    # 4.2 周级一致率
    week_summary = eval_df.groupby("week")["exact_match"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(week_summary["week"], week_summary["exact_match"], marker="o", color=COLORS["accent"])
    ax.fill_between(week_summary["week"], 0, week_summary["exact_match"], alpha=0.2, color=COLORS["accent"])
    ax.set_title("Weekly Exact Match Rate (Averaged Across Seasons)", fontweight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Exact Match Rate")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_weekly_consistency.png")
    plt.close()

    # 4.3 评分热力图（浅蓝 -> 珊瑚粉）
    pivot_data = df_long.pivot_table(values="judge_total", index="season", columns="week", aggfunc="mean")
    custom_heatmap_cmap = LinearSegmentedColormap.from_list(
        "dwts_theme", [COLORS["light"], COLORS["primary"], COLORS["secondary"], COLORS["success"]]
    )
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    sns.heatmap(pivot_data, cmap=custom_heatmap_cmap, ax=ax, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Average Judge Total Score"})
    ax.set_title("Average Judge Total Score Heatmap (All Seasons)", fontweight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Season")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_score_heatmap.png")
    plt.close()

    # 4.4 份额分布
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    ax.hist(result_df["vote_share_hat"], bins=40, edgecolor="black", color=COLORS["secondary"], alpha=0.75)
    ax.set_title("Distribution of Estimated Vote Share", fontweight="bold")
    ax.set_xlabel("Estimated Vote Share")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_vote_share_dist.png")
    plt.close()

    # 4.5 不确定性分布
    if unc_df is not None:
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
        ax.hist(unc_df["rel_ci80"].dropna(), bins=40, edgecolor="black", color=COLORS["neutral"], alpha=0.75)
        ax.set_title("Distribution of Relative Uncertainty (CI80 / Median)", fontweight="bold")
        ax.set_xlabel("Relative CI80 Width")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig5_uncertainty_distribution.png")
        plt.close()

        week_unc = unc_df.groupby("week")["rel_ci80"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=300)
        ax.plot(week_unc["week"], week_unc["mean"], marker="o", color=COLORS["secondary"])
        ax.fill_between(
            week_unc["week"], week_unc["mean"] - week_unc["std"], week_unc["mean"] + week_unc["std"],
            alpha=0.2, color=COLORS["secondary"]
        )
        ax.set_title("Uncertainty by Week (Across All Seasons)", fontweight="bold")
        ax.set_xlabel("Week")
        ax.set_ylabel("Relative CI80 Width")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig6_uncertainty_by_week.png")
        plt.close()

        # 4.6 选手不确定性排行（Top15）
        cele_unc = (
            unc_df.groupby("celebrity_name")["rel_ci80"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )
        fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=300)
        ax.bar(cele_unc.index, cele_unc.values, color=COLORS["success"], edgecolor="darkred", alpha=0.8)
        ax.set_title("Top Contestants with Highest Mean Uncertainty", fontweight="bold")
        ax.set_xlabel("Celebrity")
        ax.set_ylabel("Mean Relative CI80 Width")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig7_top_uncertainty_contestants.png")
        plt.close()

        # 4.7 不确定性热力图（示例赛季 Top12）
        example_season = int(unc_df["season"].max())
        ex_df = unc_df[unc_df["season"] == example_season].copy()
        top12 = (
            ex_df.groupby("celebrity_name")["share_q50"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .index.tolist()
        )
        heat = ex_df[ex_df["celebrity_name"].isin(top12)].pivot_table(
            index="celebrity_name", columns="week", values="rel_ci80", aggfunc="mean"
        ).fillna(0)
        fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=300)
        im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd")
        ax.set_title(f"Uncertainty Heatmap (Season {example_season}, Top12)", fontweight="bold")
        ax.set_xlabel("Week")
        ax.set_ylabel("Celebrity")
        ax.set_xticks(np.arange(heat.shape[1]))
        ax.set_xticklabels(heat.columns.tolist())
        ax.set_yticks(np.arange(heat.shape[0]))
        ax.set_yticklabels(heat.index.tolist(), fontsize=9)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Relative CI80 Width")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig8_uncertainty_heatmap.png")
        plt.close()

    print(f"[绘图] 输出目录: {OUTPUT_DIR}")

    # 6) 汇总报告（对齐 Q1.py 输出风格）
    print("=" * 70)
    print("完整分析报告")
    print("=" * 70)
    print("\n【Part 1 - 粉丝投票估算】")
    print(f"  - 预测样本数: {len(result_df)}")
    print(f"  - 平均预测票数: {result_df['votes_hat'].mean():,.0f}")
    print("  - 结果已保存至: q1_map_vote_estimates.csv")
    print("\n【Part 2 - 淘汰一致性评估】")
    print(f"  - 淘汰周精确匹配率: {exact_elim:.4f}")
    print(f"  - Bottom-2覆盖率: {bottom2:.4f}")
    if unc_df is not None:
        print("\n【Part 3 - 不确定性度量】")
        print("  - 方法: Conformal Prediction (alpha=0.2)")
        rel_vals = unc_df["rel_ci80"].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  - 平均区间宽度(CI80): {rel_vals.mean():.4f}")
        print(f"  - 中位数区间宽度(CI80): {rel_vals.median():.4f}")
        print(f"  - 区间宽度标准差: {rel_vals.std():.4f}")
    print("\n【输出文件】")
    print("  - q1_map_vote_estimates.csv (票数估算结果)")
    if unc_df is not None:
        print("  - q1_map_uncertainty.csv (不确定性结果)")
    print(f"  - {OUTPUT_DIR}/ (所有可视化图表)")


if __name__ == "__main__":
    main()

