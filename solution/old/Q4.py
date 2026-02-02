# ============================================================
# MCM 2026 Problem C - Question 4: 新投票系统设计
# 目标：实现动态权重投票系统，并与排名法/百分比法对比
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy.stats import spearmanr
import seaborn as sns

warnings.filterwarnings('ignore')

# -----------------------------
# 全局配置 & 可视化风格
# -----------------------------
np.random.seed(2026)
RANDOM_SEED = 2026

# 主题配色（与前几问一致）
COLORS = {
    "primary": "#7BADDF",      # 浅蓝
    "secondary": "#B581B4",    # 薰衣草紫
    "accent": "#EAB170",       # 暖橙
    "success": "#DA8176",      # 珊瑚粉
    "neutral": "#B1A8D3",      # 淡紫
    "light": "#BADDF3",        # 极浅蓝
    "dark": "#4A5568",         # 深灰
    "judge": "#EAB170",        # 评委颜色
    "fan": "#7BADDF",          # 粉丝颜色
}

# 柔和配色（与前几问一致风格）
PALETTE = [
    "#BADDF3", "#C8C3E1", "#B581B4", "#B1A8D3", "#B5C3EA",
    "#F4E09B", "#EAB170", "#DA8176"
]
HEATMAP_CMAP = LinearSegmentedColormap.from_list("pastel", PALETTE)

# 设置绘图风格
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

# 输出目录
OUTPUT_DIR = Path("plots/q4_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUTPUT_DIR = Path("outputs/q4")
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 动态权重参数（来自Q4思路）
WEIGHT_PARAMS = {
    "t0": 0.65,
    "beta": 12,
    "k": 0.4,
    "C": 0.3,
}

# 争议赛季（用于专项回测）
CONTROVERSIAL_SEASONS = {
    2: {"name": "Jerry Rice", "issue": "评委最低分获亚军"},
    4: {"name": "Billy Ray Cyrus", "issue": "6周评委最低分仍获第5"},
    11: {"name": "Bristol Palin", "issue": "12次评委最低分获第3"},
    27: {"name": "Bobby Bones", "issue": "评委评分持续偏低仍夺冠"},
}

# 敏感性分析参数范围
SENSITIVITY_PARAMS = {
    "beta": [8, 10, 12, 15, 18, 20],
    "t0": [0.55, 0.60, 0.65, 0.70, 0.75],
}




def sigmoid_weight(t: float, t0: float, beta: float, k: float, C: float) -> float:
    w_mid = 0.5
    w_range = 0.1  # 只波动 0.1 而非 0.2
    return w_mid + w_range * np.tanh(beta * (t0 - t))  # 递减但更平缓
def minmax_norm(series: pd.Series) -> pd.Series:
    """Min-Max归一化，避免除0"""
    s_min = series.min()
    s_max = series.max()
    if s_max - s_min == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - s_min) / (s_max - s_min)


class Q4VotingSystem:
    """问题四：动态权重投票系统 + 对比评估"""

    def __init__(self,
                 long_data_path: str = "dwts_long_format.csv",
                 vote_data_path: str = "q1_fan_vote_estimates_enhanced.csv"):
        self.long_data_path = long_data_path
        self.vote_data_path = vote_data_path
        self.df = None

        self.weekly_records = []
        self.season_metrics = []
        self.final_rankings = []

    def load_data(self):
        """加载并合并数据"""
        long_df = pd.read_csv(self.long_data_path)
        vote_df = pd.read_csv(self.vote_data_path)

        merge_cols = ["season", "week", "celebrity_name"]
        vote_cols = merge_cols + ["votes_hat", "vote_share_hat"]
        available_cols = [c for c in vote_cols if c in vote_df.columns]

        self.df = long_df.merge(
            vote_df[available_cols],
            on=merge_cols,
            how="left"
        )

        # 统一粉丝票字段
        if "votes_hat" not in self.df.columns:
            self.df["votes_hat"] = np.nan
        if "vote_share_hat" not in self.df.columns:
            self.df["vote_share_hat"] = np.nan

        # 处理缺失
        self.df["votes_hat"] = self.df["votes_hat"].fillna(0)
        self.df["vote_share_hat"] = self.df["vote_share_hat"].fillna(0)

        # 计算每周粉丝票占比（用于民意性指标）
        weekly_sum = self.df.groupby(["season", "week"])["votes_hat"].transform("sum")
        self.df["fan_share"] = np.where(
            weekly_sum > 0,
            self.df["votes_hat"] / weekly_sum,
            0
        )

        return self

    def _compute_week_features(self, g: pd.DataFrame, method: str, wj: float = None):
        """计算单周特征与组合分数"""
        g = g.copy()
        n = len(g)

        # 粉丝票数（优先使用votes_hat）
        g["fan_votes"] = g["votes_hat"]
        if g["fan_votes"].sum() == 0:
            g["fan_votes"] = g["vote_share_hat"]

        total_fan = g["fan_votes"].sum()
        g["fan_share"] = g["fan_votes"] / total_fan if total_fan > 0 else 1.0 / n

        total_judge = g["judge_total"].sum()
        g["judge_pct"] = g["judge_total"] / total_judge if total_judge > 0 else 1.0 / n

        # 排名（1为最好）
        g["judge_rank"] = g["judge_total"].rank(ascending=False, method="min")
        g["fan_rank"] = g["fan_votes"].rank(ascending=False, method="min")

        if method == "rank":
            g["combined"] = g["judge_rank"] + g["fan_rank"]
            g["score"] = -g["combined"]  # 越大越好
        elif method == "percent":
            g["combined"] = g["judge_pct"] + g["fan_share"]
            g["score"] = g["combined"]
        elif method == "dynamic":
            wj = float(wj)
            wf = 1.0 - wj
            g["judge_norm"] = minmax_norm(g["judge_total"])
            g["fan_norm"] = minmax_norm(g["fan_votes"])
            g["combined"] = wj * g["judge_norm"] + wf * g["fan_norm"]
            g["score"] = g["combined"]
        else:
            raise ValueError(f"Unknown method: {method}")

        # 统一尺度用于观赏性指标
        g["score_norm"] = minmax_norm(g["score"])
        # 统一量纲：都转换为 [0,1] 的百分位排名
        g["score"] = g["score"].rank(pct=True)  # 添加这一行统一量纲
        return g

    def _select_eliminated(self, g: pd.DataFrame, k: int, use_bottom2: bool,
                           revival_enabled: bool, revival_used: bool):
        """选择淘汰者（可选底部二选一与复活权）"""
        g_work = g.copy()
        eliminated = []
        revived = None

        # 复活权：评委第一但处于组合排名末20%
        if revival_enabled and not revival_used and len(g_work) >= 3:
            g_work["combined_rank"] = g_work["score"].rank(ascending=False, method="min")
            bottom_cut = int(np.ceil(0.8 * len(g_work)))
            judge_top = g_work.sort_values("judge_rank").iloc[0]["celebrity_name"]
            judge_top_rank = g_work.loc[
                g_work["celebrity_name"] == judge_top, "combined_rank"
            ].iloc[0]

            if judge_top_rank > bottom_cut:
                revived = judge_top
                revival_used = True
                g_work = g_work[g_work["celebrity_name"] != judge_top]

        if use_bottom2:
            for _ in range(k):
                if len(g_work) == 0:
                    break
                bottom2 = g_work.nsmallest(2, "score")
                if len(bottom2) == 1:
                    elim = bottom2.iloc[0]["celebrity_name"]
                else:
                    # 评委保留技术更好者（judge_total高者）
                    elim = bottom2.sort_values("judge_total").iloc[0]["celebrity_name"]
                eliminated.append(elim)
                g_work = g_work[g_work["celebrity_name"] != elim]
        else:
            worst = g_work.nsmallest(k, "score")
            eliminated = worst["celebrity_name"].tolist()

        return eliminated, revived, revival_used

    def _compute_fairness(self, g: pd.DataFrame, eliminated: list):
        """公平性：争议选手晋级率（单周）
        
        争议选手定义：评委排名在末40%，但粉丝排名在前40%
        这比原来的70%/30%更宽松，能捕获更多争议案例
        """
        n = len(g)
        if n < 3:
            return 0, 0
        
        # 使用更宽松的阈值
        judge_bottom_threshold = max(2, int(0.6 * n))  # 至少2人
        fan_top_threshold = max(2, int(0.4 * n))  # 至少2人
        
        judge_bottom = g["judge_rank"] > judge_bottom_threshold
        fan_top = g["fan_rank"] <= fan_top_threshold
        controversial = g[judge_bottom & fan_top]["celebrity_name"].tolist()
        
        # 备选方案：评委与粉丝排名差距 >= 3
        if len(controversial) == 0:
            g["rank_gap"] = g["judge_rank"] - g["fan_rank"]
            controversial = g[g["rank_gap"] >= 3]["celebrity_name"].tolist()

        advanced = [c for c in controversial if c not in eliminated]
        return len(controversial), len(advanced)

    def simulate(self):
        """模拟三种方法并输出指标"""
        methods = ["rank", "percent", "dynamic"]

        for method in methods:
            for season, season_df in self.df.groupby("season"):
                revival_used = False
                week_records = []

                max_week = int(season_df["week"].max())
                fairness_total = 0
                fairness_advance = 0
                excitement_list = []

                for week, g in season_df.groupby("week"):
                    g = g.copy()
                    t = week / max_week if max_week > 0 else 0
                    wj = sigmoid_weight(t, **WEIGHT_PARAMS) if method == "dynamic" else None

                    g = self._compute_week_features(g, method, wj)

                    # 计算淘汰数量（与真实赛季一致）
                    if "elimination_week" in g.columns:
                        k = int((g["elimination_week"] == week).sum())
                    else:
                        k = 1
                    if k == 0 and week < max_week:
                        k = 1

                    use_bottom2 = (method == "dynamic")
                    revival_enabled = (method == "dynamic")

                    eliminated, revived, revival_used = self._select_eliminated(
                        g, k, use_bottom2, revival_enabled, revival_used
                    )

                    # 公平性统计
                    c_total, c_adv = self._compute_fairness(g, eliminated)
                    fairness_total += c_total
                    fairness_advance += c_adv

                    # 观赏性：组合分数方差
                    excitement_list.append(float(g["score_norm"].var(ddof=0)))

                    week_records.append({
                        "season": int(season),
                        "week": int(week),
                        "method": method,
                        "w_judge": wj if method == "dynamic" else np.nan,
                        "w_fan": (1 - wj) if method == "dynamic" else np.nan,
                        "n_contestants": len(g),
                        "eliminated": ",".join(eliminated) if eliminated else "",
                        "revived": revived if revived else "",
                        "excitement_var": float(g["score_norm"].var(ddof=0))
                    })

                # 赛季级民意性：粉丝排名 vs 组合得分排名
                season_scores = season_df.groupby("celebrity_name").agg(
                    fan_share_mean=("fan_share", "mean")
                ).reset_index()

                # 计算方法对应的平均组合得分
                weekly_scores = []
                for week, g in season_df.groupby("week"):
                    t = week / max_week if max_week > 0 else 0
                    wj = sigmoid_weight(t, **WEIGHT_PARAMS) if method == "dynamic" else None
                    g = self._compute_week_features(g, method, wj)
                    weekly_scores.append(g[["celebrity_name", "score"]])

                combined_df = pd.concat(weekly_scores, ignore_index=True)
                combined_mean = combined_df.groupby("celebrity_name")["score"].mean().reset_index()

                season_scores = season_scores.merge(combined_mean, on="celebrity_name", how="left")

                fan_rank = season_scores["fan_share_mean"].rank(ascending=False, method="min")
                final_rank = season_scores["score"].rank(ascending=False, method="min")

                pop_corr = spearmanr(fan_rank, final_rank).correlation

                self.season_metrics.append({
                    "season": int(season),
                    "method": method,
                    "fairness_rate": (fairness_advance / fairness_total) if fairness_total > 0 else np.nan,
                    "popularity_spearman": pop_corr,
                    "excitement_avg_var": float(np.mean(excitement_list)) if excitement_list else np.nan,
                    "revival_used": revival_used if method == "dynamic" else False
                })

                self.weekly_records.extend(week_records)

                # 保存最终排名
                season_scores["fan_rank"] = fan_rank
                season_scores["final_rank"] = final_rank
                season_scores["season"] = int(season)
                season_scores["method"] = method
                self.final_rankings.append(season_scores)

        return self

    def save_results(self):
        """保存CSV结果"""
        weekly_df = pd.DataFrame(self.weekly_records)
        season_df = pd.DataFrame(self.season_metrics)
        ranking_df = pd.concat(self.final_rankings, ignore_index=True)

        weekly_df.to_csv(CSV_OUTPUT_DIR / "q4_weekly_results.csv", index=False)
        season_df.to_csv(CSV_OUTPUT_DIR / "q4_season_metrics.csv", index=False)
        ranking_df.to_csv(CSV_OUTPUT_DIR / "q4_final_rankings.csv", index=False)

        # 汇总
        season_df_filled = season_df.copy()
        season_df_filled["fairness_rate"] = season_df_filled["fairness_rate"].fillna(0)

        summary = (season_df_filled.groupby("method")
                   .agg({
                       "fairness_rate": "mean",
                       "popularity_spearman": "mean",
                       "excitement_avg_var": "mean"
                   })
                   .reset_index())
        summary.to_csv(CSV_OUTPUT_DIR / "q4_summary.csv", index=False)
        return self

    def plot_weights(self):
        """图1：动态权重曲线"""
        t = np.linspace(0, 1, 100)
        wj = sigmoid_weight(t, **WEIGHT_PARAMS)
        wf = 1 - wj

        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        ax.plot(t, wj, color=COLORS["judge"], label="Judge Weight")
        ax.plot(t, wf, color=COLORS["fan"], label="Fan Weight")
        ax.set_xlabel("Season Progress (t)")
        ax.set_ylabel("Weight")
        ax.legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig1_weight_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Fig1] 动态权重曲线图已保存")

    def plot_metric_comparison(self):
        """图2：三指标对比（柱状图 + 雷达图）"""
        season_df = pd.DataFrame(self.season_metrics)
        season_df["fairness_rate"] = season_df["fairness_rate"].fillna(0)
        summary = (season_df.groupby("method")
                   .agg({
                       "fairness_rate": "mean",
                       "popularity_spearman": "mean",
                       "excitement_avg_var": "mean"
                   })
                   .reindex(["rank", "percent", "dynamic"]))

        # 创建2x2布局：3个柱状图 + 1个雷达图
        fig = plt.figure(figsize=(14, 10), dpi=300)
        
        # 柱状图部分
        methods = summary.index.tolist()
        method_labels = ["Rank", "Percent", "Dynamic"]
        bar_colors = [COLORS["neutral"], COLORS["primary"], COLORS["accent"]]
        
        metrics = [
            ("fairness_rate", "Fairness\n(Controversy Advance Rate)", True),  # 越低越好
            ("popularity_spearman", "Popularity\n(Fan-Final Rank Spearman)", False),  # 越高越好
            ("excitement_avg_var", "Excitement\n(Score Variance)", False)  # 越高越好
        ]
        
        for idx, (metric, ylabel, lower_better) in enumerate(metrics):
            ax = fig.add_subplot(2, 2, idx + 1)
            values = summary[metric].values
            bars = ax.bar(method_labels, values, color=bar_colors, edgecolor="white", linewidth=1.5)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 5), textcoords="offset points",
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xlabel("Method", fontsize=10)
            
            # 标记最优方法
            if lower_better:
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)
            bars[best_idx].set_edgecolor("green")
            bars[best_idx].set_linewidth(3)
        
        # 雷达图
        ax_radar = fig.add_subplot(2, 2, 4, polar=True)
        
        # 归一化数据（0-1范围）
        radar_data = summary.copy()
        # Fairness 反转（越低越好 -> 越高越好）
        radar_data["fairness_rate"] = 1 - (radar_data["fairness_rate"] / radar_data["fairness_rate"].max()) if radar_data["fairness_rate"].max() > 0 else 0
        radar_data["popularity_spearman"] = radar_data["popularity_spearman"] / radar_data["popularity_spearman"].max() if radar_data["popularity_spearman"].max() > 0 else 0
        radar_data["excitement_avg_var"] = radar_data["excitement_avg_var"] / radar_data["excitement_avg_var"].max() if radar_data["excitement_avg_var"].max() > 0 else 0
        
        categories = ["Fairness\n(lower=better)", "Popularity", "Excitement"]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合
        
        for method, color, label in zip(methods, bar_colors, method_labels):
            values = radar_data.loc[method].values.tolist()
            values += values[:1]
            ax_radar.plot(angles, values, 'o-', linewidth=2, color=color, label=label)
            ax_radar.fill(angles, values, alpha=0.15, color=color)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=10)
        ax_radar.set_ylim(0, 1.1)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax_radar.set_title("Multi-dimensional Comparison", fontsize=12, fontweight='bold', y=1.1)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig2_metric_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Fig2] 三指标对比图（含雷达图）已保存")

    def sensitivity_analysis(self):
        """参数敏感性分析"""
        print("\n[Step 5] 参数敏感性分析...")
        self.sensitivity_results = []

        base_params = WEIGHT_PARAMS.copy()

        # 对 beta 进行敏感性分析
        for beta_val in SENSITIVITY_PARAMS["beta"]:
            params = base_params.copy()
            params["beta"] = beta_val

            metrics = self._simulate_with_params(params)
            metrics["param"] = "beta"
            metrics["value"] = beta_val
            self.sensitivity_results.append(metrics)

        # 对 t0 进行敏感性分析
        for t0_val in SENSITIVITY_PARAMS["t0"]:
            params = base_params.copy()
            params["t0"] = t0_val

            metrics = self._simulate_with_params(params)
            metrics["param"] = "t0"
            metrics["value"] = t0_val
            self.sensitivity_results.append(metrics)

        self.sensitivity_df = pd.DataFrame(self.sensitivity_results)
        self.sensitivity_df.to_csv(CSV_OUTPUT_DIR / "q4_sensitivity.csv", index=False)
        print(f"  敏感性分析完成，共 {len(self.sensitivity_results)} 组参数")
        return self

    def _simulate_with_params(self, params: dict) -> dict:
        """使用指定参数模拟动态方法并返回汇总指标"""
        fairness_list = []
        pop_list = []
        excitement_list = []

        for season, season_df in self.df.groupby("season"):
            max_week = int(season_df["week"].max())
            fairness_total = 0
            fairness_advance = 0
            week_excitement = []

            for week, g in season_df.groupby("week"):
                g = g.copy()
                t = week / max_week if max_week > 0 else 0
                wj = sigmoid_weight(t, **params)

                g = self._compute_week_features(g, "dynamic", wj)

                # 淘汰
                if "elimination_week" in g.columns:
                    k = int((g["elimination_week"] == week).sum())
                else:
                    k = 1
                if k == 0 and week < max_week:
                    k = 1

                eliminated, _, _ = self._select_eliminated(g, k, True, True, False)
                c_total, c_adv = self._compute_fairness(g, eliminated)
                fairness_total += c_total
                fairness_advance += c_adv
                week_excitement.append(float(g["score_norm"].var(ddof=0)))

            if fairness_total > 0:
                fairness_list.append(fairness_advance / fairness_total)
            excitement_list.extend(week_excitement)

            # 民意性
            season_scores = season_df.groupby("celebrity_name").agg(
                fan_share_mean=("fan_share", "mean")
            ).reset_index()
            weekly_scores = []
            for week, g in season_df.groupby("week"):
                t = week / max_week if max_week > 0 else 0
                wj = sigmoid_weight(t, **params)
                g = self._compute_week_features(g, "dynamic", wj)
                weekly_scores.append(g[["celebrity_name", "score"]])
            combined_df = pd.concat(weekly_scores, ignore_index=True)
            combined_mean = combined_df.groupby("celebrity_name")["score"].mean().reset_index()
            season_scores = season_scores.merge(combined_mean, on="celebrity_name", how="left")
            fan_rank = season_scores["fan_share_mean"].rank(ascending=False, method="min")
            final_rank = season_scores["score"].rank(ascending=False, method="min")
            pop_list.append(spearmanr(fan_rank, final_rank).correlation)

        return {
            "fairness_rate": np.mean(fairness_list) if fairness_list else np.nan,
            "popularity_spearman": np.mean(pop_list) if pop_list else np.nan,
            "excitement_avg_var": np.mean(excitement_list) if excitement_list else np.nan,
        }

    def analyze_controversial_cases(self):
        """争议案例专项分析"""
        print("\n[Step 6] 争议案例分析...")
        self.controversial_results = []

        for season_num, info in CONTROVERSIAL_SEASONS.items():
            season_df = self.df[self.df["season"] == season_num]
            if len(season_df) == 0:
                print(f"  [!] Season {season_num} 数据不存在，跳过")
                continue

            celebrity_name = info["name"]
            # 尝试匹配选手名（模糊匹配）
            matches = season_df[season_df["celebrity_name"].str.contains(celebrity_name, case=False, na=False)]
            if len(matches) == 0:
                # 全部选手
                matches = season_df

            for method in ["rank", "percent", "dynamic"]:
                max_week = int(season_df["week"].max())
                rank_by_week = []

                for week, g in season_df.groupby("week"):
                    t = week / max_week if max_week > 0 else 0
                    wj = sigmoid_weight(t, **WEIGHT_PARAMS) if method == "dynamic" else None
                    g = self._compute_week_features(g, method, wj)
                    g["score_rank"] = g["score"].rank(ascending=False, method="min")

                    # 找到争议选手的排名
                    celeb_row = g[g["celebrity_name"].str.contains(celebrity_name, case=False, na=False)]
                    if len(celeb_row) > 0:
                        rank_by_week.append({
                            "season": season_num,
                            "week": int(week),
                            "method": method,
                            "celebrity": celebrity_name,
                            "score_rank": int(celeb_row["score_rank"].iloc[0]),
                            "n_contestants": len(g),
                            "judge_rank": int(celeb_row["judge_rank"].iloc[0]),
                            "fan_rank": int(celeb_row["fan_rank"].iloc[0])
                        })

                self.controversial_results.extend(rank_by_week)

        self.controversial_df = pd.DataFrame(self.controversial_results)
        if len(self.controversial_df) > 0:
            self.controversial_df.to_csv(CSV_OUTPUT_DIR / "q4_controversial.csv", index=False)
            print(f"  争议案例分析完成，共 {len(self.controversial_df)} 条记录")
        return self

    def plot_phase_weights(self):
        """图3: 分阶段权重曲线（标注筛选期/过渡期/决战期）"""
        t = np.linspace(0, 1, 100)
        wj = sigmoid_weight(t, **WEIGHT_PARAMS)
        wf = 1 - wj

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        ax.plot(t, wj, color=COLORS["judge"], linewidth=2.5, label="Judge Weight")
        ax.plot(t, wf, color=COLORS["fan"], linewidth=2.5, label="Fan Weight")

        # 阶段分界线
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7)
        ax.axvline(x=0.8, color="gray", linestyle="--", alpha=0.7)

        # 阶段标注
        ax.fill_betweenx([0, 1], 0, 0.5, alpha=0.1, color=COLORS["judge"])
        ax.fill_betweenx([0, 1], 0.5, 0.8, alpha=0.1, color=COLORS["neutral"])
        ax.fill_betweenx([0, 1], 0.8, 1.0, alpha=0.1, color=COLORS["fan"])

        ax.text(0.25, 0.95, "Screening\nPhase", ha="center", va="top", fontsize=10,
                color=COLORS["judge"], fontweight="bold", transform=ax.transAxes)
        ax.text(0.55, 0.95, "Transition\nPhase", ha="center", va="top", fontsize=10,
                color=COLORS["dark"], fontweight="bold", transform=ax.transAxes)
        ax.text(0.85, 0.95, "Finals\nPhase", ha="center", va="top", fontsize=10,
                color=COLORS["fan"], fontweight="bold", transform=ax.transAxes)

        ax.set_xlabel("Season Progress (t)", fontsize=11)
        ax.set_ylabel("Weight", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="center right")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig3_phase_weights.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Fig3] 分阶段权重曲线图已保存")

    def plot_season_heatmap(self):
        """图4: 赛季级指标热力图（选取代表性赛季）"""
        season_df = pd.DataFrame(self.season_metrics)
        season_df["fairness_rate"] = season_df["fairness_rate"].fillna(0)

        # 选取代表性赛季：争议赛季 + 每5季抽样
        key_seasons = list(CONTROVERSIAL_SEASONS.keys())  # [2, 4, 11, 27]
        all_seasons = sorted(season_df["season"].unique())
        sampled_seasons = [s for s in all_seasons if s % 5 == 0]  # 5, 10, 15, 20, 25, 30
        selected_seasons = sorted(set(key_seasons + sampled_seasons + [1, max(all_seasons)]))
        
        season_df_filtered = season_df[season_df["season"].isin(selected_seasons)]

        # 透视表
        metrics = ["fairness_rate", "popularity_spearman", "excitement_avg_var"]
        metric_names = ["Fairness\n(↓ better)", "Popularity\n(↑ better)", "Excitement\n(↑ better)"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=300)

        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            pivot = season_df_filtered.pivot(index="season", columns="method", values=metric)
            pivot = pivot.reindex(columns=["rank", "percent", "dynamic"])

            # 使用更大的字体
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap=HEATMAP_CMAP,
                       ax=axes[idx], cbar_kws={"shrink": 0.7},
                       annot_kws={"fontsize": 11, "fontweight": "bold"})
            # MCM图表不显示标题
            axes[idx].set_xlabel("Method", fontsize=12)
            axes[idx].set_ylabel("Season", fontsize=12)
            axes[idx].tick_params(axis='both', labelsize=11)
            
            # 标注争议赛季
            for row_idx, season in enumerate(pivot.index):
                if season in CONTROVERSIAL_SEASONS:
                    axes[idx].add_patch(plt.Rectangle((0, row_idx), 3, 1, fill=False, 
                                                       edgecolor='red', linewidth=2))

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig4_season_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Fig4] 赛季级热力图已保存（选取{len(selected_seasons)}个代表赛季）")

    def plot_controversial_cases(self):
        """图5: 争议案例排名轨迹（改进版）"""
        if not hasattr(self, "controversial_df") or len(self.controversial_df) == 0:
            print("  [Fig5] 无争议案例数据，跳过")
            return

        seasons = self.controversial_df["season"].unique()
        n_seasons = len(seasons)

        fig, axes = plt.subplots(1, min(n_seasons, 4), figsize=(5 * min(n_seasons, 4), 5), dpi=300)
        if n_seasons == 1:
            axes = [axes]

        # 使用不同线型区分方法
        method_styles = {
            "rank": {"color": COLORS["neutral"], "linestyle": "-", "marker": "o", "label": "Rank"},
            "percent": {"color": COLORS["primary"], "linestyle": "--", "marker": "s", "label": "Percent"},
            "dynamic": {"color": COLORS["accent"], "linestyle": "-.", "marker": "^", "label": "Dynamic"}
        }

        for idx, season in enumerate(seasons[:4]):
            ax = axes[idx]
            season_data = self.controversial_df[self.controversial_df["season"] == season]
            celeb = season_data["celebrity"].iloc[0]
            info = CONTROVERSIAL_SEASONS.get(season, {})

            for method in ["rank", "percent", "dynamic"]:
                method_data = season_data[season_data["method"] == method].sort_values("week")
                style = method_styles[method]
                ax.plot(method_data["week"], method_data["score_rank"],
                       marker=style["marker"], linestyle=style["linestyle"],
                       color=style["color"], label=style["label"], 
                       linewidth=2.5, markersize=8, alpha=0.9)
            
            # 添加评委排名参考线
            if "judge_rank" in season_data.columns:
                judge_data = season_data[season_data["method"] == "rank"].sort_values("week")
                ax.plot(judge_data["week"], judge_data["judge_rank"],
                       ":", color="gray", linewidth=1.5, alpha=0.6, label="Judge Only")

            ax.invert_yaxis()  # 排名1在上
            ax.set_xlabel("Week", fontsize=11)
            ax.set_ylabel("Combined Rank", fontsize=11)
            # MCM图表不显示标题
            ax.legend(fontsize=9, loc="best")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig5_controversial_cases.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Fig5] 争议案例分析图已保存")

    def plot_sensitivity(self):
        """图6: 参数敏感性分析"""
        if not hasattr(self, "sensitivity_df") or len(self.sensitivity_df) == 0:
            print("  [Fig6] 无敏感性分析数据，跳过")
            return

        fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=300)
        metrics = ["fairness_rate", "popularity_spearman", "excitement_avg_var"]
        metric_names = ["Fairness Rate", "Popularity (Spearman)", "Excitement (Var)"]

        # Beta 敏感性
        beta_df = self.sensitivity_df[self.sensitivity_df["param"] == "beta"]
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[0, idx]
            ax.plot(beta_df["value"], beta_df[metric], "o-", color=COLORS["primary"], linewidth=2)
            ax.axvline(x=WEIGHT_PARAMS["beta"], color="red", linestyle="--", alpha=0.7, label="Default")
            ax.set_xlabel("β (steepness)", fontsize=10)
            ax.set_ylabel(name, fontsize=10)
            ax.set_title(f"{name} vs β", fontsize=11)

        # t0 敏感性
        t0_df = self.sensitivity_df[self.sensitivity_df["param"] == "t0"]
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[1, idx]
            ax.plot(t0_df["value"], t0_df[metric], "o-", color=COLORS["accent"], linewidth=2)
            ax.axvline(x=WEIGHT_PARAMS["t0"], color="red", linestyle="--", alpha=0.7, label="Default")
            ax.set_xlabel("t₀ (transition point)", fontsize=10)
            ax.set_ylabel(name, fontsize=10)
            ax.set_title(f"{name} vs t₀", fontsize=11)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig6_sensitivity.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Fig6] 敏感性分析图已保存")

    def generate_report(self):
        """生成详细汇总报告"""
        print("\n" + "=" * 70)
        print("Q4 分析结果汇总")
        print("=" * 70)

        season_df = pd.DataFrame(self.season_metrics)
        fairness_missing = season_df["fairness_rate"].isna().all()
        season_df["fairness_rate"] = season_df["fairness_rate"].fillna(0)

        summary = season_df.groupby("method").agg({
            "fairness_rate": "mean",
            "popularity_spearman": "mean",
            "excitement_avg_var": "mean"
        }).reindex(["rank", "percent", "dynamic"])

        print("\n【三种方法指标对比】")
        print(f"  {'方法':<12} {'公平性(↓)':<15} {'民意性(↑)':<15} {'观赏性(↑)':<15}")
        print("  " + "-" * 55)
        for method in ["rank", "percent", "dynamic"]:
            row = summary.loc[method]
            print(f"  {method:<12} {row['fairness_rate']:.4f}          "
                  f"{row['popularity_spearman']:.4f}          {row['excitement_avg_var']:.4f}")

        # 改进幅度
        print("\n【动态方法相对改进】")
        for metric, direction, name in [("fairness_rate", -1, "公平性"),
                                         ("popularity_spearman", 1, "民意性"),
                                         ("excitement_avg_var", 1, "观赏性")]:
            rank_val = summary.loc["rank", metric]
            percent_val = summary.loc["percent", metric]
            dynamic_val = summary.loc["dynamic", metric]
            baseline = (rank_val + percent_val) / 2

            if baseline == 0:
                improvement = 0
            elif direction == -1:
                improvement = (baseline - dynamic_val) / baseline * 100
            else:
                improvement = (dynamic_val - baseline) / baseline * 100

            print(f"  {name}: {improvement:+.1f}% (相对于rank/percent均值)")

        if fairness_missing:
            print("\n【提示】公平性=0表示本数据中未识别到争议选手（按top30%/bottom30%定义）")

        # 争议赛季
        if hasattr(self, "controversial_df") and len(self.controversial_df) > 0:
            print("\n【争议赛季分析】")
            for season in self.controversial_df["season"].unique():
                info = CONTROVERSIAL_SEASONS.get(season, {})
                season_data = self.controversial_df[self.controversial_df["season"] == season]
                avg_ranks = season_data.groupby("method")["score_rank"].mean()
                print(f"  S{season} ({info.get('name', 'Unknown')}):")
                print(f"    Rank法平均排名: {avg_ranks.get('rank', 'N/A'):.1f}")
                print(f"    Percent法平均排名: {avg_ranks.get('percent', 'N/A'):.1f}")
                print(f"    Dynamic法平均排名: {avg_ranks.get('dynamic', 'N/A'):.1f}")

        # 敏感性分析
        if hasattr(self, "sensitivity_df") and len(self.sensitivity_df) > 0:
            print("\n【参数敏感性分析结论】")
            # 计算指标变动范围
            for param in ["beta", "t0"]:
                param_df = self.sensitivity_df[self.sensitivity_df["param"] == param]
                for metric in ["fairness_rate", "popularity_spearman"]:
                    min_val = param_df[metric].min()
                    max_val = param_df[metric].max()
                    range_pct = (max_val - min_val) / ((max_val + min_val) / 2) * 100 if (max_val + min_val) > 0 else 0
                    status = "稳健 ✓" if abs(range_pct) <= 15 else "敏感 ⚠"
                    print(f"  {param} 对 {metric}: 变动 {range_pct:.1f}% {status}")

        print("\n【输出文件】")
        print(f"  CSV文件: {CSV_OUTPUT_DIR}/")
        for f in CSV_OUTPUT_DIR.glob("q4_*.csv"):
            print(f"    - {f.name}")
        print(f"  图表文件: {OUTPUT_DIR}/")
        for f in OUTPUT_DIR.glob("fig*.png"):
            print(f"    - {f.name}")

        print("\n" + "=" * 70)
        return self

    def run(self):
        """执行全流程"""
        print("=" * 70)
        print("MCM 2026 Problem C - Question 4: 新投票系统设计")
        print("=" * 70)

        print("\n[Step 1] 加载数据...")
        self.load_data()

        print("\n[Step 2] 模拟三种投票方法...")
        self.simulate()

        print("\n[Step 3] 保存结果...")
        self.save_results()

        print("\n[Step 4] 生成可视化...")
        self.plot_weights()
        self.plot_metric_comparison()
        self.plot_phase_weights()
        self.plot_season_heatmap()

        self.sensitivity_analysis()
        self.analyze_controversial_cases()
        self.plot_controversial_cases()
        self.plot_sensitivity()

        self.generate_report()

        print(f"\n所有分析完成！")
        return self


def main():
    Q4VotingSystem().run()


if __name__ == "__main__":
    main()

