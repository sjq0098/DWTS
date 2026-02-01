# ============================================================
# MCM 2026 Problem C - Question 4: E-SHAP-TOPSIS 新投票系统
# 基于“SHAP(Trend) + 熵权(Confidence)”的动态权重 + TOPSIS排名
# 参考 Q3 的 RankSHAP（排列重要性 + NDCG）
# ============================================================

import warnings
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

np.random.seed(2026)
RANDOM_SEED = 2026

COLORS = {
    "primary": "#7BADDF",
    "secondary": "#B581B4",
    "accent": "#EAB170",
    "success": "#DA8176",
    "neutral": "#B1A8D3",
    "light": "#BADDF3",
    "dark": "#4A5568",
    "judge": "#EAB170",
    "fan": "#7BADDF",
    "entropy": "#4ECDC4",
}

PALETTE = [
    "#BADDF3", "#C8C3E1", "#B581B4", "#B1A8D3", "#B5C3EA",
    "#F4E09B", "#EAB170", "#DA8176"
]
HEATMAP_CMAP = LinearSegmentedColormap.from_list("pastel", PALETTE)

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

OUTPUT_DIR = Path("plots/q4_new")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUTPUT_DIR = Path("outputs/q4_new")
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTROVERSIAL_SEASONS = {
    2: {"name": "Jerry Rice", "issue": "评委最低分获亚军"},
    4: {"name": "Billy Ray Cyrus", "issue": "6周评委最低分仍获第5"},
    11: {"name": "Bristol Palin", "issue": "12次评委最低分获第3"},
    27: {"name": "Bobby Bones", "issue": "评委评分持续偏低仍夺冠"},
}

# 指标定义（与Q4.md一致）
FAIRNESS_JUDGE_BOTTOM = 0.30
FAIRNESS_FAN_TOP = 0.30
ADVANCE_CUTOFF = 0.70
POPULARITY_TOP_K = 5
POPULARITY_JUDGE_FILTER = FAIRNESS_JUDGE_BOTTOM


def minmax_norm(series: pd.Series) -> pd.Series:
    s_min = series.min()
    s_max = series.max()
    if s_max - s_min == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - s_min) / (s_max - s_min)


def compute_entropy_weight(values: np.ndarray) -> float:
    """基于信息熵计算权重（越分散权重越高）"""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return 0.5
    total = values.sum()
    if total <= 0:
        p = np.ones(n) / n
    else:
        p = values / total
    p = np.clip(p, 1e-12, 1.0)
    k = 1.0 / math.log(n) if n > 1 else 1.0
    entropy = -k * np.sum(p * np.log(p))
    diff = 1.0 - entropy
    return diff


def compute_ndcg(relevance: np.ndarray, order_scores: np.ndarray, k: int = None) -> float:
    """计算NDCG（归一化折损累计增益）"""
    if k is None:
        k = len(relevance)
    order_idx = np.argsort(order_scores)[::-1]
    rel_sorted = relevance[order_idx]
    dcg = np.sum(rel_sorted[:k] / np.log2(np.arange(2, k + 2)))
    ideal_rel = np.sort(relevance)[::-1]
    idcg = np.sum(ideal_rel[:k] / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_top3_overlap(fan_rank_series: pd.Series, final_rank_series: pd.Series) -> float:
    """计算Top3重合度（粉丝Top3中有多少进入了最终Top3）
    Popularity = |Top3(Fan) ∩ Top3(Final)| / 3
    """
    fan_top3 = set(fan_rank_series.nsmallest(3).index)
    final_top3 = set(final_rank_series.nsmallest(3).index)
    if not fan_top3:
        return 0.0
    overlap = len(fan_top3.intersection(final_top3))
    return overlap / 3.0


def compute_fan_ndcg(season_scores: pd.DataFrame,
                     k: int = POPULARITY_TOP_K,
                     judge_filter: float = POPULARITY_JUDGE_FILTER) -> float:
    """
    民意性指标：在“评委认可的人群”里看粉丝支持是否被满足
    - 先按评委平均得分筛掉最低的一部分，再计算 NDCG@K
    """
    if season_scores is None or len(season_scores) == 0:
        return 0.0
    df = season_scores.copy()
    if "judge_pct_mean" in df.columns and len(df) >= 3:
        n = len(df)
        judge_rank = df["judge_pct_mean"].rank(ascending=False, method="min")
        judge_top_threshold = max(2, int((1 - judge_filter) * n))
        df = df[judge_rank <= judge_top_threshold]
    if len(df) == 0:
        return 0.0
    k = min(k, len(df))
    relevance = df["fan_share_mean"].values.astype(float)
    order_scores = df["score"].values.astype(float)
    return compute_ndcg(relevance, order_scores, k)


def compute_reversal_rate(prev_g: pd.DataFrame, curr_g: pd.DataFrame) -> float:
    """
    观赏性指标：逆转率（排名变化≥2位的比例）
    - 更符合“剧情反转”的直观观赏性
    """
    if prev_g is None or curr_g is None:
        return 0.0
    prev_rank = prev_g.set_index("celebrity_name")["score_rank"]
    curr_rank = curr_g.set_index("celebrity_name")["score_rank"]
    aligned_prev, aligned_curr = prev_rank.align(curr_rank, join="inner")
    if len(aligned_curr) == 0:
        return 0.0
    rank_change = (aligned_prev - aligned_curr).abs()
    return float((rank_change >= 2).sum() / len(aligned_curr))


class RankSHAPTrendAnalyzer:
    """参考Q3的RankSHAP，用排列重要性近似SHAP趋势"""

    def __init__(self):
        self.model = None
        self.feature_names = ["judge_norm", "fan_norm", "week_progress"]

    def fit(self, df: pd.DataFrame):
        X = df[self.feature_names].values
        y = df["weekly_rank"].values  # 排名越小越好
        if XGB_AVAILABLE:
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror",
                max_depth=4,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        self.model.fit(X, y)
        return self

    def compute_rankshap_for_group(self, g: pd.DataFrame, n_permutations: int = 10):
        if len(g) < 3:
            return {f: 0.0 for f in self.feature_names}
        X = g[self.feature_names].values
        y_rank = g["weekly_rank"].values
        # relevance: 名次越小越好 -> 转为越大越好
        max_rank = np.max(y_rank)
        relevance = (max_rank - y_rank + 1).astype(float)

        y_pred = self.model.predict(X)
        base_ndcg = compute_ndcg(relevance, -y_pred)

        rankshap_values = {}
        for feat_idx, feat_name in enumerate(self.feature_names):
            ndcg_drops = []
            for _ in range(n_permutations):
                X_perm = X.copy()
                np.random.shuffle(X_perm[:, feat_idx])
                y_perm = self.model.predict(X_perm)
                ndcg_perm = compute_ndcg(relevance, -y_perm)
                ndcg_drops.append(base_ndcg - ndcg_perm)
            rankshap_values[feat_name] = float(np.mean(ndcg_drops))

        rankshap_values["_ndcg_base"] = float(base_ndcg)
        rankshap_values["_n_samples"] = len(g)
        return rankshap_values


class Q4NewVotingSystem:
    """E-SHAP-TOPSIS 动态评价模型实现"""

    def __init__(self,
                 long_data_path="dwts_long_format.csv",
                 vote_data_path="q1_fan_vote_estimates_enhanced.csv"):
        self.long_data_path = long_data_path
        self.vote_data_path = vote_data_path
        self.df = None
        self.rankshap = RankSHAPTrendAnalyzer()

        self.weekly_records = []
        self.season_metrics = []
        self.weight_records = []
        self.controversial_results = []
        self.gamma = 0.6
        self.gamma_tuning = None

    def load_data(self):
        long_df = pd.read_csv(self.long_data_path)
        vote_df = pd.read_csv(self.vote_data_path)

        merge_cols = ["season", "week", "celebrity_name"]
        vote_cols = [c for c in ["season", "week", "celebrity_name", "votes_hat", "vote_share_hat"]
                     if c in vote_df.columns]
        self.df = long_df.merge(vote_df[vote_cols], on=merge_cols, how="left")

        if "votes_hat" not in self.df.columns:
            self.df["votes_hat"] = np.nan
        if "vote_share_hat" not in self.df.columns:
            self.df["vote_share_hat"] = np.nan
        self.df["votes_hat"] = self.df["votes_hat"].fillna(0)
        self.df["vote_share_hat"] = self.df["vote_share_hat"].fillna(0)

        weekly_sum = self.df.groupby(["season", "week"])["votes_hat"].transform("sum")
        self.df["fan_share"] = np.where(weekly_sum > 0, self.df["votes_hat"] / weekly_sum, 0)

        # 进度特征
        season_max_week = self.df.groupby("season")["week"].transform("max")
        self.df["week_progress"] = self.df["week"] / season_max_week

        return self

    def prepare_features(self):
        """计算每周归一化特征"""
        df = self.df.copy()
        df["judge_norm"] = df.groupby(["season", "week"])["judge_total"].transform(minmax_norm)
        df["fan_norm"] = df.groupby(["season", "week"])["votes_hat"].transform(minmax_norm)
        # 仅保留有周排名的数据用于SHAP训练
        self.rankshap_train_df = df[df["weekly_rank"].notna()].copy()
        self.df = df
        return self

    def run_rankshap_trend(self):
        """训练RankSHAP模型并计算每周权重"""
        self.rankshap.fit(self.rankshap_train_df)

        for (season, week), g in self.df.groupby(["season", "week"]):
            rs = self.rankshap.compute_rankshap_for_group(g)
            j_val = abs(rs.get("judge_norm", 0.0))
            f_val = abs(rs.get("fan_norm", 0.0))
            if j_val + f_val > 0:
                w_shap_j = j_val / (j_val + f_val)
            else:
                w_shap_j = 0.5

            # 熵权（基于当周分布）
            dj = compute_entropy_weight(g["judge_norm"].values)
            df_ = compute_entropy_weight(g["fan_norm"].values)
            w_ent_j = dj / (dj + df_) if (dj + df_) > 0 else 0.5

            # 融合权重
            w_j = self.gamma * w_shap_j + (1 - self.gamma) * w_ent_j
            w_f = 1 - w_j

            self.weight_records.append({
                "season": int(season),
                "week": int(week),
                "t": float(g["week_progress"].iloc[0]),
                "w_shap_judge": w_shap_j,
                "w_entropy_judge": w_ent_j,
                "w_final_judge": w_j,
                "w_final_fan": w_f,
            })

        self.weights_df = pd.DataFrame(self.weight_records)
        return self

    def rebuild_final_weights(self, gamma: float):
        """基于已计算的SHAP/熵权重重建融合权重"""
        self.gamma = float(gamma)
        if not hasattr(self, "weights_df"):
            return self
        self.weights_df = self.weights_df.copy()
        self.weights_df["w_final_judge"] = (
            self.gamma * self.weights_df["w_shap_judge"]
            + (1 - self.gamma) * self.weights_df["w_entropy_judge"]
        )
        self.weights_df["w_final_fan"] = 1 - self.weights_df["w_final_judge"]
        return self

    def _evaluate_eshap_metrics(self):
        """仅评估 E-SHAP-TOPSIS 的三指标"""
        weight_map = {(row.season, row.week): row for row in self.weights_df.itertuples()}
        fairness_total = 0
        fairness_advance = 0
        excitement_list = []
        pop_list = []

        for season, season_df in self.df.groupby("season"):
            max_week = int(season_df["week"].max())
            prev_g = None
            for week, g in season_df.groupby("week"):
                _ = week / max_week if max_week > 0 else 0
                w_j = weight_map[(season, week)].w_final_judge
                g = self._compute_week_features(g, "eshap_topsis", w_j)
                g["score_rank"] = g["score"].rank(ascending=False, method="min")
                c_total, c_adv = self._compute_fairness(g)
                fairness_total += c_total
                fairness_advance += c_adv
                if prev_g is not None:
                    reversal = compute_reversal_rate(prev_g, g)
                    excitement_list.append(reversal)
                prev_g = g

            season_scores = season_df.groupby("celebrity_name").agg(
                fan_share_mean=("fan_share", "mean")
            ).reset_index()
            weekly_scores = []
            for week, g in season_df.groupby("week"):
                w_j = weight_map[(season, week)].w_final_judge
                g = self._compute_week_features(g.copy(), "eshap_topsis", w_j)
                weekly_scores.append(g[["celebrity_name", "score", "judge_pct"]])
            combined_df = pd.concat(weekly_scores, ignore_index=True)
            combined_mean = combined_df.groupby("celebrity_name").agg(
                score=("score", "mean"),
                judge_pct_mean=("judge_pct", "mean"),
            ).reset_index()
            season_scores = season_scores.merge(combined_mean, on="celebrity_name", how="left")
            pop_list.append(compute_fan_ndcg(season_scores))

        return {
            "fairness_rate": (fairness_advance / fairness_total) if fairness_total > 0 else np.nan,
            "popularity_fan_ndcg": float(np.mean(pop_list)) if pop_list else np.nan,
            "excitement_reversal_rate": float(np.mean(excitement_list)) if excitement_list else np.nan,
        }

    def tune_gamma(self, gamma_candidates=None, objective="fairness_excitement"):
        """参数调优：在gamma范围内寻找更平衡的指标组合"""
        if gamma_candidates is None:
            gamma_candidates = np.linspace(0.2, 0.8, 7)

        records = []
        for gamma in gamma_candidates:
            self.rebuild_final_weights(gamma)
            metrics = self._evaluate_eshap_metrics()
            records.append({
                "gamma": float(gamma),
                **metrics,
            })

        df = pd.DataFrame(records)
        fairness = df["fairness_rate"].values
        excitement = df["excitement_reversal_rate"].values
        popularity = df["popularity_fan_ndcg"].values

        def norm_desc(x):
            if np.nanmax(x) - np.nanmin(x) == 0:
                return np.ones_like(x) * 0.5
            return (np.nanmax(x) - x) / (np.nanmax(x) - np.nanmin(x))

        def norm_asc(x):
            if np.nanmax(x) - np.nanmin(x) == 0:
                return np.ones_like(x) * 0.5
            return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

        fairness_score = norm_desc(fairness)
        excitement_score = norm_asc(excitement)
        popularity_score = norm_asc(popularity)

        if objective == "fairness_excitement":
            score = 0.5 * fairness_score + 0.5 * excitement_score
        elif objective == "fairness_popularity":
            score = 0.5 * fairness_score + 0.5 * popularity_score
        elif objective == "popularity_focus":
            # 优先民意性，牺牲部分公平性
            score = 0.25 * fairness_score + 0.60 * popularity_score + 0.15 * excitement_score
        else:  # balanced
            score = (0.34 * fairness_score
                     + 0.33 * popularity_score
                     + 0.33 * excitement_score)

        df["tuning_score"] = score
        best_idx = df["tuning_score"].idxmax()
        best_gamma = float(df.loc[best_idx, "gamma"])

        self.gamma_tuning = df
        self.rebuild_final_weights(best_gamma)
        return self

    def _topsis_score(self, g: pd.DataFrame, w_j: float, w_f: float):
        judge = g["judge_norm"].values
        fan = g["fan_norm"].values
        d_pos = np.sqrt(w_j * (1 - judge) ** 2 + w_f * (1 - fan) ** 2)
        d_neg = np.sqrt(w_j * (judge) ** 2 + w_f * (fan) ** 2)
        score = d_neg / (d_pos + d_neg + 1e-12)
        return score

    def _compute_week_features(self, g: pd.DataFrame, method: str, w_j: float = None):
        g = g.copy()
        n = len(g)

        g["fan_votes"] = g["votes_hat"]
        total_fan = g["fan_votes"].sum()
        g["fan_share"] = g["fan_votes"] / total_fan if total_fan > 0 else 1.0 / n

        total_judge = g["judge_total"].sum()
        g["judge_pct"] = g["judge_total"] / total_judge if total_judge > 0 else 1.0 / n

        g["judge_rank"] = g["judge_total"].rank(ascending=False, method="min")
        g["fan_rank"] = g["fan_votes"].rank(ascending=False, method="min")

        if method == "rank":
            g["combined"] = g["judge_rank"] + g["fan_rank"]
            g["score"] = -g["combined"]
        elif method == "percent":
            g["combined"] = g["judge_pct"] + g["fan_share"]
            g["score"] = g["combined"]
        elif method == "eshap_topsis":
            w_f = 1.0 - w_j
            g["combined"] = self._topsis_score(g, w_j, w_f)
            g["score"] = g["combined"]
        else:
            raise ValueError(f"Unknown method: {method}")

        g["score_norm"] = minmax_norm(g["score"])
        return g

    def _compute_fairness(self, g: pd.DataFrame):
        n = len(g)
        if n < 3:
            return 0, 0
        judge_bottom_threshold = max(2, int((1 - FAIRNESS_JUDGE_BOTTOM) * n))
        fan_top_threshold = max(2, int(FAIRNESS_FAN_TOP * n))
        judge_bottom = g["judge_rank"] > judge_bottom_threshold
        fan_top = g["fan_rank"] <= fan_top_threshold
        controversial = g[judge_bottom & fan_top]
        if len(controversial) == 0:
            g["rank_gap"] = g["judge_rank"] - g["fan_rank"]
            controversial = g[g["rank_gap"] >= 3]
        advanced = controversial[g["score_rank"] <= int(ADVANCE_CUTOFF * n)]
        return len(controversial), len(advanced)

    def simulate(self):
        methods = ["rank", "percent", "eshap_topsis"]

        weight_map = {(row.season, row.week): row for row in self.weights_df.itertuples()}

        for method in methods:
            for season, season_df in self.df.groupby("season"):
                max_week = int(season_df["week"].max())
                fairness_total = 0
                fairness_advance = 0
                excitement_list = []

                week_records = []
                prev_g = None
                for week, g in season_df.groupby("week"):
                    t = week / max_week if max_week > 0 else 0
                    if method == "eshap_topsis":
                        w_j = weight_map[(season, week)].w_final_judge
                    else:
                        w_j = None
                    g = self._compute_week_features(g, method, w_j)
                    g["score_rank"] = g["score"].rank(ascending=False, method="min")

                    c_total, c_adv = self._compute_fairness(g)
                    fairness_total += c_total
                    fairness_advance += c_adv

                    if prev_g is not None:
                        reversal = compute_reversal_rate(prev_g, g)
                        excitement_list.append(reversal)
                    prev_g = g

                    week_records.append({
                        "season": int(season),
                        "week": int(week),
                        "t": t,
                        "method": method,
                        "w_judge": w_j if method == "eshap_topsis" else np.nan,
                        "w_fan": (1 - w_j) if method == "eshap_topsis" else np.nan,
                        "n_contestants": len(g),
                        "excitement_reversal_rate": float(excitement_list[-1]) if excitement_list else np.nan,
                    })

                season_scores = season_df.groupby("celebrity_name").agg(
                    fan_share_mean=("fan_share", "mean")
                ).reset_index()

                weekly_scores = []
                for week, g in season_df.groupby("week"):
                    if method == "eshap_topsis":
                        w_j = weight_map[(season, week)].w_final_judge
                    else:
                        w_j = None
                    g = self._compute_week_features(g.copy(), method, w_j)
                weekly_scores.append(g[["celebrity_name", "score", "judge_pct"]])
                combined_df = pd.concat(weekly_scores, ignore_index=True)
                combined_mean = combined_df.groupby("celebrity_name").agg(
                    score=("score", "mean"),
                    judge_pct_mean=("judge_pct", "mean"),
                ).reset_index()
                season_scores = season_scores.merge(combined_mean, on="celebrity_name", how="left")

                popularity_ndcg = compute_fan_ndcg(season_scores)

                self.season_metrics.append({
                    "season": int(season),
                    "method": method,
                    "fairness_rate": (fairness_advance / fairness_total) if fairness_total > 0 else np.nan,
                    "popularity_fan_ndcg": popularity_ndcg,
                    "excitement_reversal_rate": float(np.mean(excitement_list)) if excitement_list else np.nan,
                })

                self.weekly_records.extend(week_records)

        return self

    def save_results(self):
        weights_df = pd.DataFrame(self.weight_records)
        weekly_df = pd.DataFrame(self.weekly_records)
        season_df = pd.DataFrame(self.season_metrics)

        weights_df.to_csv(CSV_OUTPUT_DIR / "q4_new_weights.csv", index=False)
        weekly_df.to_csv(CSV_OUTPUT_DIR / "q4_new_weekly_results.csv", index=False)
        season_df.to_csv(CSV_OUTPUT_DIR / "q4_new_season_metrics.csv", index=False)
        if isinstance(self.gamma_tuning, pd.DataFrame) and len(self.gamma_tuning) > 0:
            self.gamma_tuning.to_csv(CSV_OUTPUT_DIR / "q4_new_gamma_tuning.csv", index=False)
        if hasattr(self, "controversial_df") and len(self.controversial_df) > 0:
            self.controversial_df.to_csv(CSV_OUTPUT_DIR / "q4_new_controversial.csv", index=False)

        summary = (season_df.groupby("method")
                   .agg({
                       "fairness_rate": "mean",
                       "popularity_fan_ndcg": "mean",
                       "excitement_reversal_rate": "mean"
                   })
                   .reset_index())
        summary.to_csv(CSV_OUTPUT_DIR / "q4_new_summary.csv", index=False)
        return self

    def analyze_controversial_cases(self):
        """争议赛季专项验证：跟踪选手在不同方法下的排名轨迹"""
        weight_map = {(row.season, row.week): row for row in self.weights_df.itertuples()}
        results = []

        for season_num, info in CONTROVERSIAL_SEASONS.items():
            season_df = self.df[self.df["season"] == season_num]
            if len(season_df) == 0:
                continue

            celebrity_name = info["name"]
            matches = season_df[
                season_df["celebrity_name"].str.contains(celebrity_name, case=False, na=False)
            ]
            if len(matches) == 0:
                continue

            for method in ["rank", "percent", "eshap_topsis"]:
                for week, g in season_df.groupby("week"):
                    if method == "eshap_topsis":
                        w_j = weight_map[(season_num, week)].w_final_judge
                    else:
                        w_j = None
                    g = self._compute_week_features(g, method, w_j)
                    g["score_rank"] = g["score"].rank(ascending=False, method="min")
                    celeb_row = g[g["celebrity_name"].str.contains(celebrity_name, case=False, na=False)]
                    if len(celeb_row) == 0:
                        continue
                    results.append({
                        "season": season_num,
                        "week": int(week),
                        "method": method,
                        "celebrity": celeb_row["celebrity_name"].iloc[0],
                        "score_rank": int(celeb_row["score_rank"].iloc[0]),
                        "n_contestants": len(g),
                        "judge_rank": int(celeb_row["judge_rank"].iloc[0]),
                        "fan_rank": int(celeb_row["fan_rank"].iloc[0]),
                    })

        self.controversial_results = results
        self.controversial_df = pd.DataFrame(results)
        return self

    # ------------------ Plotting ------------------
    def plot_topsis_demo(self):
        """TOPSIS二维示意图（A/B点对比）"""
        if hasattr(self, "weights_df") and len(self.weights_df) > 0:
            w_j = float(self.weights_df["w_final_judge"].mean())
        else:
            w_j = 0.6
        w_f = 0.4
        points = {
            "A (0.8, 0.8)": (0.8, 0.8),
            "B (0.2, 1.0)": (0.2, 1.0),
        }
        ideal = (1.0, 1.0)
        nadir = (0.0, 0.0)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        ax.scatter([ideal[0]], [ideal[1]], color="green", s=120, marker="*", label="Ideal (1,1)")
        ax.scatter([nadir[0]], [nadir[1]], color="gray", s=80, marker="x", label="Nadir (0,0)")

        for label, (x, y) in points.items():
            d_pos = math.sqrt(w_j * (1 - x) ** 2 + w_f * (1 - y) ** 2)
            d_neg = math.sqrt(w_j * x ** 2 + w_f * y ** 2)
            score = d_neg / (d_pos + d_neg)
            ax.scatter([x], [y], s=100, label=f"{label}\nC={score:.2f}")
            ax.plot([x, ideal[0]], [y, ideal[1]], linestyle="--", color="#999999", linewidth=1)

        ax.set_xlabel("Judge Score (normalized)")
        ax.set_ylabel("Fan Vote (normalized)")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=8)
        ax.set_title("TOPSIS 2D Illustration (A vs B)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig4_topsis_demo.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_weights(self):
        if not hasattr(self, "weights_df"):
            return
        df = self.weights_df.copy()
        df = df.groupby("t").agg({
            "w_shap_judge": "mean",
            "w_entropy_judge": "mean",
            "w_final_judge": "mean",
        }).reset_index()

        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        ax.plot(df["t"], df["w_shap_judge"], color=COLORS["judge"], linestyle="--", label="SHAP Trend")
        ax.plot(df["t"], df["w_entropy_judge"], color=COLORS["entropy"], linestyle=":", label="Entropy Confidence")
        ax.plot(df["t"], df["w_final_judge"], color=COLORS["accent"], label="Final Weight (Judge)")
        ax.plot(df["t"], 1 - df["w_final_judge"], color=COLORS["fan"], label="Final Weight (Fan)")
        ax.set_xlabel("Season Progress (t)")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig1_weights_eshap_topsis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_method_comparison(self):
        season_df = pd.DataFrame(self.season_metrics)
        season_df["fairness_rate"] = season_df["fairness_rate"].fillna(0)
        summary = (season_df.groupby("method")
                   .agg({
                       "fairness_rate": "mean",
                       "popularity_fan_ndcg": "mean",
                       "excitement_reversal_rate": "mean"
                   })
                   .reindex(["rank", "percent", "eshap_topsis"]))
        fig = plt.figure(figsize=(12, 8), dpi=300)
        methods = summary.index.tolist()
        method_labels = ["Rank", "Percent", "E-SHAP-TOPSIS"]
        bar_colors = [COLORS["neutral"], COLORS["primary"], COLORS["accent"]]
        metrics = [
            ("fairness_rate", "Fairness\n(Controversy Advance Rate)", True),
            ("popularity_fan_ndcg", "Popularity\n(Fan NDCG@K)", False),
            ("excitement_reversal_rate", "Excitement\n(Rank Reversal Rate)", False)
        ]
        for idx, (metric, ylabel, lower_better) in enumerate(metrics):
            ax = fig.add_subplot(2, 2, idx + 1)
            values = summary[metric].values
            bars = ax.bar(method_labels, values, color=bar_colors, edgecolor="white", linewidth=1.5)
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f"{val:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlabel("Method", fontsize=9)
            best_idx = np.argmin(values) if lower_better else np.argmax(values)
            bars[best_idx].set_edgecolor("green")
            bars[best_idx].set_linewidth(3)

        ax_radar = fig.add_subplot(2, 2, 4, polar=True)
        radar_data = summary.copy()
        radar_data["fairness_rate"] = 1 - (radar_data["fairness_rate"] / radar_data["fairness_rate"].max()) \
            if radar_data["fairness_rate"].max() > 0 else 0
        radar_data["popularity_fan_ndcg"] = radar_data["popularity_fan_ndcg"] / radar_data["popularity_fan_ndcg"].max() \
            if radar_data["popularity_fan_ndcg"].max() > 0 else 0
        radar_data["excitement_reversal_rate"] = radar_data["excitement_reversal_rate"] / radar_data["excitement_reversal_rate"].max() \
            if radar_data["excitement_reversal_rate"].max() > 0 else 0
        categories = ["Fairness\n(lower=better)", "Popularity", "Excitement"]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        for method, color, label in zip(methods, bar_colors, method_labels):
            values = radar_data.loc[
                method,
                ["fairness_rate", "popularity_fan_ndcg", "excitement_reversal_rate"]
            ].values.tolist()
            values += values[:1]
            ax_radar.plot(angles, values, "o-", linewidth=2, color=color, label=label)
            ax_radar.fill(angles, values, alpha=0.15, color=color)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=9)
        ax_radar.set_ylim(0, 1.1)
        ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig2_method_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_season_heatmap(self):
        season_df = pd.DataFrame(self.season_metrics)
        season_df["fairness_rate"] = season_df["fairness_rate"].fillna(0)

        key_seasons = list(CONTROVERSIAL_SEASONS.keys())
        all_seasons = sorted(season_df["season"].unique())
        sampled_seasons = [s for s in all_seasons if s % 5 == 0]
        selected_seasons = sorted(set(key_seasons + sampled_seasons + [1, max(all_seasons)]))

        season_df_filtered = season_df[season_df["season"].isin(selected_seasons)]

        metrics = ["fairness_rate", "popularity_fan_ndcg", "excitement_reversal_rate"]
        metric_names = ["Fairness\n(↓ better)", "Popularity\n(↑ better)", "Excitement\n(↑ better)"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=300)
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            pivot = season_df_filtered.pivot(index="season", columns="method", values=metric)
            pivot = pivot.reindex(columns=["rank", "percent", "eshap_topsis"])
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap=HEATMAP_CMAP,
                        ax=axes[idx], cbar_kws={"shrink": 0.7},
                        annot_kws={"fontsize": 9})
            axes[idx].set_xlabel("Method", fontsize=11)
            axes[idx].set_ylabel("Season", fontsize=11)
            for row_idx, season in enumerate(pivot.index):
                if season in CONTROVERSIAL_SEASONS:
                    axes[idx].add_patch(plt.Rectangle((0, row_idx), 3, 1, fill=False,
                                                       edgecolor="red", linewidth=2))
            axes[idx].set_title(name, fontsize=11, fontweight="bold")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig3_season_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_controversial_cases(self):
        """争议赛季排名轨迹图"""
        if not hasattr(self, "controversial_df") or len(self.controversial_df) == 0:
            return
        seasons = self.controversial_df["season"].unique()
        n_seasons = len(seasons)
        fig, axes = plt.subplots(1, min(n_seasons, 4), figsize=(5 * min(n_seasons, 4), 5), dpi=300)
        if n_seasons == 1:
            axes = [axes]

        method_styles = {
            "rank": {"color": COLORS["neutral"], "linestyle": "-", "marker": "o", "label": "Rank"},
            "percent": {"color": COLORS["primary"], "linestyle": "--", "marker": "s", "label": "Percent"},
            "eshap_topsis": {"color": COLORS["accent"], "linestyle": "-.", "marker": "^", "label": "E-SHAP-TOPSIS"},
        }

        for idx, season in enumerate(seasons[:4]):
            ax = axes[idx]
            season_data = self.controversial_df[self.controversial_df["season"] == season]
            celeb = season_data["celebrity"].iloc[0]

            for method in ["rank", "percent", "eshap_topsis"]:
                method_data = season_data[season_data["method"] == method].sort_values("week")
                style = method_styles[method]
                ax.plot(method_data["week"], method_data["score_rank"],
                        marker=style["marker"], linestyle=style["linestyle"],
                        color=style["color"], label=style["label"],
                        linewidth=2.5, markersize=7, alpha=0.9)

            if "judge_rank" in season_data.columns:
                judge_data = season_data[season_data["method"] == "rank"].sort_values("week")
                ax.plot(judge_data["week"], judge_data["judge_rank"],
                        ":", color="gray", linewidth=1.5, alpha=0.6, label="Judge Only")

            ax.invert_yaxis()
            ax.set_xlabel("Week", fontsize=10)
            ax.set_ylabel("Combined Rank", fontsize=10)
            ax.set_title(f"S{season}: {celeb}", fontsize=11)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig5_controversial_cases.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_gamma_tuning(self):
        """图6: gamma调优曲线"""
        if not isinstance(self.gamma_tuning, pd.DataFrame) or len(self.gamma_tuning) == 0:
            return
        df = self.gamma_tuning.copy()
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        ax.plot(df["gamma"], df["fairness_rate"], "o-", color=COLORS["judge"], label="Fairness (↓)")
        ax.plot(df["gamma"], df["popularity_fan_ndcg"], "s--", color=COLORS["primary"], label="Popularity (↑)")
        ax.plot(df["gamma"], df["excitement_reversal_rate"], "^-", color=COLORS["accent"], label="Excitement (↑)")
        ax.set_xlabel("Gamma (SHAP weight)")
        ax.set_ylabel("Metric Value")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig6_gamma_tuning.png", dpi=300, bbox_inches="tight")
        plt.close()

    def generate_report(self):
        print("\n" + "=" * 70)
        print("Q4 New (E-SHAP-TOPSIS) 分析结果汇总")
        print("=" * 70)
        season_df = pd.DataFrame(self.season_metrics)
        season_df["fairness_rate"] = season_df["fairness_rate"].fillna(0)
        summary = season_df.groupby("method").agg({
            "fairness_rate": "mean",
            "popularity_fan_ndcg": "mean",
            "excitement_reversal_rate": "mean"
        }).reindex(["rank", "percent", "eshap_topsis"])

        print("\n【方法指标对比】")
        print(f"  {'方法':<15} {'公平性(↓)':<15} {'民意性(NDCG@K)(↑)':<18} {'观赏性(逆转率)(↑)':<20}")
        print("  " + "-" * 70)
        for method in ["rank", "percent", "eshap_topsis"]:
            row = summary.loc[method]
            print(f"  {method:<15} {row['fairness_rate']:.4f}          "
                  f"{row['popularity_fan_ndcg']:.4f}          {row['excitement_reversal_rate']:.4f}")

        if isinstance(self.gamma_tuning, pd.DataFrame) and len(self.gamma_tuning) > 0:
            best_row = self.gamma_tuning.sort_values("tuning_score", ascending=False).iloc[0]
            print("\n【参数调优】")
            print("  目标: fairness_excitement")
            print(f"  最佳gamma: {best_row['gamma']:.2f}")
            print(f"  对应公平性: {best_row['fairness_rate']:.4f}")
            print(f"  对应观赏性: {best_row['excitement_reversal_rate']:.4f}")

        print("\n【输出文件】")
        print(f"  CSV文件: {CSV_OUTPUT_DIR}/")
        for f in CSV_OUTPUT_DIR.glob("q4_new_*.csv"):
            print(f"    - {f.name}")
        print(f"  图表文件: {OUTPUT_DIR}/")
        for f in OUTPUT_DIR.glob("fig*.png"):
            print(f"    - {f.name}")
        print("\n" + "=" * 70)
        return self

    def run(self):
        print("=" * 70)
        print("MCM 2026 Problem C - Q4: E-SHAP-TOPSIS 新投票系统")
        print("核心：RankSHAP趋势 + 熵权修正 + TOPSIS排名")
        print("=" * 70)

        self.load_data()
        self.prepare_features()
        self.run_rankshap_trend()
        self.tune_gamma(objective="popularity_focus")
        self.simulate()
        self.analyze_controversial_cases()
        self.save_results()
        self.plot_topsis_demo()
        self.plot_weights()
        self.plot_method_comparison()
        self.plot_season_heatmap()
        self.plot_controversial_cases()
        self.plot_gamma_tuning()
        self.generate_report()
        print("\n所有分析完成！")
        return self


def main():
    Q4NewVotingSystem().run()


if __name__ == "__main__":
    main()

