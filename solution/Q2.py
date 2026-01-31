# ============================================================
# MCM 2026 Problem C - Question 2: 投票组合方法对比与争议分析
# 包含三个小问：
#   Part 1: 跨赛季方法对比（排名法 vs 百分比法）
#   Part 2: 争议案例分析（反事实推演）
#   Part 3: 方法推荐与底部二选一规则评估
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import kendalltau

warnings.filterwarnings('ignore')

# -----------------------------
# 全局配置 & 可视化风格
# -----------------------------
np.random.seed(2026)
RANDOM_SEED = 2026

# 主题配色
COLORS = {
    "primary": "#7BADDF",      # 浅蓝
    "secondary": "#B581B4",    # 薰衣草紫
    "accent": "#EAB170",       # 暖橙
    "success": "#DA8176",      # 珊瑚粉
    "neutral": "#B1A8D3",      # 淡紫
    "light": "#BADDF3",        # 极浅蓝
    "dark": "#4A5568",         # 深灰
    "rank": "#3182CE",         # 排名法颜色
    "percent": "#E53E3E"       # 百分比法颜色
}

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

# 创建输出目录
OUTPUT_DIR = Path("plots/q2_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 赛制规则配置
MAX_SEASON = 34
RANK_SEASONS = set([1, 2]) | set(range(28, MAX_SEASON + 1))
PCT_SEASONS = set(range(3, 28))
TWIST_START_SEASON = 28

# 争议选手配置（题目明确给出的案例）
CONTROVERSY_CASES = {
    2: {"name": "Jerry Rice", "final_place": 2, "description": "5周最低分获亚军"},
    4: {"name": "Billy Ray Cyrus", "final_place": 5, "description": "6周最低分排名第5"},
    11: {"name": "Bristol Palin", "final_place": 3, "description": "12次最低分排名第3"},
    27: {"name": "Bobby Bones", "final_place": 1, "description": "持续低分夺冠"},
}


# ============================================================
# PART 1: 两种组合方法的核心实现
# ============================================================
class VotingMethodAnalyzer:
    """投票组合方法分析器"""
    
    def __init__(self, vote_estimates_path: str = "q1_fan_vote_estimates.csv",
                 cleaned_data_path: str = "dwts_cleaned.csv"):
        self.vote_path = vote_estimates_path
        self.cleaned_path = cleaned_data_path
        self.df = None
        self.cleaned_df = None
        self.weekly_results = None
        self.method_comparison = None
        self.conflict_summary = None
        
    def load_data(self):
        """加载数据"""
        self.df = pd.read_csv(self.vote_path)
        self.cleaned_df = pd.read_csv(self.cleaned_path)
        
        # 确保数据类型正确
        self.df["season"] = self.df["season"].astype(int)
        self.df["week"] = self.df["week"].astype(int)
        
        # 补充淘汰信息
        if "elimination_week" not in self.df.columns:
            elim_map = self.cleaned_df.set_index(
                ["season", "celebrity_name"]
            )["elimination_week"].to_dict()
            self.df["elimination_week"] = self.df.apply(
                lambda r: elim_map.get((r["season"], r["celebrity_name"]), np.nan),
                axis=1
            )
        
        return self
    
    def compute_rank_method(self, g: pd.DataFrame) -> pd.DataFrame:
        """排名法计算
        
        C^Rank = R^J + R^F (得分越低排名越优)
        """
        gg = g.copy()
        n = len(gg)
        
        # 计算评委得分排名（分数高排名低，即更好）
        gg["judge_rank"] = gg["judge_total"].rank(ascending=False, method="min")
        # 计算粉丝票数排名（票数高排名低，即更好）
        gg["vote_rank"] = gg["votes_hat"].rank(ascending=False, method="min")
        # 组合排名（数值越低越好）
        gg["combined_rank"] = gg["judge_rank"] + gg["vote_rank"]
        # 最终排序（combined_rank最高的被淘汰）
        gg["rank_order"] = gg["combined_rank"].rank(ascending=True, method="min")
        
        return gg
    
    def compute_percent_method(self, g: pd.DataFrame) -> pd.DataFrame:
        """百分比法计算
        
        C^Percent = P^J + P^F (得分越高排名越优)
        """
        gg = g.copy()
        
        # 计算评委得分百分比
        judge_sum = gg["judge_total"].sum()
        gg["judge_pct"] = gg["judge_total"] / judge_sum if judge_sum > 0 else 0.0
        
        # 计算粉丝票数百分比
        vote_sum = gg["votes_hat"].sum()
        gg["vote_pct"] = gg["votes_hat"] / vote_sum if vote_sum > 0 else 0.0
        
        # 组合得分（数值越高越好）
        gg["combined_pct"] = gg["judge_pct"] + gg["vote_pct"]
        # 最终排序（combined_pct最低的被淘汰）
        gg["pct_order"] = gg["combined_pct"].rank(ascending=False, method="min")
        
        return gg
    
    def compute_ndcg(self, relevance: np.ndarray, order_scores: np.ndarray, k: int = None) -> float:
        """计算NDCG（归一化折损累计增益）

        relevance: 真实相关性分数（越高越好）
        order_scores: 用于排序的分数（越高越排前）
        """
        if k is None:
            k = len(relevance)

        # 预测排序下的DCG
        order_idx = np.argsort(order_scores)[::-1]
        rel_sorted = relevance[order_idx]
        dcg = np.sum(rel_sorted[:k] / np.log2(np.arange(2, k + 2)))

        # 理想排序下的IDCG
        ideal_rel = np.sort(relevance)[::-1]
        idcg = np.sum(ideal_rel[:k] / np.log2(np.arange(2, k + 2)))

        return dcg / idcg if idcg > 0 else 0.0
    
    def compute_rankshap(self, g: pd.DataFrame, method: str = "rank") -> dict:
        """计算RankSHAP归因值（简化版）

        采用固定“相关性”作为参考（评委+粉丝的标准化之和），
        比较不同排序策略带来的NDCG差异，得到J/F贡献。
        """
        n = len(g)
        if n < 2:
            return {"phi_J": 0.5, "phi_F": 0.5, "bias": 0.5, "ndcg": 0.0}

        gg = g.copy()

        # 固定相关性：评委与粉丝标准化加和
        judge_z = (gg["judge_total"] - gg["judge_total"].mean()) / (gg["judge_total"].std() + 1e-8)
        vote_z = (gg["votes_hat"] - gg["votes_hat"].mean()) / (gg["votes_hat"].std() + 1e-8)
        relevance = (judge_z + vote_z).values

        v_empty = 0.0

        if method == "rank":
            judge_rank = gg["judge_total"].rank(ascending=False, method="min").values
            vote_rank = gg["votes_hat"].rank(ascending=False, method="min").values

            score_J = -judge_rank
            score_F = -vote_rank
            score_JF = -(judge_rank + vote_rank)

        else:  # percent
            judge_pct = gg["judge_total"] / gg["judge_total"].sum()
            vote_pct = gg["votes_hat"] / gg["votes_hat"].sum()

            score_J = judge_pct.values
            score_F = vote_pct.values
            score_JF = (judge_pct + vote_pct).values

        v_J = self.compute_ndcg(relevance, score_J)
        v_F = self.compute_ndcg(relevance, score_F)
        v_JF = self.compute_ndcg(relevance, score_JF)

        # Shapley值计算（2特征简化版）
        phi_J = 0.5 * (v_J - v_empty) + 0.5 * (v_JF - v_F)
        phi_F = 0.5 * (v_F - v_empty) + 0.5 * (v_JF - v_J)

        # 偏向性指标
        total_phi = abs(phi_J) + abs(phi_F)
        bias = abs(phi_F) / total_phi if total_phi > 0 else 0.5

        return {
            "phi_J": phi_J,
            "phi_F": phi_F,
            "bias": bias,
            "ndcg": v_JF
        }
    
    def analyze_all_weeks(self):
        """分析所有周次的两种方法结果"""
        results = []
        
        for (season, week), g in self.df.groupby(["season", "week"]):
            if len(g) < 2:
                continue
            
            # 原始规则
            original_rule = "rank" if season in RANK_SEASONS else "percent"
            
            # 计算两种方法
            g_rank = self.compute_rank_method(g)
            g_pct = self.compute_percent_method(g)
            
            # 计算RankSHAP
            shap_rank = self.compute_rankshap(g, "rank")
            shap_pct = self.compute_rankshap(g, "percent")
            
            # 获取实际淘汰者
            true_elim = g[g["elimination_week"] == week]["celebrity_name"].tolist()
            k = len(true_elim)
            
            # 预测淘汰者（按combined排序取最后k个）
            if k > 0:
                # 排名法：combined_rank最高的被淘汰
                rank_elim = g_rank.nlargest(k, "combined_rank")["celebrity_name"].tolist()
                # 百分比法：combined_pct最低的被淘汰
                pct_elim = g_pct.nsmallest(k, "combined_pct")["celebrity_name"].tolist()
                
                # 底部2人
                rank_bottom2 = g_rank.nlargest(2, "combined_rank")["celebrity_name"].tolist()
                pct_bottom2 = g_pct.nsmallest(2, "combined_pct")["celebrity_name"].tolist()
            else:
                rank_elim = []
                pct_elim = []
                rank_bottom2 = []
                pct_bottom2 = []
            
            # 方法冲突判断
            method_conflict = set(rank_elim) != set(pct_elim) if k > 0 else False
            
            # 肯德尔相关系数
            merged = g_rank.merge(
                g_pct[["celebrity_name", "pct_order"]], 
                on="celebrity_name"
            )
            if len(merged) >= 2:
                tau, _ = kendalltau(merged["rank_order"], merged["pct_order"])
            else:
                tau = np.nan
            
            results.append({
                "season": int(season),
                "week": int(week),
                "n_contestants": len(g),
                "original_rule": original_rule,
                "true_k": k,
                "true_eliminated": true_elim,
                "rank_eliminated": rank_elim,
                "pct_eliminated": pct_elim,
                "rank_bottom2": rank_bottom2,
                "pct_bottom2": pct_bottom2,
                "method_conflict": method_conflict,
                "kendall_tau": tau,
                "rank_bias": shap_rank["bias"],
                "pct_bias": shap_pct["bias"],
                "rank_ndcg": shap_rank["ndcg"],
                "pct_ndcg": shap_pct["ndcg"],
                "rank_phi_J": shap_rank["phi_J"],
                "rank_phi_F": shap_rank["phi_F"],
                "pct_phi_J": shap_pct["phi_J"],
                "pct_phi_F": shap_pct["phi_F"],
            })
        
        self.weekly_results = pd.DataFrame(results)
        return self
    
    def compute_conflict_rate(self):
        """计算规则冲突率"""
        elim_weeks = self.weekly_results[self.weekly_results["true_k"] > 0]
        
        # 总体冲突率
        total_conflict_rate = elim_weeks["method_conflict"].mean()
        
        # 按赛季分组冲突率
        season_conflict = elim_weeks.groupby("season").agg({
            "method_conflict": ["sum", "count", "mean"]
        }).reset_index()
        season_conflict.columns = ["season", "conflict_count", "total_weeks", "conflict_rate"]
        
        # 按原始规则分组
        rule_conflict = elim_weeks.groupby("original_rule").agg({
            "method_conflict": ["sum", "count", "mean"]
        }).reset_index()
        rule_conflict.columns = ["original_rule", "conflict_count", "total_weeks", "conflict_rate"]
        
        self.conflict_summary = {
            "total_conflict_rate": total_conflict_rate,
            "total_conflict_count": int(elim_weeks["method_conflict"].sum()),
            "total_elim_weeks": len(elim_weeks),
            "season_conflict": season_conflict,
            "rule_conflict": rule_conflict
        }
        
        return self
    
    def compute_method_comparison(self):
        """计算两种方法的整体对比指标"""
        elim_weeks = self.weekly_results[self.weekly_results["true_k"] > 0].copy()
        
        # 一致性评估
        elim_weeks["rank_correct"] = elim_weeks.apply(
            lambda r: set(r["rank_eliminated"]) == set(r["true_eliminated"]), axis=1
        )
        elim_weeks["pct_correct"] = elim_weeks.apply(
            lambda r: set(r["pct_eliminated"]) == set(r["true_eliminated"]), axis=1
        )
        
        # Bottom-2覆盖率
        elim_weeks["rank_bottom2_cover"] = elim_weeks.apply(
            lambda r: len(set(r["rank_bottom2"]) & set(r["true_eliminated"])) > 0 
                      if r["true_k"] > 0 else False, axis=1
        )
        elim_weeks["pct_bottom2_cover"] = elim_weeks.apply(
            lambda r: len(set(r["pct_bottom2"]) & set(r["true_eliminated"])) > 0 
                      if r["true_k"] > 0 else False, axis=1
        )
        
        self.method_comparison = {
            "rank_accuracy": elim_weeks["rank_correct"].mean(),
            "pct_accuracy": elim_weeks["pct_correct"].mean(),
            "rank_bottom2_coverage": elim_weeks["rank_bottom2_cover"].mean(),
            "pct_bottom2_coverage": elim_weeks["pct_bottom2_cover"].mean(),
            "avg_kendall_tau": elim_weeks["kendall_tau"].mean(),
            "avg_rank_bias": elim_weeks["rank_bias"].mean(),
            "avg_pct_bias": elim_weeks["pct_bias"].mean(),
            "avg_rank_ndcg": elim_weeks["rank_ndcg"].mean(),
            "avg_pct_ndcg": elim_weeks["pct_ndcg"].mean(),
        }
        
        # 按原始规则分组的准确率
        for rule in ["rank", "percent"]:
            rule_data = elim_weeks[elim_weeks["original_rule"] == rule]
            if len(rule_data) > 0:
                self.method_comparison[f"{rule}_rule_rank_acc"] = rule_data["rank_correct"].mean()
                self.method_comparison[f"{rule}_rule_pct_acc"] = rule_data["pct_correct"].mean()
        
        return self
    
    def run(self):
        """运行完整分析"""
        return (self.load_data()
                .analyze_all_weeks()
                .compute_conflict_rate()
                .compute_method_comparison())


# ============================================================
# PART 2: 争议案例分析
# ============================================================
class ControversyAnalyzer:
    """争议案例分析器"""
    
    def __init__(self, method_analyzer: VotingMethodAnalyzer):
        self.analyzer = method_analyzer
        self.df = method_analyzer.df
        self.weekly_results = method_analyzer.weekly_results
        self.controversy_results = {}
        self.counterfactual_results = []
        
    def identify_controversy_cases(self) -> pd.DataFrame:
        """识别争议选手（评委排名显著低于最终排名）"""
        controversy_list = []
        
        for (season, name), g in self.df.groupby(["season", "celebrity_name"]):
            if len(g) < 3:
                continue
            
            # 获取最终排名
            placement = g["placement"].iloc[0]
            if pd.isna(placement):
                continue
            
            # 计算每周评委排名
            weeks_lowest = 0
            weeks_bottom2 = 0
            total_weeks = len(g)
            
            for week, wg in g.groupby("week"):
                week_data = self.df[(self.df["season"] == season) & (self.df["week"] == week)]
                if len(week_data) < 2:
                    continue
                
                judge_rank = week_data["judge_total"].rank(ascending=True, method="min")
                celeb_rank = judge_rank[week_data["celebrity_name"] == name].values
                
                if len(celeb_rank) > 0:
                    if celeb_rank[0] == 1:  # 最低分
                        weeks_lowest += 1
                    if celeb_rank[0] <= 2:  # 倒数第2
                        weeks_bottom2 += 1
            
            # 争议条件：至少3周在倒数前2，且最终排名在前5
            if weeks_bottom2 >= 3 and placement <= 5:
                controversy_list.append({
                    "season": int(season),
                    "celebrity_name": name,
                    "placement": int(placement),
                    "weeks_lowest": weeks_lowest,
                    "weeks_bottom2": weeks_bottom2,
                    "total_weeks": total_weeks,
                    "controversy_score": weeks_bottom2 / total_weeks * (6 - placement)
                })
        
        return pd.DataFrame(controversy_list).sort_values("controversy_score", ascending=False)
    
    def analyze_specific_case(self, season: int, celebrity_name: str) -> dict:
        """分析特定争议案例的详细情况"""
        case_data = self.df[(self.df["season"] == season) & 
                            (self.df["celebrity_name"] == celebrity_name)]
        
        if len(case_data) == 0:
            return None
        
        result = {
            "season": season,
            "celebrity_name": celebrity_name,
            "placement": int(case_data["placement"].iloc[0]),
            "weeks": [],
            "counterfactual": {}
        }
        
        # 分析每周表现
        for _, row in case_data.iterrows():
            week = int(row["week"])
            week_data = self.df[(self.df["season"] == season) & (self.df["week"] == week)]
            
            # 评委排名
            judge_ranks = week_data["judge_total"].rank(ascending=True, method="min")
            celeb_judge_rank = judge_ranks[week_data["celebrity_name"] == celebrity_name].values[0]
            
            # 粉丝票数排名
            vote_ranks = week_data["votes_hat"].rank(ascending=False, method="min")
            celeb_vote_rank = vote_ranks[week_data["celebrity_name"] == celebrity_name].values[0]
            
            # 两种方法下的组合排名
            g_rank = self.analyzer.compute_rank_method(week_data)
            g_pct = self.analyzer.compute_percent_method(week_data)
            
            rank_order = g_rank[g_rank["celebrity_name"] == celebrity_name]["rank_order"].values[0]
            pct_order = g_pct[g_pct["celebrity_name"] == celebrity_name]["pct_order"].values[0]
            
            result["weeks"].append({
                "week": week,
                "judge_total": row["judge_total"],
                "votes_hat": row["votes_hat"],
                "judge_rank": int(celeb_judge_rank),
                "vote_rank": int(celeb_vote_rank),
                "rank_method_order": int(rank_order),
                "pct_method_order": int(pct_order),
                "n_contestants": len(week_data)
            })
        
        # 反事实分析：如果使用不同方法会怎样
        original_rule = "rank" if season in RANK_SEASONS else "percent"
        alt_rule = "percent" if original_rule == "rank" else "rank"
        
        # 统计在替代规则下的表现
        weeks_would_be_bottom = 0
        weeks_would_survive = 0
        
        for week_info in result["weeks"]:
            alt_order = week_info["pct_method_order"] if original_rule == "rank" else week_info["rank_method_order"]
            if alt_order >= week_info["n_contestants"] - 1:
                weeks_would_be_bottom += 1
            else:
                weeks_would_survive += 1
        
        result["counterfactual"] = {
            "original_rule": original_rule,
            "alternative_rule": alt_rule,
            "weeks_would_be_bottom": weeks_would_be_bottom,
            "weeks_would_survive": weeks_would_survive,
            "survival_rate_change": (weeks_would_survive - len(result["weeks"])) / len(result["weeks"])
        }
        
        return result
    
    def run_counterfactual_simulation(self):
        """运行全季节反事实模拟"""
        for (season, week), g in self.df.groupby(["season", "week"]):
            if len(g) < 2:
                continue
            
            original_rule = "rank" if season in RANK_SEASONS else "percent"
            
            # 获取实际淘汰者
            true_elim = g[g["elimination_week"] == week]["celebrity_name"].tolist()
            k = len(true_elim)
            
            if k == 0:
                continue
            
            # 计算两种方法的淘汰结果
            g_rank = self.analyzer.compute_rank_method(g)
            g_pct = self.analyzer.compute_percent_method(g)
            
            rank_elim = g_rank.nlargest(k, "combined_rank")["celebrity_name"].tolist()
            pct_elim = g_pct.nsmallest(k, "combined_pct")["celebrity_name"].tolist()
            
            # 底部2人（用于twist规则）
            rank_bottom2 = g_rank.nlargest(2, "combined_rank")
            pct_bottom2 = g_pct.nsmallest(2, "combined_pct")
            
            # 模拟twist规则（评委从底部2人中选择淘汰）
            def simulate_twist(bottom2_df):
                if len(bottom2_df) < 2:
                    return bottom2_df["celebrity_name"].iloc[0] if len(bottom2_df) > 0 else None
                # 评委倾向于保留技术更好的
                scores = bottom2_df["judge_total"].values
                names = bottom2_df["celebrity_name"].values
                # 评分低的被淘汰
                return names[0] if scores[0] < scores[1] else names[1]
            
            rank_twist_elim = simulate_twist(rank_bottom2)
            pct_twist_elim = simulate_twist(pct_bottom2)
            
            self.counterfactual_results.append({
                "season": int(season),
                "week": int(week),
                "n_contestants": len(g),
                "original_rule": original_rule,
                "true_eliminated": true_elim,
                "rank_eliminated": rank_elim,
                "pct_eliminated": pct_elim,
                "rank_twist_eliminated": rank_twist_elim,
                "pct_twist_eliminated": pct_twist_elim,
                "rank_matches_true": set(rank_elim) == set(true_elim),
                "pct_matches_true": set(pct_elim) == set(true_elim),
                "methods_agree": set(rank_elim) == set(pct_elim),
                "twist_changes_rank": rank_twist_elim not in rank_elim if rank_twist_elim else False,
                "twist_changes_pct": pct_twist_elim not in pct_elim if pct_twist_elim else False,
            })
        
        self.counterfactual_results = pd.DataFrame(self.counterfactual_results)
        return self
    
    def analyze_controversy_cases(self):
        """分析预定义的争议案例"""
        for season, case_info in CONTROVERSY_CASES.items():
            name = case_info["name"]
            result = self.analyze_specific_case(season, name)
            
            if result:
                result["expected_place"] = case_info["final_place"]
                result["description"] = case_info["description"]
                self.controversy_results[f"S{season}_{name}"] = result
        
        return self
    
    def run(self):
        """运行完整分析"""
        return (self.run_counterfactual_simulation()
                .analyze_controversy_cases())


# ============================================================
# PART 3: 底部二选一规则评估
# ============================================================
class TwistRuleEvaluator:
    """底部二选一规则评估器"""
    
    def __init__(self, controversy_analyzer: ControversyAnalyzer):
        self.analyzer = controversy_analyzer
        self.method_analyzer = controversy_analyzer.analyzer
        self.df = self.method_analyzer.df
        self.twist_impact = None
        self.recommendation = None
        
    def evaluate_twist_impact(self):
        """评估底部二选一规则的影响"""
        cf_results = self.analyzer.counterfactual_results
        
        if cf_results is None or len(cf_results) == 0:
            return self
        
        # 基本统计
        total_weeks = len(cf_results)
        
        # 计算twist规则的影响
        twist_changes_outcome = (
            cf_results["twist_changes_rank"].sum() + 
            cf_results["twist_changes_pct"].sum()
        ) / (2 * total_weeks)
        
        # 按赛季分析twist影响
        season_impact = cf_results.groupby("season").agg({
            "rank_matches_true": "mean",
            "pct_matches_true": "mean",
            "methods_agree": "mean",
            "twist_changes_rank": "mean",
            "twist_changes_pct": "mean"
        }).reset_index()
        
        # 争议选手是否会被twist规则保护
        controversy_protection = {}
        for key, case in self.analyzer.controversy_results.items():
            if "weeks" not in case:
                continue
            
            protected_weeks = 0
            total_weeks_case = len(case["weeks"])
            
            for week_info in case["weeks"]:
                season = case["season"]
                week = week_info["week"]
                
                # 检查该周在counterfactual结果中
                cf_week = cf_results[
                    (cf_results["season"] == season) & 
                    (cf_results["week"] == week)
                ]
                
                if len(cf_week) > 0:
                    row = cf_week.iloc[0]
                    # 如果twist规则改变了淘汰结果
                    if row["twist_changes_rank"] or row["twist_changes_pct"]:
                        protected_weeks += 1
            
            controversy_protection[key] = {
                "protected_weeks": protected_weeks,
                "total_weeks": total_weeks_case,
                "protection_rate": protected_weeks / total_weeks_case if total_weeks_case > 0 else 0
            }
        
        self.twist_impact = {
            "overall_change_rate": twist_changes_outcome,
            "season_impact": season_impact,
            "controversy_protection": controversy_protection
        }
        
        return self
    
    def generate_recommendation(self):
        """生成方法推荐"""
        mc = self.method_analyzer.method_comparison
        cs = self.method_analyzer.conflict_summary
        
        # 基于分析结果的推荐逻辑
        rank_score = 0
        pct_score = 0
        
        # 准确性比较
        if mc["rank_accuracy"] > mc["pct_accuracy"]:
            rank_score += 1
        else:
            pct_score += 1
        
        # 偏向性比较（越接近0.5越平衡）
        rank_balance = abs(mc["avg_rank_bias"] - 0.5)
        pct_balance = abs(mc["avg_pct_bias"] - 0.5)
        
        if rank_balance < pct_balance:
            rank_score += 1
        else:
            pct_score += 1
        
        # NDCG比较
        if mc["avg_rank_ndcg"] > mc["avg_pct_ndcg"]:
            rank_score += 1
        else:
            pct_score += 1
        
        # Twist规则评估
        twist_recommended = False
        if self.twist_impact:
            # 如果twist规则能有效改变争议结果
            avg_protection = np.mean([
                v["protection_rate"] 
                for v in self.twist_impact["controversy_protection"].values()
            ]) if self.twist_impact["controversy_protection"] else 0
            
            twist_recommended = avg_protection > 0.1  # 10%以上的保护率
        
        self.recommendation = {
            "recommended_method": "rank" if rank_score > pct_score else "percent",
            "rank_score": rank_score,
            "pct_score": pct_score,
            "twist_recommended": twist_recommended,
            "rationale": {
                "accuracy_winner": "rank" if mc["rank_accuracy"] > mc["pct_accuracy"] else "percent",
                "balance_winner": "rank" if rank_balance < pct_balance else "percent",
                "ndcg_winner": "rank" if mc["avg_rank_ndcg"] > mc["avg_pct_ndcg"] else "percent",
            }
        }
        
        return self
    
    def run(self):
        """运行完整评估"""
        return (self.evaluate_twist_impact()
                .generate_recommendation())


# ============================================================
# PART 4: 可视化
# ============================================================
class Q2Visualizer:
    """问题二可视化器"""
    
    def __init__(self, method_analyzer: VotingMethodAnalyzer,
                 controversy_analyzer: ControversyAnalyzer,
                 twist_evaluator: TwistRuleEvaluator):
        self.method_analyzer = method_analyzer
        self.controversy_analyzer = controversy_analyzer
        self.twist_evaluator = twist_evaluator
        
    def plot_conflict_rate_by_season(self):
        """图1: 按赛季的规则冲突率"""
        cs = self.method_analyzer.conflict_summary
        season_data = cs["season_conflict"]
        
        fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
        
        bars = ax.bar(
            season_data["season"].astype(str),
            season_data["conflict_rate"],
            color=[COLORS["rank"] if s in RANK_SEASONS else COLORS["percent"] 
                   for s in season_data["season"]],
            edgecolor="white",
            alpha=0.85
        )
        
        ax.axhline(y=cs["total_conflict_rate"], color="red", linestyle="--",
                   linewidth=2, label=f'Average: {cs["total_conflict_rate"]:.3f}')
        
        ax.set_xlabel("Season", fontsize=12)
        ax.set_ylabel("Conflict Rate", fontsize=12)
        ax.set_title("Method Conflict Rate by Season (Rank vs Percent)", fontsize=14)
        ax.legend()
        
        # 添加规则说明
        ax.axvspan(-0.5, 1.5, alpha=0.1, color=COLORS["rank"], label="Rank Rule")
        ax.axvspan(1.5, 26.5, alpha=0.1, color=COLORS["percent"])
        ax.axvspan(26.5, 33.5, alpha=0.1, color=COLORS["rank"])
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig1_conflict_rate_by_season.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_bias_comparison(self):
        """图2: 两种方法的偏向性对比"""
        wr = self.method_analyzer.weekly_results
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        
        # 左图：偏向性分布
        ax1 = axes[0]
        ax1.hist(wr["rank_bias"], bins=30, alpha=0.6, label="Rank Method", 
                 color=COLORS["rank"], edgecolor="white")
        ax1.hist(wr["pct_bias"], bins=30, alpha=0.6, label="Percent Method",
                 color=COLORS["percent"], edgecolor="white")
        ax1.axvline(x=0.5, color="black", linestyle="--", linewidth=2, label="Balanced (0.5)")
        ax1.set_xlabel("Fan Bias (higher = more fan-favored)", fontsize=11)
        ax1.set_ylabel("Count", fontsize=11)
        ax1.set_title("Distribution of Fan Bias by Method", fontsize=12)
        ax1.legend()
        
        # 右图：按赛季的平均偏向性
        ax2 = axes[1]
        season_bias = wr.groupby("season").agg({
            "rank_bias": "mean",
            "pct_bias": "mean"
        }).reset_index()
        
        x = np.arange(len(season_bias))
        width = 0.35
        
        ax2.bar(x - width/2, season_bias["rank_bias"], width, label="Rank Method",
                color=COLORS["rank"], alpha=0.8)
        ax2.bar(x + width/2, season_bias["pct_bias"], width, label="Percent Method",
                color=COLORS["percent"], alpha=0.8)
        ax2.axhline(y=0.5, color="black", linestyle="--", linewidth=1.5)
        
        ax2.set_xlabel("Season", fontsize=11)
        ax2.set_ylabel("Average Fan Bias", fontsize=11)
        ax2.set_title("Average Fan Bias by Season", fontsize=12)
        ax2.set_xticks(x[::3])
        ax2.set_xticklabels(season_bias["season"].values[::3])
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig2_bias_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_accuracy_comparison(self):
        """图3: 两种方法的准确性对比"""
        mc = self.method_analyzer.method_comparison
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        
        # 左图：整体准确性
        ax1 = axes[0]
        metrics = ["Accuracy", "Bottom-2 Coverage"]
        rank_vals = [mc["rank_accuracy"], mc["rank_bottom2_coverage"]]
        pct_vals = [mc["pct_accuracy"], mc["pct_bottom2_coverage"]]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, rank_vals, width, label="Rank Method",
                color=COLORS["rank"], alpha=0.85)
        ax1.bar(x + width/2, pct_vals, width, label="Percent Method",
                color=COLORS["percent"], alpha=0.85)
        
        ax1.set_ylabel("Rate", fontsize=11)
        ax1.set_title("Overall Method Comparison", fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for i, (rv, pv) in enumerate(zip(rank_vals, pct_vals)):
            ax1.text(i - width/2, rv + 0.02, f"{rv:.3f}", ha="center", fontsize=9)
            ax1.text(i + width/2, pv + 0.02, f"{pv:.3f}", ha="center", fontsize=9)
        
        # 右图：NDCG和Kendall Tau
        ax2 = axes[1]
        metrics2 = ["NDCG", "Kendall Tau"]
        rank_vals2 = [mc["avg_rank_ndcg"], mc["avg_kendall_tau"]]
        pct_vals2 = [mc["avg_pct_ndcg"], mc["avg_kendall_tau"]]
        
        x2 = np.arange(len(metrics2))
        
        ax2.bar(x2 - width/2, rank_vals2, width, label="Rank Method",
                color=COLORS["rank"], alpha=0.85)
        ax2.bar(x2 + width/2, pct_vals2, width, label="Percent Method",
                color=COLORS["percent"], alpha=0.85)
        
        ax2.set_ylabel("Score", fontsize=11)
        ax2.set_title("Ranking Quality Metrics", fontsize=12)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(metrics2)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig3_accuracy_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_controversy_analysis(self):
        """图4: 争议案例分析"""
        cases = self.controversy_analyzer.controversy_results
        
        if not cases:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
        
        case_keys = list(cases.keys())[:4]  # 最多展示4个案例
        
        for idx, key in enumerate(case_keys):
            ax = axes[idx // 2, idx % 2]
            case = cases[key]
            
            weeks = [w["week"] for w in case["weeks"]]
            judge_ranks = [w["judge_rank"] for w in case["weeks"]]
            vote_ranks = [w["vote_rank"] for w in case["weeks"]]
            rank_orders = [w["rank_method_order"] for w in case["weeks"]]
            pct_orders = [w["pct_method_order"] for w in case["weeks"]]
            
            ax.plot(weeks, judge_ranks, "o-", label="Judge Rank", 
                    color=COLORS["accent"], linewidth=2, markersize=8)
            ax.plot(weeks, vote_ranks, "s-", label="Fan Vote Rank",
                    color=COLORS["primary"], linewidth=2, markersize=8)
            ax.plot(weeks, rank_orders, "^--", label="Rank Method Order",
                    color=COLORS["rank"], linewidth=1.5, markersize=6, alpha=0.7)
            ax.plot(weeks, pct_orders, "v--", label="Percent Method Order",
                    color=COLORS["percent"], linewidth=1.5, markersize=6, alpha=0.7)
            
            ax.set_xlabel("Week", fontsize=10)
            ax.set_ylabel("Rank (lower = better)", fontsize=10)
            ax.set_title(f'{case["celebrity_name"]} (S{case["season"]}) - Place: {case["placement"]}',
                        fontsize=11)
            ax.legend(loc="upper right", fontsize=8)
            ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig4_controversy_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_counterfactual_summary(self):
        """图5: 反事实模拟结果"""
        cf = self.controversy_analyzer.counterfactual_results
        
        if cf is None or len(cf) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        
        # 左图：方法一致率按赛季
        ax1 = axes[0]
        season_agree = cf.groupby("season")["methods_agree"].mean().reset_index()
        
        colors = [COLORS["rank"] if s in RANK_SEASONS else COLORS["percent"] 
                  for s in season_agree["season"]]
        
        ax1.bar(season_agree["season"].astype(str), season_agree["methods_agree"],
                color=colors, edgecolor="white", alpha=0.85)
        ax1.axhline(y=cf["methods_agree"].mean(), color="red", linestyle="--",
                    linewidth=2, label=f'Average: {cf["methods_agree"].mean():.3f}')
        
        ax1.set_xlabel("Season", fontsize=11)
        ax1.set_ylabel("Agreement Rate", fontsize=11)
        ax1.set_title("Method Agreement Rate by Season", fontsize=12)
        ax1.legend()
        plt.sca(ax1)
        plt.xticks(rotation=45, ha="right")
        
        # 右图：Twist规则影响
        ax2 = axes[1]
        twist_data = cf.groupby("original_rule").agg({
            "twist_changes_rank": "mean",
            "twist_changes_pct": "mean"
        }).reset_index()
        
        x = np.arange(len(twist_data))
        width = 0.35
        
        ax2.bar(x - width/2, twist_data["twist_changes_rank"], width,
                label="Twist on Rank", color=COLORS["rank"], alpha=0.85)
        ax2.bar(x + width/2, twist_data["twist_changes_pct"], width,
                label="Twist on Percent", color=COLORS["percent"], alpha=0.85)
        
        ax2.set_xlabel("Original Rule", fontsize=11)
        ax2.set_ylabel("Outcome Change Rate", fontsize=11)
        ax2.set_title("Twist Rule Impact by Original Method", fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(twist_data["original_rule"])
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig5_counterfactual_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_recommendation_summary(self):
        """图6: 方法推荐总结"""
        rec = self.twist_evaluator.recommendation
        mc = self.method_analyzer.method_comparison
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        
        # 创建雷达图数据
        categories = ["Accuracy", "Balance\n(1-|bias-0.5|)", "NDCG", "Bottom-2\nCoverage"]
        
        rank_vals = [
            mc["rank_accuracy"],
            1 - abs(mc["avg_rank_bias"] - 0.5),
            mc["avg_rank_ndcg"],
            mc["rank_bottom2_coverage"]
        ]
        
        pct_vals = [
            mc["pct_accuracy"],
            1 - abs(mc["avg_pct_bias"] - 0.5),
            mc["avg_pct_ndcg"],
            mc["pct_bottom2_coverage"]
        ]
        
        # 归一化到0-1范围
        max_vals = [max(r, p) for r, p in zip(rank_vals, pct_vals)]
        rank_norm = [r / m if m > 0 else 0 for r, m in zip(rank_vals, max_vals)]
        pct_norm = [p / m if m > 0 else 0 for p, m in zip(pct_vals, max_vals)]
        
        # 绘制条形图比较
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.barh(x - width/2, rank_vals, width, label="Rank Method",
                        color=COLORS["rank"], alpha=0.85)
        bars2 = ax.barh(x + width/2, pct_vals, width, label="Percent Method",
                        color=COLORS["percent"], alpha=0.85)
        
        ax.set_yticks(x)
        ax.set_yticklabels(categories)
        ax.set_xlabel("Score", fontsize=11)
        ax.set_title(f'Method Comparison Summary\nRecommended: {rec["recommended_method"].upper()} Method',
                    fontsize=14)
        ax.legend(loc="lower right")
        
        # 添加数值标签
        for bar, val in zip(bars1, rank_vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f"{val:.3f}", va="center", fontsize=9)
        for bar, val in zip(bars2, pct_vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va="center", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig6_recommendation_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_heatmap_comparison(self):
        """图7: 两种方法下的排序差异热力图"""
        wr = self.method_analyzer.weekly_results
        
        # 创建赛季x周次的肯德尔相关系数热力图
        pivot_data = wr.pivot_table(
            values="kendall_tau",
            index="season",
            columns="week",
            aggfunc="mean"
        )
        
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
        
        custom_cmap = LinearSegmentedColormap.from_list(
            "correlation", ["#E53E3E", "#FFFFFF", "#3182CE"]
        )
        
        sns.heatmap(
            pivot_data, cmap=custom_cmap, ax=ax,
            center=0.5, vmin=0, vmax=1,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Kendall Tau (method agreement)"}
        )
        
        ax.set_xlabel("Week", fontsize=12)
        ax.set_ylabel("Season", fontsize=12)
        ax.set_title("Method Ranking Agreement (Kendall Tau) by Season and Week", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig7_kendall_tau_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_phi_comparison(self):
        """图8: RankSHAP Phi值对比"""
        wr = self.method_analyzer.weekly_results
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        
        # 左图：Phi_J和Phi_F的分布
        ax1 = axes[0]
        
        data_to_plot = [
            wr["rank_phi_J"].dropna(),
            wr["rank_phi_F"].dropna(),
            wr["pct_phi_J"].dropna(),
            wr["pct_phi_F"].dropna()
        ]
        
        bp = ax1.boxplot(data_to_plot, labels=["Rank φ_J", "Rank φ_F", "Pct φ_J", "Pct φ_F"],
                        patch_artist=True)
        
        colors_box = [COLORS["rank"], COLORS["rank"], COLORS["percent"], COLORS["percent"]]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax1.set_ylabel("Shapley Value", fontsize=11)
        ax1.set_title("Distribution of RankSHAP Values", fontsize=12)
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        
        # 右图：Phi_F/Phi_J比值
        ax2 = axes[1]
        
        wr["rank_ratio"] = wr["rank_phi_F"] / (wr["rank_phi_J"].abs() + 1e-8)
        wr["pct_ratio"] = wr["pct_phi_F"] / (wr["pct_phi_J"].abs() + 1e-8)
        
        ax2.scatter(wr["rank_ratio"], wr["pct_ratio"], alpha=0.4, 
                   c=wr["season"], cmap="viridis", s=30)
        
        ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax2.axvline(x=1, color="gray", linestyle="--", alpha=0.5)
        ax2.plot([0, 5], [0, 5], "r--", alpha=0.5, label="y=x")
        
        ax2.set_xlabel("Rank Method φ_F/|φ_J|", fontsize=11)
        ax2.set_ylabel("Percent Method φ_F/|φ_J|", fontsize=11)
        ax2.set_title("Fan vs Judge Contribution Ratio", fontsize=12)
        ax2.legend()
        ax2.set_xlim(0, 5)
        ax2.set_ylim(0, 5)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig8_phi_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def run(self):
        """生成所有图表"""
        self.plot_conflict_rate_by_season()
        self.plot_bias_comparison()
        self.plot_accuracy_comparison()
        self.plot_controversy_analysis()
        self.plot_counterfactual_summary()
        self.plot_recommendation_summary()
        self.plot_heatmap_comparison()
        self.plot_phi_comparison()
        return self


# ============================================================
# 主函数 & 指标汇总
# ============================================================
def main():
    """主函数：运行完整分析并输出汇总指标"""
    
    # Part 1: 方法对比分析
    method_analyzer = VotingMethodAnalyzer()
    method_analyzer.run()
    
    # Part 2: 争议案例分析
    controversy_analyzer = ControversyAnalyzer(method_analyzer)
    controversy_analyzer.run()
    
    # Part 3: Twist规则评估
    twist_evaluator = TwistRuleEvaluator(controversy_analyzer)
    twist_evaluator.run()
    
    # Part 4: 可视化
    visualizer = Q2Visualizer(method_analyzer, controversy_analyzer, twist_evaluator)
    visualizer.run()
    
    # ============================================================
    # 汇总指标输出
    # ============================================================
    summary = {
        "=== Q2 分析结果汇总 ===": "",
        
        # 规则冲突率
        "【规则冲突率】": "",
        "总体冲突率": f"{method_analyzer.conflict_summary['total_conflict_rate']:.4f}",
        "冲突周次数/总淘汰周次数": f"{method_analyzer.conflict_summary['total_conflict_count']}/{method_analyzer.conflict_summary['total_elim_weeks']}",
        
        # 方法准确性
        "【方法准确性对比】": "",
        "排名法淘汰预测准确率": f"{method_analyzer.method_comparison['rank_accuracy']:.4f}",
        "百分比法淘汰预测准确率": f"{method_analyzer.method_comparison['pct_accuracy']:.4f}",
        "排名法Bottom-2覆盖率": f"{method_analyzer.method_comparison['rank_bottom2_coverage']:.4f}",
        "百分比法Bottom-2覆盖率": f"{method_analyzer.method_comparison['pct_bottom2_coverage']:.4f}",
        
        # 偏向性分析
        "【偏向性分析】": "",
        "排名法平均粉丝偏向": f"{method_analyzer.method_comparison['avg_rank_bias']:.4f}",
        "百分比法平均粉丝偏向": f"{method_analyzer.method_comparison['avg_pct_bias']:.4f}",
        "平均肯德尔相关系数": f"{method_analyzer.method_comparison['avg_kendall_tau']:.4f}",
        
        # NDCG
        "【排序质量(NDCG)】": "",
        "排名法平均NDCG": f"{method_analyzer.method_comparison['avg_rank_ndcg']:.4f}",
        "百分比法平均NDCG": f"{method_analyzer.method_comparison['avg_pct_ndcg']:.4f}",
        
        # 争议案例
        "【争议案例分析】": "",
        "分析的争议案例数": f"{len(controversy_analyzer.controversy_results)}",
        
        # 反事实模拟
        "【反事实模拟】": "",
        "方法一致率(两种方法结果相同)": f"{controversy_analyzer.counterfactual_results['methods_agree'].mean():.4f}",
        "排名法匹配真实淘汰率": f"{controversy_analyzer.counterfactual_results['rank_matches_true'].mean():.4f}",
        "百分比法匹配真实淘汰率": f"{controversy_analyzer.counterfactual_results['pct_matches_true'].mean():.4f}",
        
        # Twist规则影响
        "【底部二选一规则影响】": "",
        "Twist规则改变结果比例": f"{twist_evaluator.twist_impact['overall_change_rate']:.4f}",
        
        # 推荐
        "【方法推荐】": "",
        "推荐方法": twist_evaluator.recommendation['recommended_method'],
        "推荐使用Twist规则": "是" if twist_evaluator.recommendation['twist_recommended'] else "否",
        "排名法得分": f"{twist_evaluator.recommendation['rank_score']}",
        "百分比法得分": f"{twist_evaluator.recommendation['pct_score']}",
    }
    
    # 输出汇总
    print("\n" + "=" * 70)
    for key, value in summary.items():
        if value == "":
            print(f"\n{key}")
        else:
            print(f"  {key}: {value}")
    print("=" * 70)
    
    # 保存详细结果
    method_analyzer.weekly_results.to_csv("q2_weekly_comparison.csv", index=False)
    controversy_analyzer.counterfactual_results.to_csv("q2_counterfactual_results.csv", index=False)
    
    # 保存汇总指标
    summary_df = pd.DataFrame([
        {"metric": k, "value": v} for k, v in summary.items() if v != ""
    ])
    summary_df.to_csv("q2_summary_metrics.csv", index=False)
    
    print(f"\n【输出文件】")
    print(f"  - q2_weekly_comparison.csv (周级对比结果)")
    print(f"  - q2_counterfactual_results.csv (反事实模拟结果)")
    print(f"  - q2_summary_metrics.csv (汇总指标)")
    print(f"  - {OUTPUT_DIR}/ (所有可视化图表)")
    
    return method_analyzer, controversy_analyzer, twist_evaluator, summary


if __name__ == "__main__":
    method_analyzer, controversy_analyzer, twist_evaluator, summary = main()

