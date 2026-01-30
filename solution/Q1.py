# ============================================================
# MCM 2026 Problem C - Question 1: 粉丝投票估算完整解决方案
# 包含三个小问：
#   Part 1: 粉丝投票估算模型
#   Part 2: 淘汰一致性评估
#   Part 3: 不确定性度量分析
# ============================================================

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')

# -----------------------------
# 全局配置 & 可视化风格 (对齐 data_visualization.py)
# -----------------------------
np.random.seed(42)
RANDOM_SEED = 2026

# 主题配色
COLORS = {
    "primary": "#7BADDF",      # 浅蓝
    "secondary": "#B581B4",    # 薰衣草紫
    "accent": "#EAB170",       # 暖橙
    "success": "#DA8176",      # 珊瑚粉
    "neutral": "#B1A8D3",      # 淡紫
    "light": "#BADDF3"         # 极浅蓝
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
OUTPUT_DIR = Path("plots/q1_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PART 1: 粉丝投票估算模型
# ============================================================
class FanVoteEstimator:
    """粉丝投票估算器：基于生存概率的弱监督学习"""
    
    def __init__(self, data_path: str = "dwts_cleaned.csv", long_data_path: str = "dwts_long_format.csv"):
        self.data_path = data_path
        self.long_data_path = long_data_path
        self.df = None
        self.long_df = None
        self.train_df = None
        self.pred_df = None
        self.result_df = None
        self.season_weeks = {}
        self.pipe = None
        
        # 特征定义（基于清洗后的字段）
        self.num_features = [
            "week", "judge_total", "judge_mean", "judge_count",
            "judge_total_delta", "cum_judge_mean", "cum_judge_sum",
            "roll3_mean", "roll3_std", "weekly_rank", "weekly_rank_pct",
            "judge_total_week_z", "age", "partner_experience", "is_us",
        ]
        self.cat_features = [
            "celebrity_industry", "industry_category", "age_group",
            "home_state", "home_country", "partner",
        ]
    
    def load_data(self):
        """加载清洗后的宽表与长表数据"""
        print("=" * 60)
        print("PART 1: 粉丝投票估算模型")
        print("=" * 60)
        
        if not (Path(self.data_path).exists() and Path(self.long_data_path).exists()):
            raise FileNotFoundError("请先运行 data_cleaning.py 生成 dwts_cleaned.csv 与 dwts_long_format.csv。")

        self.df = pd.read_csv(self.data_path)
        self.long_df = pd.read_csv(self.long_data_path)

        # 基于清洗结果补充必要字段
        if "weeks_participated" in self.df.columns:
            self.df["last_active_week"] = self.df["weeks_participated"]
        self.season_weeks = self.df.groupby("season")["weeks_participated"].max().to_dict()
        self.long_df["season_weeks"] = self.long_df["season"].map(self.season_weeks)
        self.long_df["is_active"] = True
        if "is_elimination_week" not in self.long_df.columns and "elimination_week" in self.df.columns:
            elim_map = self.df.set_index(["season", "celebrity_name"])["elimination_week"].to_dict()
            self.long_df["is_elimination_week"] = self.long_df.apply(
                lambda r: int(elim_map.get((r["season"], r["celebrity_name"])) == r["week"]),
                axis=1
            )

        print(f"\n[Step 1] 清洗数据加载完成")
        print(f"  Cleaned Data Shape: {self.df.shape}")
        print(f"  Long Format Shape: {self.long_df.shape}")
        return self
    
    def build_features(self):
        """构建动态特征"""
        self.long_df = self.long_df.sort_values(
            ["season", "celebrity_name", "week"]
        ).reset_index(drop=True)
        
        # 上一周总分与变化量
        self.long_df["judge_total_prev"] = self.long_df.groupby(
            ["season", "celebrity_name"]
        )["judge_total"].shift(1)
        self.long_df["judge_total_delta"] = (
            self.long_df["judge_total"] - self.long_df["judge_total_prev"]
        )
        
        # 累计均值/累计总分
        self.long_df["cum_judge_mean"] = (
            self.long_df.groupby(["season", "celebrity_name"])["judge_total"]
        .expanding().mean().reset_index(level=[0, 1], drop=True)
    )
        self.long_df["cum_judge_sum"] = self.long_df.groupby(
            ["season", "celebrity_name"]
        )["judge_total"].cumsum()

        # 近3周滚动均值/波动
        self.long_df["roll3_mean"] = (
            self.long_df.groupby(["season", "celebrity_name"])["judge_total"]
        .rolling(3, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    )
        self.long_df["roll3_std"] = (
            self.long_df.groupby(["season", "celebrity_name"])["judge_total"]
        .rolling(3, min_periods=1).std().reset_index(level=[0, 1], drop=True)
    )

        # 当周评分排名（若清洗数据未包含，则补充）
        if "weekly_rank" not in self.long_df.columns:
            self.long_df["weekly_rank"] = self.long_df.groupby(
                ["season", "week"]
            )["judge_total"].rank(ascending=False, method="min")
        if "weekly_rank_pct" not in self.long_df.columns:
            self.long_df["weekly_rank_pct"] = self.long_df.groupby(
                ["season", "week"]
            )["judge_total"].rank(ascending=False, pct=True)
        
        # 当周评分标准化
        self.long_df["judge_total_week_mean"] = self.long_df.groupby(
            ["season", "week"]
        )["judge_total"].transform("mean")
        self.long_df["judge_total_week_std"] = self.long_df.groupby(
            ["season", "week"]
        )["judge_total"].transform("std").replace(0, np.nan)
        self.long_df["judge_total_week_z"] = (
            (self.long_df["judge_total"] - self.long_df["judge_total_week_mean"]) /
            self.long_df["judge_total_week_std"]
        )
        
        # 填补缺失值
        num_fill_cols = list({c for c in self.num_features if c in self.long_df.columns})
        for col in num_fill_cols:
            mean_val = self.long_df[col].mean()
            self.long_df[col] = self.long_df[col].fillna(mean_val)
        cat_fill_cols = [c for c in self.cat_features if c in self.long_df.columns]
        for col in cat_fill_cols:
            self.long_df[col] = self.long_df[col].fillna("Unknown")
        
        print(f"\n[Step 5] 特征构造完成")
        return self
    
    def prepare_training_data(self):
        """准备训练集"""
        self.train_df = self.long_df[
            self.long_df["is_active"] & 
            (self.long_df["week"] < self.long_df["season_weeks"])
        ].copy()
        self.train_df["y_survive_next"] = 1 - self.train_df["is_elimination_week"].fillna(0).astype(int)
        
        # 填补训练集缺失值
        num_fill_cols = list({c for c in self.num_features if c in self.train_df.columns})
        for col in num_fill_cols:
            mean_val = self.train_df[col].mean()
            self.train_df[col] = self.train_df[col].fillna(mean_val)
        cat_fill_cols = [c for c in self.cat_features if c in self.train_df.columns]
        for col in cat_fill_cols:
            self.train_df[col] = self.train_df[col].fillna("Unknown")
        
        print(f"\n[Step 6] 训练集构建完成")
        print(f"  Train shape: {self.train_df.shape}")
        print(f"  Label distribution: survive={self.train_df['y_survive_next'].sum()}, "
              f"eliminated={len(self.train_df) - self.train_df['y_survive_next'].sum()}")
        return self
    
    def train_model(self):
        """训练生存概率模型"""
        preprocess = ColumnTransformer(
        transformers=[
                ("num", StandardScaler(), self.num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features),
            ],
            remainder="drop"
        )
        
        clf = GradientBoostingClassifier(random_state=42)
        self.pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
        
        X = self.train_df[self.num_features + self.cat_features]
        y = self.train_df["y_survive_next"].astype(int)
        groups = self.train_df["season"].values
        
        # 交叉验证评估
        print(f"\n[Step 7] GroupKFold 交叉验证评估：")
        gkf = GroupKFold(n_splits=5)
        auc_list, acc_list = [], []
        
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            
            self.pipe.fit(X_tr, y_tr)
            proba_va = self.pipe.predict_proba(X_va)[:, 1]
            pred_va = (proba_va >= 0.5).astype(int)
            
            auc = roc_auc_score(y_va, proba_va)
            acc = accuracy_score(y_va, pred_va)
            auc_list.append(auc)
            acc_list.append(acc)
            print(f"  Fold {fold}: AUC={auc:.4f}, ACC={acc:.4f}")
        
        print(f"\n  CV Summary: AUC={np.mean(auc_list):.4f}±{np.std(auc_list):.4f}, "
              f"ACC={np.mean(acc_list):.4f}±{np.std(acc_list):.4f}")
        
        # 全量训练
        self.pipe.fit(X, y)
        return self
    
    def predict_votes(self):
        """预测粉丝票数"""
        self.pred_df = self.long_df[self.long_df["is_active"]].copy()
        X_pred = self.pred_df[self.num_features + self.cat_features]
        
        self.pred_df["p_survive_next"] = self.pipe.predict_proba(X_pred)[:, 1]
        self.pred_df["p_eliminate_end_week"] = 1 - self.pred_df["p_survive_next"]
        
        # 融合评委表现得到人气分数
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        self.pred_df["judge_z_sigmoid"] = sigmoid(
            self.pred_df["judge_total_week_z"].fillna(0)
        )
        self.pred_df["popularity_raw"] = (
            0.5 * self.pred_df["p_survive_next"] + 
            0.5 * self.pred_df["judge_z_sigmoid"]
        )
        self.pred_df["popularity_score"] = np.exp(self.pred_df["popularity_raw"])

    # 周内归一化成份额
        self.pred_df["vote_share_hat"] = self.pred_df.groupby(
            ["season", "week"]
        )["popularity_score"].transform(lambda s: s / s.sum())
        
        # 设定票池
    BASE_TOTAL_VOTES = 1_000_000
        
    def week_vote_pool(row):
        w = row["week"]
        W = row["season_weeks"]
        if W <= 1:
            return BASE_TOTAL_VOTES
        factor = 0.8 + 0.6 * (w - 1) / (W - 1)
        return BASE_TOTAL_VOTES * factor
    
        self.pred_df["total_votes_hat"] = self.pred_df.apply(week_vote_pool, axis=1)
        self.pred_df["votes_hat"] = (
            self.pred_df["vote_share_hat"] * self.pred_df["total_votes_hat"]
        )
        
        # 构建结果表
        result_cols = [
            "season", "week", "celebrity_name", "partner",
            "celebrity_industry", "industry_category", "home_state", "home_country",
            "age", "age_group", "judge_total", "judge_mean", "judge_count",
            "p_survive_next", "vote_share_hat", "total_votes_hat", "votes_hat",
            "placement", "elimination_week"
        ]
        self.result_df = self.pred_df[result_cols].sort_values(
            ["season", "week", "votes_hat"], ascending=[True, True, False]
        )
        
        print(f"\n[Step 8] 票数预测完成")
        print(f"  预测行数: {len(self.pred_df)}")
        print(f"  平均预测票数: {self.pred_df['votes_hat'].mean():,.0f}")
        return self
    
    def save_results(self, output_path: str = "q1_fan_vote_estimates.csv"):
        """保存结果"""
        self.result_df.to_csv(output_path, index=False)
        print(f"\n[Step 9] 结果已保存至: {output_path}")
        return self
    
    def run(self):
        """运行完整流程"""
        return (self.load_data()
                .build_features()
                .prepare_training_data()
                .train_model()
                .predict_votes()
                .save_results())


# ============================================================
# PART 2: 淘汰一致性评估
# ============================================================
class EliminationConsistencyEvaluator:
    """淘汰一致性评估器"""
    
    def __init__(self, estimator: FanVoteEstimator):
        self.estimator = estimator
        self.df = estimator.df
        self.result_df = estimator.result_df
        self.season_weeks = estimator.season_weeks
        self.truth_map = {}
        self.eval_week_df = None
        self.season_summary = None
        
        # 赛制规则配置
        self.max_season = int(self.result_df["season"].max())
        self.rank_seasons = set([1, 2]) | set(range(28, self.max_season + 1))
        self.pct_seasons = set(range(3, 28))
        self.TWIST_START_SEASON = 28
        self.ENABLE_TWIST = True
    
    def build_truth_map(self):
        """构建真实淘汰映射"""
        print("\n" + "=" * 60)
        print("PART 2: 淘汰一致性评估")
        print("=" * 60)
        
        df_truth = self.df[["season", "celebrity_name", "elimination_week"]].copy()
        df_truth["elim_week"] = np.where(
            df_truth["elimination_week"].notna() & (df_truth["elimination_week"] > 0),
            df_truth["elimination_week"],
            np.nan
        )
        
        self.truth_map = (
            df_truth.dropna(subset=["elim_week"])
            .groupby(["season", "elim_week"])["celebrity_name"]
            .apply(list)
            .to_dict()
        )
        
        print(f"\n[Step 2-1] 真实淘汰映射构建完成")
        print(f"  淘汰周次数: {len(self.truth_map)}")
        return self
    
    def predict_elims_rank(self, g: pd.DataFrame, k: int):
        """Rank法预测淘汰"""
        if k <= 0:
            return [], []
        
        gg = g.copy()
        gg["judge_rank_w"] = gg["judge_total"].rank(ascending=False, method="min")
        gg["vote_rank_w"] = gg["votes_hat"].rank(ascending=False, method="min")
        gg["combined_rank"] = gg["judge_rank_w"] + gg["vote_rank_w"]
        
        gg = gg.sort_values(
            ["combined_rank", "judge_total", "votes_hat"],
            ascending=[False, True, True]
        )
        
        pred = gg["celebrity_name"].head(k).tolist()
        bottom2 = gg["celebrity_name"].head(2).tolist()
        return pred, bottom2
    
    def predict_elims_percent(self, g: pd.DataFrame, k: int):
        """Percentage法预测淘汰"""
        if k <= 0:
            return [], []
        
        gg = g.copy()
        judge_sum = gg["judge_total"].sum()
        vote_sum = gg["votes_hat"].sum()
        
        gg["judge_pct"] = gg["judge_total"] / judge_sum if judge_sum > 0 else 0.0
        gg["vote_pct"] = gg["votes_hat"] / vote_sum if vote_sum > 0 else 0.0
        gg["combined_pct"] = gg["judge_pct"] + gg["vote_pct"]
        
        gg = gg.sort_values(
            ["combined_pct", "judge_total", "votes_hat"],
            ascending=[True, True, True]
        )
        
        pred = gg["celebrity_name"].head(k).tolist()
        bottom2 = gg["celebrity_name"].head(2).tolist()
        return pred, bottom2
    
    def apply_twist(self, g: pd.DataFrame, base_bottom2: list, k: int):
        """应用Twist机制（评委救人）"""
        if (k != 1) or (len(base_bottom2) < 2):
            return None
        
        gg = g.set_index("celebrity_name")
        a, b = base_bottom2[0], base_bottom2[1]
        ja = gg.loc[a, "judge_total"] if a in gg.index else np.nan
        jb = gg.loc[b, "judge_total"] if b in gg.index else np.nan
        
        if np.isfinite(ja) and np.isfinite(jb):
            elim = a if ja < jb else b
        else:
            va = gg.loc[a, "votes_hat"] if a in gg.index else np.nan
            vb = gg.loc[b, "votes_hat"] if b in gg.index else np.nan
            elim = a if va < vb else b
        
        return [elim]
    
    def evaluate_consistency(self):
        """评估淘汰一致性"""
        pred_rows = []
        week_groups = self.result_df.groupby(["season", "week"], sort=True)
        
        for (s, w), g in week_groups:
            s, w = int(s), int(w)
            
            true_list = self.truth_map.get((s, w), self.truth_map.get((s, float(w)), []))
            true_set = set(true_list)
            k = len(true_list)
            
            rule = "rank" if s in self.rank_seasons else "percent"
            use_twist = self.ENABLE_TWIST and (s >= self.TWIST_START_SEASON)
            
            if rule == "rank":
                pred_list, bottom2 = self.predict_elims_rank(g, k)
            else:
                pred_list, bottom2 = self.predict_elims_percent(g, k)
            
            if use_twist and k == 1:
                twist_pred = self.apply_twist(g, bottom2, k)
                if twist_pred is not None:
                    pred_list = twist_pred
            
            pred_set = set(pred_list)
            exact_match = int(pred_set == true_set)
            hit = int(len(true_set.intersection(pred_set)) == len(true_set)) if k > 0 else int(len(pred_set) == 0)
            
            bottom2_set = set(bottom2) if isinstance(bottom2, list) else set()
            bottom2_cover = int((k > 0) and (len(true_set.intersection(bottom2_set)) > 0))
            
            pred_rows.append({
                "season": s, "week": w,
                "rule_used": ("rank+twist" if (rule == "rank" and use_twist and k == 1) else rule),
                "true_k": k,
                "true_eliminated": sorted(list(true_set)),
                "pred_eliminated": sorted(list(pred_set)),
                "bottom2_pred": sorted(list(bottom2_set)),
                "exact_match": exact_match,
                "hit_all_true": hit,
                "bottom2_cover_true": bottom2_cover
            })
        
        self.eval_week_df = pd.DataFrame(pred_rows).sort_values(["season", "week"]).reset_index(drop=True)
        return self
    
    def compute_metrics(self):
        """计算一致性指标"""
        mask_elim = self.eval_week_df["true_k"] > 0
        
        exact_elim = self.eval_week_df.loc[mask_elim, "exact_match"].mean()
        exact_all = self.eval_week_df["exact_match"].mean()
        hit_elim = self.eval_week_df.loc[mask_elim, "hit_all_true"].mean()
        bottom2_cov = self.eval_week_df.loc[mask_elim, "bottom2_cover_true"].mean()
        
        print(f"\n[Metrics] 周级一致性评估指标：")
        print(f"  淘汰周精确匹配率 = {exact_elim:.4f}")
        print(f"  全部周精确匹配率 = {exact_all:.4f}")
        print(f"  淘汰命中率       = {hit_elim:.4f}")
        print(f"  Bottom-2覆盖率   = {bottom2_cov:.4f}")
        
        # 赛季级汇总
        self.season_summary = (
            self.eval_week_df.groupby("season")
            .apply(lambda x: pd.Series({
                "weeks": len(x),
                "elim_weeks": int((x["true_k"] > 0).sum()),
                "exact_all": x["exact_match"].mean(),
                "exact_elim": x.loc[x["true_k"] > 0, "exact_match"].mean() if (x["true_k"] > 0).any() else np.nan,
                "bottom2_cover": x.loc[x["true_k"] > 0, "bottom2_cover_true"].mean() if (x["true_k"] > 0).any() else np.nan
            }))
            .reset_index()
        )
        
        print(f"\n[Metrics] 赛季级一致性汇总（前10行）：")
        print(self.season_summary.head(10).to_string())
        return self
    
    def plot_consistency(self):
        """绘制一致性图表"""
        # 图1: 赛季级淘汰一致率柱状图
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
        ax.bar(self.season_summary["season"].astype(str), 
               self.season_summary["exact_elim"], color=COLORS["primary"], edgecolor='navy', alpha=0.85)
        ax.axhline(y=self.season_summary["exact_elim"].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Average: {self.season_summary["exact_elim"].mean():.3f}')
        ax.set_title("Season-level Elimination Exact Match Rate", fontsize=14, fontweight='bold')
        ax.set_xlabel("Season", fontsize=12)
        ax.set_ylabel("Exact Match Rate", fontsize=12)
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig1_season_consistency.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 图2: 周级一致率（按周次聚合）
        week_summary = (
            self.eval_week_df.groupby("week", as_index=False)
            .agg(exact_match_mean=("exact_match", "mean"))
        )
        
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        ax.plot(week_summary["week"], week_summary["exact_match_mean"], 
                marker="o", linewidth=2, markersize=8, color=COLORS["accent"])
        ax.fill_between(week_summary["week"], 0, week_summary["exact_match_mean"], alpha=0.2, color='orange')
        ax.set_title("Weekly Exact Match Rate (Averaged Across Seasons)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Week", fontsize=12)
        ax.set_ylabel("Exact Match Rate", fontsize=12)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig2_weekly_consistency.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 图3: Bottom-2覆盖率
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
        ax.bar(self.season_summary["season"].astype(str), 
               self.season_summary["bottom2_cover"], color=COLORS["success"], edgecolor='darkred', alpha=0.85)
        ax.axhline(y=self.season_summary["bottom2_cover"].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Average: {self.season_summary["bottom2_cover"].mean():.3f}')
        ax.set_title("Bottom-2 Coverage Rate by Season", fontsize=14, fontweight='bold')
        ax.set_xlabel("Season", fontsize=12)
        ax.set_ylabel("Bottom-2 Coverage Rate", fontsize=12)
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig3_bottom2_coverage.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 图4: 评分热力图（浅蓝 -> 珊瑚粉）
        pivot_data = self.result_df.pivot_table(
            values="judge_total", index="season", columns="week", aggfunc="mean"
        )
        custom_heatmap_cmap = LinearSegmentedColormap.from_list(
            "dwts_theme", [COLORS["light"], COLORS["primary"], COLORS["secondary"], COLORS["success"]]
        )
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
        sns.heatmap(
            pivot_data, cmap=custom_heatmap_cmap, ax=ax,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Average Judge Total Score"}
        )
        ax.set_title("Average Judge Total Score Heatmap (All Seasons)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Week", fontsize=12)
        ax.set_ylabel("Season", fontsize=12)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig2_score_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\n[Plots] 一致性图表已保存至 {OUTPUT_DIR}")
        return self
    
    def run(self):
        """运行完整评估"""
        return (self.build_truth_map()
                .evaluate_consistency()
                .compute_metrics()
                .plot_consistency())


# ============================================================
# PART 3: 不确定性度量分析
# ============================================================
class UncertaintyAnalyzer:
    """不确定性分析器：Bootstrap方法"""
    
    def __init__(self, estimator: FanVoteEstimator, B: int = 60, sigma_pool: float = 0.18):
        self.estimator = estimator
        self.B = B  # Bootstrap次数
        self.SIGMA_POOL = sigma_pool  # 票池噪声强度
        self.rng = np.random.default_rng(RANDOM_SEED)
        
        self.unc_df = None
        self.votes_boot = None
        self.share_boot = None
        self.pool_boot = None
        self.week_unc = None
        self.cele_unc = None
    
    def prepare_data(self):
        """准备不确定性分析数据"""
        print("\n" + "=" * 60)
        print("PART 3: 不确定性度量分析")
        print("=" * 60)
        
        self.unc_df = self.estimator.pred_df.copy()
        self.unc_df = self.unc_df.sort_values(
            ["season", "week", "celebrity_name"]
        ).reset_index(drop=True)
        
        self.unc_df["key"] = (
            self.unc_df["season"].astype(str) + "|" +
            self.unc_df["week"].astype(str) + "|" +
            self.unc_df["celebrity_name"].astype(str)
        )
        
        # 准备season-week索引
        sw_unique = self.unc_df[["season", "week"]].drop_duplicates().reset_index(drop=True)
        sw_unique["sw_key"] = sw_unique["season"].astype(str) + "|" + sw_unique["week"].astype(str)
        self.SW = len(sw_unique)
        self.sw_index = {k: i for i, k in enumerate(sw_unique["sw_key"].tolist())}
        self.sw_unique = sw_unique
        
        self.unc_df["sw_key"] = self.unc_df["season"].astype(str) + "|" + self.unc_df["week"].astype(str)
        self.unc_df["sw_idx"] = self.unc_df["sw_key"].map(self.sw_index).astype(int)
        
        self.N = len(self.unc_df)
        print(f"\n[Step 3-1] 数据准备完成")
        print(f"  样本数: {self.N}")
        print(f"  Season-Week数: {self.SW}")
        return self
    
    def run_bootstrap(self):
        """运行Bootstrap"""
        print(f"\n[Step 3-2] 开始Bootstrap (B={self.B})...")
        
        train_df = self.estimator.train_df.copy()
        num_features = self.estimator.num_features
        cat_features = self.estimator.cat_features
        pipe = self.estimator.pipe
        
        X_base = train_df[num_features + cat_features]
        y_base = train_df["y_survive_next"].astype(int).values
        season_base = train_df["season"].astype(int).values
        seasons_all = np.sort(train_df["season"].unique())
        
        self.votes_boot = np.zeros((self.B, self.N), dtype=float)
        self.share_boot = np.zeros((self.B, self.N), dtype=float)
        self.pool_boot = np.zeros((self.B, self.SW), dtype=float)
        
        def _sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        for b in range(self.B):
            # 分层重采样
            sampled_seasons = self.rng.choice(seasons_all, size=len(seasons_all), replace=True)
            idx_list = [np.where(season_base == s)[0] for s in sampled_seasons]
            idx_boot = np.concatenate(idx_list)
            
            Xb = X_base.iloc[idx_boot]
            yb = y_base[idx_boot]
            
            # 重新拟合模型
            pipe.fit(Xb, yb)
            
            # 预测
            X_unc = self.unc_df[num_features + cat_features]
            p_survive = pipe.predict_proba(X_unc)[:, 1]
            
            # 融合得到份额
            judge_z_sig = _sigmoid(self.unc_df["judge_total_week_z"].fillna(0).values)
            popularity_raw = 0.5 * p_survive + 0.5 * judge_z_sig
            popularity_score = np.exp(popularity_raw)
            
            sw_sum = (
                pd.Series(popularity_score)
                .groupby([self.unc_df["season"].values, self.unc_df["week"].values])
                .transform("sum")
                .values
            )
            vote_share = popularity_score / np.where(sw_sum > 0, sw_sum, 1.0)
            
            # 票池噪声
            base_pool = self.unc_df["total_votes_hat"].values
            mu = -0.5 * (self.SIGMA_POOL ** 2)
            factors = np.exp(self.rng.normal(loc=mu, scale=self.SIGMA_POOL, size=self.SW))
            pool_row = base_pool * factors[self.unc_df["sw_idx"].values]
            
            sw_base_pool = (
                self.unc_df.groupby("sw_idx")["total_votes_hat"].first()
                .reindex(range(self.SW)).values
            )
            self.pool_boot[b, :] = sw_base_pool * factors
            
            # 票数
            votes = vote_share * pool_row
            
            self.votes_boot[b, :] = votes
            self.share_boot[b, :] = vote_share
            
            if (b + 1) % 20 == 0 or b == 0:
                print(f"  Bootstrap {b+1}/{self.B} 完成")
        
        print(f"\n[Step 3-2] Bootstrap完成")
        return self
    
    def compute_uncertainty_metrics(self):
        """计算不确定性指标"""
        def q(a, p):
            return np.quantile(a, p, axis=0)
        
        self.unc_df["votes_q05"] = q(self.votes_boot, 0.05)
        self.unc_df["votes_q10"] = q(self.votes_boot, 0.10)
        self.unc_df["votes_q50"] = q(self.votes_boot, 0.50)
        self.unc_df["votes_q90"] = q(self.votes_boot, 0.90)
        self.unc_df["votes_q95"] = q(self.votes_boot, 0.95)
        
        self.unc_df["votes_mean"] = self.votes_boot.mean(axis=0)
        self.unc_df["votes_std"] = self.votes_boot.std(axis=0)
        
        self.unc_df["ci80_width"] = self.unc_df["votes_q90"] - self.unc_df["votes_q10"]
        self.unc_df["ci95_width"] = self.unc_df["votes_q95"] - self.unc_df["votes_q05"]
        
        self.unc_df["rel_ci80"] = self.unc_df["ci80_width"] / np.where(
            self.unc_df["votes_q50"] > 0, self.unc_df["votes_q50"], np.nan
        )
        self.unc_df["rel_ci95"] = self.unc_df["ci95_width"] / np.where(
            self.unc_df["votes_q50"] > 0, self.unc_df["votes_q50"], np.nan
        )
        self.unc_df["cv_votes"] = self.unc_df["votes_std"] / np.where(
            self.unc_df["votes_mean"] > 0, self.unc_df["votes_mean"], np.nan
        )
        
        print(f"\n[Step 3-3] 不确定性指标计算完成")
        print(f"  rel_ci80 分布:")
        print(f"    Mean: {self.unc_df['rel_ci80'].mean():.4f}")
        print(f"    Std:  {self.unc_df['rel_ci80'].std():.4f}")
        print(f"    Min:  {self.unc_df['rel_ci80'].min():.4f}")
        print(f"    Max:  {self.unc_df['rel_ci80'].max():.4f}")
        return self
    
    def aggregate_uncertainty(self):
        """按周/按人汇总不确定性"""
        # 按周汇总
        self.week_unc = (
            self.unc_df.groupby("week", as_index=False)
            .agg(
                rel_ci80_mean=("rel_ci80", "mean"),
                rel_ci80_std=("rel_ci80", "std"),
                cv_mean=("cv_votes", "mean"),
                n=("key", "count")
            )
            .sort_values("week")
        )
        
        # 按选手汇总
        self.cele_unc = (
            self.unc_df.groupby("celebrity_name", as_index=False)
            .agg(
                rel_ci80_mean=("rel_ci80", "mean"),
                rel_ci80_max=("rel_ci80", "max"),
                cv_mean=("cv_votes", "mean"),
                n=("key", "count")
            )
            .sort_values("rel_ci80_mean", ascending=False)
        )
        
        print(f"\n[Step 3-4] 汇总完成")
        print(f"  周维度样本数: {len(self.week_unc)}")
        print(f"  选手维度样本数: {len(self.cele_unc)}")
        return self
    
    def plot_uncertainty(self):
        """绘制不确定性图表"""
        # 图4: rel_ci80整体分布
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax.hist(self.unc_df["rel_ci80"].dropna(), bins=40, edgecolor="black", 
                color='mediumpurple', alpha=0.7)
        ax.axvline(x=self.unc_df["rel_ci80"].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.unc_df["rel_ci80"].mean():.3f}')
        ax.set_title("Distribution of Relative Uncertainty (CI80 / Median)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Relative CI80 Width", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig4_uncertainty_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 图5: 不确定性随周变化
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax.plot(self.week_unc["week"], self.week_unc["rel_ci80_mean"], 
                marker="o", linewidth=2, markersize=8, color='teal')
        ax.fill_between(
            self.week_unc["week"],
            self.week_unc["rel_ci80_mean"] - self.week_unc["rel_ci80_std"],
            self.week_unc["rel_ci80_mean"] + self.week_unc["rel_ci80_std"],
            alpha=0.25, color='teal', label="±1 Std"
        )
        ax.set_title("Uncertainty by Week (Across All Seasons)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Week", fontsize=12)
        ax.set_ylabel("Relative CI80 Width", fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig5_uncertainty_by_week.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 图6: 不确定性最高的选手
        topk = 15
        top_cele = self.cele_unc.head(topk)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        bars = ax.bar(range(topk), top_cele["rel_ci80_mean"], color='coral', edgecolor='darkred', alpha=0.8)
        ax.set_xticks(range(topk))
        ax.set_xticklabels(top_cele["celebrity_name"], rotation=45, ha="right", fontsize=9)
        ax.set_title("Top Contestants with Highest Mean Uncertainty", fontsize=14, fontweight='bold')
        ax.set_xlabel("Celebrity", fontsize=12)
        ax.set_ylabel("Mean Relative CI80 Width", fontsize=12)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig6_top_uncertainty_contestants.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 图7: 示例赛季热力图
        example_season = int(self.unc_df["season"].max())
        ex_df_u = self.unc_df[self.unc_df["season"] == example_season].copy()
        
        top12_u = (
            ex_df_u.groupby("celebrity_name")["votes_q50"].sum()
            .sort_values(ascending=False).head(12).index.tolist()
        )
        
        heat_u = ex_df_u[ex_df_u["celebrity_name"].isin(top12_u)].pivot_table(
            index="celebrity_name", columns="week", values="rel_ci80", aggfunc="mean"
        ).fillna(0)
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        im = ax.imshow(heat_u.values, aspect="auto", cmap='YlOrRd')
        ax.set_title(f"Uncertainty Heatmap (Season {example_season}, Top12)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Week", fontsize=12)
        ax.set_ylabel("Celebrity", fontsize=12)
        ax.set_xticks(np.arange(heat_u.shape[1]))
        ax.set_xticklabels(heat_u.columns.tolist())
        ax.set_yticks(np.arange(heat_u.shape[0]))
        ax.set_yticklabels(heat_u.index.tolist(), fontsize=9)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Relative CI80 Width")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig7_uncertainty_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 图8: 预测票数 vs 最终名次散点图
        season_total = (
            self.estimator.result_df.groupby(["season", "celebrity_name"], as_index=False)
            .agg(pred_total_votes=("votes_hat", "sum"))
        )
        placement_df = self.estimator.df[["season", "celebrity_name", "placement"]].copy()
        
        scatter_df = season_total.merge(placement_df, on=["season", "celebrity_name"], how="inner")
        scatter_df = scatter_df.dropna(subset=["pred_total_votes", "placement"])
        
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        scatter = ax.scatter(scatter_df["pred_total_votes"], scatter_df["placement"], 
                            alpha=0.5, c='steelblue', edgecolor='navy', s=50)
        ax.set_title("Predicted Total Votes vs Final Placement", fontsize=14, fontweight='bold')
        ax.set_xlabel("Predicted Total Votes", fontsize=12)
        ax.set_ylabel("Final Placement (Lower is Better)", fontsize=12)
        
        # 计算相关系数
        corr = scatter_df["pred_total_votes"].corr(scatter_df["placement"])
        ax.text(0.95, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig8_votes_vs_placement.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n[Plots] 不确定性图表已保存至 {OUTPUT_DIR}")
        return self
    
    def run(self):
        """运行完整分析"""
        return (self.prepare_data()
                .run_bootstrap()
                .compute_uncertainty_metrics()
                .aggregate_uncertainty()
                .plot_uncertainty())


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("MCM 2026 Problem C - Question 1: 粉丝投票估算完整解决方案")
    print("=" * 70)
    
    # Part 1: 粉丝投票估算
    estimator = FanVoteEstimator()
    estimator.run()
    
    # Part 2: 淘汰一致性评估
    evaluator = EliminationConsistencyEvaluator(estimator)
    evaluator.run()
    
    # Part 3: 不确定性度量分析
    analyzer = UncertaintyAnalyzer(estimator, B=60, sigma_pool=0.18)
    analyzer.run()
    
    # 汇总报告
    print("\n" + "=" * 70)
    print("完整分析报告")
    print("=" * 70)
    
    print("\n【Part 1 - 粉丝投票估算】")
    print(f"  - 预测样本数: {len(estimator.pred_df)}")
    print(f"  - 平均预测票数: {estimator.pred_df['votes_hat'].mean():,.0f}")
    print(f"  - 结果已保存至: q1_fan_vote_estimates.csv")
    
    print("\n【Part 2 - 淘汰一致性评估】")
    mask_elim = evaluator.eval_week_df["true_k"] > 0
    print(f"  - 淘汰周精确匹配率: {evaluator.eval_week_df.loc[mask_elim, 'exact_match'].mean():.4f}")
    print(f"  - Bottom-2覆盖率: {evaluator.eval_week_df.loc[mask_elim, 'bottom2_cover_true'].mean():.4f}")
    
    print("\n【Part 3 - 不确定性度量】")
    print(f"  - Bootstrap次数: {analyzer.B}")
    print(f"  - 平均相对不确定性(rel_ci80): {analyzer.unc_df['rel_ci80'].mean():.4f}")
    print(f"  - 不确定性标准差: {analyzer.unc_df['rel_ci80'].std():.4f}")
    
    print(f"\n【输出文件】")
    print(f"  - q1_fan_vote_estimates.csv (票数估算结果)")
    print(f"  - {OUTPUT_DIR}/ (所有可视化图表)")
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    
    return estimator, evaluator, analyzer


if __name__ == "__main__":
    estimator, evaluator, analyzer = main()

