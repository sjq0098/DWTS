# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 22:18:16 2026

@author: 25046
"""

# ============================================================
# MCM 2026 Problem C - Question 1: 粉丝投票估算完整解决方案（增强版）
# 改进内容：
# 1. 区分赛季类型建模 (S1-2 Rank制, S3-27 Percentage制, S28+ Rank+Twist)
# 2. 特征工程增强 (相对表现、动量、危险区、赛季阶段等)
# 3. 两阶段估计 (约束优化反演票数)
# ============================================================

import re
import warnings
from pandas.api.types import is_categorical_dtype
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.optimize import minimize

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')

# -----------------------------
# 全局配置 & 可视化风格
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
# PART 1: 增强版粉丝投票估算器
# ============================================================
class FanVoteEstimator:
    """增强版粉丝投票估算器：基于分Era建模和两阶段估计"""
    
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
        
        # 赛季类型定义
        self.eras = {
            's1_2': (1, 2),           # Rank制早期
            's3_27': (3, 27),         # Percentage制时期  
            's28_plus': (28, 34)      # Rank制回归 + Twist
        }
        self.era_pipes = {}  # 存储分时代的模型
        self.train_data_by_era = {}
        
        # 扩展特征定义（包含新增特征）
        self.num_features = [
            "week", "judge_total", "judge_mean", "judge_count",
            "judge_total_delta", "cum_judge_mean", "cum_judge_sum",
            "roll3_mean", "roll3_std", "weekly_rank", "weekly_rank_pct",
            "judge_total_week_z", "age", "partner_experience", "is_us",
            # === 新增增强特征 ===
            "judge_pctile",           # 评委分百分位
            "judge_gap_to_top",       # 与第一名分差
            "judge_gap_to_safe",      # 与安全线分差
            "momentum",               # 近期动量
            "was_in_danger_last_week", # 上周是否危险
            "weeks_to_finale",        # 距离决赛周数
            "judge_score_normalized", # 标准化评委分
        ]
        
        self.cat_features = [
            "celebrity_industry", "industry_category", "age_group",
            "home_state", "home_country", "partner",
            # === 新增分类特征 ===
            "partner_experience_cat", # 舞伴经验分级
            "season_stage",           # 赛季阶段
        ]
    
    def load_data(self):
        """加载清洗后的宽表与长表数据"""
        print("=" * 60)
        print("PART 1: 增强版粉丝投票估算模型")
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
        """构建增强版动态特征"""
        self.long_df = self.long_df.sort_values(
            ["season", "celebrity_name", "week"]
        ).reset_index(drop=True)
        
        # 基础特征（保留原有）
        self.long_df["judge_total_prev"] = self.long_df.groupby(
            ["season", "celebrity_name"]
        )["judge_total"].shift(1)
        self.long_df["judge_total_delta"] = (
            self.long_df["judge_total"] - self.long_df["judge_total_prev"]
        )
        
        self.long_df["cum_judge_mean"] = (
            self.long_df.groupby(["season", "celebrity_name"])["judge_total"]
            .expanding().mean().reset_index(level=[0, 1], drop=True)
        )
        self.long_df["cum_judge_sum"] = self.long_df.groupby(
            ["season", "celebrity_name"]
        )["judge_total"].cumsum()

        self.long_df["roll3_mean"] = (
            self.long_df.groupby(["season", "celebrity_name"])["judge_total"]
            .rolling(3, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        )
        self.long_df["roll3_std"] = (
            self.long_df.groupby(["season", "celebrity_name"])["judge_total"]
            .rolling(3, min_periods=1).std().reset_index(level=[0, 1], drop=True)
        )

        if "weekly_rank" not in self.long_df.columns:
            self.long_df["weekly_rank"] = self.long_df.groupby(
                ["season", "week"]
            )["judge_total"].rank(ascending=False, method="min")
        if "weekly_rank_pct" not in self.long_df.columns:
            self.long_df["weekly_rank_pct"] = self.long_df.groupby(
                ["season", "week"]
            )["judge_total"].rank(ascending=False, pct=True)
        
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
        
        # ========================================
        # 新增增强特征
        # ========================================
        print(f"\n[Step 5-Enhanced] 构建增强特征...")
        
        # 1. 相对表现特征
        # 评委分百分位 (0-1之间，0表示当周最差)
        self.long_df['judge_pctile'] = self.long_df.groupby(['season', 'week'])['judge_total']\
                                      .transform(lambda x: (x.rank(method='min') - 1) / (len(x) - 1))
        
        # 标准化评委分 (Z-score的稳健版本)
        self.long_df['judge_score_normalized'] = self.long_df.groupby(['season', 'week'])['judge_total']\
                                                .transform(lambda x: (x - x.median()) / (x.std() + 1e-6))
        
        # 与顶部分差 (标准化到0-1)
        self.long_df['judge_gap_to_top'] = self.long_df.groupby(['season', 'week'])['judge_total']\
                                          .transform(lambda x: (x.max() - x) / (x.max() - x.min() + 1e-6))
        
        # 与安全线(第30百分位)的分差 - 负值表示在安全区
        self.long_df['judge_gap_to_safe'] = self.long_df.groupby(['season', 'week'])['judge_total']\
                                           .transform(lambda x: x.quantile(0.3) - x)
        
        # 2. 动量与趋势
        self.long_df['momentum'] = self.long_df.groupby(['season', 'celebrity_name'])['judge_total']\
                                    .transform(lambda x: x.diff().rolling(2, min_periods=1).mean())
        self.long_df['is_improving'] = (self.long_df['momentum'] > 0).astype(int)
        
        # 3. 舞伴经验分级
        self.long_df['partner_experience_cat'] = pd.cut(
            self.long_df['partner_experience'], 
            bins=[-1, 0, 2, 5, 20], 
            labels=['new', 'junior', 'senior', 'veteran']
        )
        
        # 4. 赛季阶段和距离决赛
        self.long_df['weeks_to_finale'] = self.long_df['season_weeks'] - self.long_df['week']
        self.long_df['season_stage'] = self.long_df.apply(
            lambda x: 'early' if x['week'] <= x['season_weeks'] * 0.3 
            else ('late' if x['week'] >= x['season_weeks'] * 0.8 else 'mid'), 
            axis=1
        )
        
        # 5. 危险区指示器 (上周排名bottom 2)
        self.long_df['prev_week_rank'] = self.long_df.groupby(['season', 'celebrity_name'])['weekly_rank'].shift(1)
        max_rank = self.long_df.groupby(['season', 'week'])['weekly_rank'].transform('max')
        self.long_df['was_in_danger_last_week'] = (self.long_df['prev_week_rank'] >= max_rank - 2).astype(int)
        
        # 填补新特征的缺失值
        for col in ['judge_pctile', 'judge_gap_to_top', 'judge_gap_to_safe', 
                    'momentum', 'judge_score_normalized']:
            if col in self.long_df.columns:
                self.long_df[col] = self.long_df[col].fillna(self.long_df[col].median())
        
        self.long_df['was_in_danger_last_week'] = self.long_df['was_in_danger_last_week'].fillna(0)
        self.long_df['partner_experience_cat'] = self.long_df['partner_experience_cat'].fillna('new')
        
        print(f"  - 新增特征: judge_pctile, judge_gap_to_top, momentum, season_stage等")
        
        # 填补基础缺失值
        num_fill_cols = list({c for c in self.num_features if c in self.long_df.columns})
        for col in num_fill_cols:
            mean_val = self.long_df[col].mean()
            self.long_df[col] = self.long_df[col].fillna(mean_val)
        cat_fill_cols = [c for c in self.cat_features if c in self.long_df.columns]
        for col in cat_fill_cols:
            if is_categorical_dtype(self.long_df[col]):
                if "Unknown" not in self.long_df[col].cat.categories:
                    self.long_df[col] = self.long_df[col].cat.add_categories(["Unknown"])
            self.long_df[col] = self.long_df[col].fillna("Unknown")
        
        print(f"\n[Step 5] 特征构造完成（含增强特征）")
        return self
    
    def prepare_training_data_by_era(self):
        """分Era准备训练数据"""
        print(f"\n[Step 6-Enhanced] 分Era构建训练集...")
        
        # 基础训练数据
        base_train = self.long_df[
            self.long_df["is_active"] & 
            (self.long_df["week"] < self.long_df["season_weeks"])
        ].copy()
        
        base_train["y_survive_next"] = 1 - base_train["is_elimination_week"].fillna(0).astype(int)
        
        # 按Era拆分
        for era_name, (start_season, end_season) in self.eras.items():
            mask = (base_train['season'] >= start_season) & (base_train['season'] <= end_season)
            era_data = base_train[mask].copy()
            
            if len(era_data) > 0:
                # 填补缺失
                for col in self.num_features:
                    if col in era_data.columns:
                        era_data[col] = era_data[col].fillna(era_data[col].median())
                for col in self.cat_features:
                    if col in era_data.columns:
                        if is_categorical_dtype(era_data[col]):
                            if "Unknown" not in era_data[col].cat.categories:
                                era_data[col] = era_data[col].cat.add_categories(["Unknown"])
                        era_data[col] = era_data[col].fillna("Unknown")
                
                self.train_data_by_era[era_name] = era_data
                print(f"  {era_name} (S{start_season}-{end_season}): {len(era_data)} 条训练记录")
        
        return self
    
    def train_model_by_era(self):
        """分赛季类型训练模型（关键改进）"""
        print(f"\n[Step 7-Enhanced] 分Era训练专用模型...")
        
        for era_name, train_df in self.train_data_by_era.items():
            print(f"\n  训练 {era_name} 模型...")
            
            # 根据Era特性调整模型参数
            if era_name == 's1_2':
                # Rank制时期：更注重排名特征，模型简单些
                n_estimators, max_depth, lr = 150, 4, 0.08
            elif era_name == 's3_27':
                # Percentage制：需要精确数值预测，模型复杂些
                n_estimators, max_depth, lr = 200, 6, 0.05
            else:  # s28_plus
                # Twist时期：复杂交互，最深模型
                n_estimators, max_depth, lr = 250, 7, 0.04
            
            preprocess = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), [c for c in self.num_features if c in train_df.columns]),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in self.cat_features if c in train_df.columns]),
                ],
                remainder="drop"
            )
            
            clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                random_state=42
            )
            
            pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
            
            X = train_df[[c for c in self.num_features + self.cat_features if c in train_df.columns]]
            y = train_df["y_survive_next"].astype(int)
            
            # 交叉验证评估
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(pipe, X, y, cv=3, scoring='roc_auc')
            print(f"    CV AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
            
            # 训练最终模型
            pipe.fit(X, y)
            self.era_pipes[era_name] = pipe
            
            # 输出Top3重要特征
            if hasattr(clf, 'feature_importances_'):
                feature_names = (pipe.named_steps['preprocess']
                               .get_feature_names_out()
                               .tolist())
                importances = clf.feature_importances_
                top3 = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:3]
                print(f"    Top3特征: {[f[0] for f in top3]}")
        
        return self
    
    def two_stage_vote_mapping(self, era_name, era_data):
        """
        两阶段估计第二阶段：约束优化反演票数
        目标：在满足淘汰结果约束下，估计最合理的票数分布
        """
        if len(era_data) == 0:
            return era_data
        
        print(f"    应用两阶段估计于 {era_name}...")
        
        # 按周处理
        result_frames = []
        
        for (season, week), group in era_data.groupby(['season', 'week']):
            n = len(group)
            if n <= 1:
                group['vote_share_hat'] = 1.0
                group['votes_hat'] = 1_000_000
                result_frames.append(group)
                continue
            
            judge_scores = group['judge_total'].values
            is_eliminated = group['is_elimination_week'].fillna(0).values
            
            # 根据赛制选择融合方式
            if era_name in ['s1_2', 's28_plus']:  # Rank制
                def combined_score(vote_shares):
                    judge_ranks = stats.rankdata(-judge_scores, method='min')
                    vote_ranks = stats.rankdata(-vote_shares, method='min')
                    return judge_ranks + vote_ranks
                is_rank_method = True
            else:  # Percentage制
                def combined_score(vote_shares):
                    judge_pct = judge_scores / (judge_scores.sum() + 1e-6)
                    vote_pct = vote_shares / (vote_shares.sum() + 1e-6)
                    alpha = 0.5  # 可学习参数，这里固定0.5
                    return alpha * judge_pct + (1-alpha) * vote_pct
                is_rank_method = False
            
            # 优化目标：最小化与"合理先验"的偏离 + 熵正则化
            # 先验：评委分越高，票数应该倾向于越高（但允许粉丝逆转）
            prior = judge_scores / judge_scores.sum()
            
            def objective(v):
                v = np.clip(v, 0.001, 0.999)
                v_norm = v / v.sum()
                # 偏离先验的L2惩罚（允许偏离，但惩罚过大偏离）
                prior_penalty = 0.5 * np.sum((v_norm - prior)**2)
                # 熵正则化（鼓励分布不要太极端）
                entropy = -np.sum(v_norm * np.log(v_norm + 1e-10))
                return prior_penalty - 0.1 * entropy  # 最大化熵 = 最小化负熵
            
            # 硬约束：被淘汰者的综合分必须是最低的
            def constraint(v):
                v_norm = np.clip(v, 0.001, 0.999)
                v_norm = v_norm / v_norm.sum()
                scores = combined_score(v_norm)
                
                elim_idx = np.where(is_eliminated == 1)[0]
                if len(elim_idx) == 0:
                    return 1.0  # 无淘汰，无约束
                
                elim_score = scores[elim_idx[0]]
                other_scores = scores[is_eliminated == 0]
                
                if len(other_scores) == 0:
                    return 1.0
                
                # 被淘汰者的分数必须严格小于其他所有人
                min_safe = np.min(other_scores)
                margin = min_safe - elim_score
                return margin  # 必须 >= 0
            
            # 求解优化问题
            x0 = prior.copy()  # 初始值：与评委分成正比
            
            cons = {'type': 'ineq', 'fun': constraint}
            bounds = [(0.001, 0.999) for _ in range(n)]
            
            try:
                result = minimize(objective, x0, method='SLSQP', 
                                bounds=bounds, constraints=cons,
                                options={'ftol': 1e-9, 'disp': False, 'maxiter': 1000})
                
                if result.success:
                    vote_shares = result.x / result.x.sum()
                else:
                    raise ValueError("Optimization failed")
                    
            except Exception as e:
                # 回退策略：基于评委分的幂律变换（给予低分选手更多粉丝同情分）
                # 这是为了处理"争议选手"（如Bobby Bones）
                inv_judge = 1.0 / (judge_scores + 1e-6)
                # 如果上周被淘汰，减少其票数
                if 'was_in_danger_last_week' in group.columns:
                    danger_factor = group['was_in_danger_last_week'].values
                    inv_judge = inv_judge * (0.5 + 0.5 * (1 - danger_factor))
                vote_shares = inv_judge / inv_judge.sum()
            
            group = group.copy()
            group['vote_share_hat'] = vote_shares
            
            # 总票数规模（根据 era 和 周次调整）
            base_votes = 1_000_000
            if era_name == 's28_plus':
                base_votes = 1_200_000  # Twist时期投票更激烈
            
            # 投票池随周次增长（决赛周投票最多）
            week_factor = 0.8 + 0.6 * (week - 1) / (group['season_weeks'].iloc[0] - 1 + 1e-6)
            group['total_votes_hat'] = base_votes * week_factor
            group['votes_hat'] = group['vote_share_hat'] * group['total_votes_hat']
            
            result_frames.append(group)
        
        return pd.concat(result_frames, ignore_index=True)
    
    def predict_votes_enhanced(self):
        """增强版票数预测：分Era + 两阶段估计"""
        print(f"\n[Step 8-Enhanced] 分Era票数预测...")
        
        all_results = []
        
        for era_name, pipe in self.era_pipes.items():
            era_range = self.eras[era_name]
            mask = (self.long_df['season'] >= era_range[0]) & (self.long_df['season'] <= era_range[1])
            era_data = self.long_df[mask].copy()
            
            if len(era_data) == 0:
                continue
            
            print(f"  处理 {era_name} (S{era_range[0]}-{era_range[1]}): {len(era_data)} 条记录")
            
            # 第一阶段：预测生存概率（排名代理）
            X = era_data[[c for c in self.num_features + self.cat_features if c in era_data.columns]]
            era_data['p_survive_next'] = pipe.predict_proba(X)[:, 1]
            
            # 第二阶段：约束优化反演为具体票数
            era_result = self.two_stage_vote_mapping(era_name, era_data)
            all_results.append(era_result)
        
        self.pred_df = pd.concat(all_results, ignore_index=True)
        
        # 构建结果表（与原始格式兼容）
        result_cols = [
            "season", "week", "celebrity_name", "partner",
            "celebrity_industry", "industry_category", "home_state", "home_country",
            "age", "age_group", "judge_total", "judge_mean", "judge_count",
            "p_survive_next", "vote_share_hat", "total_votes_hat", "votes_hat",
            "placement", "elimination_week", "judge_pctile", "momentum"  # 保留新特征供分析
        ]
        
        available_cols = [c for c in result_cols if c in self.pred_df.columns]
        self.result_df = self.pred_df[available_cols].sort_values(
            ["season", "week", "votes_hat"], ascending=[True, True, False]
        )
        
        print(f"\n[Step 8] 票数预测完成")
        print(f"  预测行数: {len(self.pred_df)}")
        print(f"  平均预测票数: {self.pred_df['votes_hat'].mean():,.0f}")
        return self
    
    def save_results(self, output_path: str = "q1_fan_vote_estimates_enhanced.csv"):
        """保存结果"""
        self.result_df.to_csv(output_path, index=False)
        print(f"\n[Step 9] 结果已保存至: {output_path}")
        return self
    
    def run(self):
        """运行增强版完整流程"""
        return (self.load_data()
                .build_features()           # 增强特征
                .prepare_training_data_by_era()  # 分Era准备
                .train_model_by_era()       # 分Era训练
                .predict_votes_enhanced()   # 分Era预测 + 两阶段估计
                .save_results())


# ============================================================
# PART 2: 淘汰一致性评估（保持原有，仅优化评估指标）
# ============================================================
class EliminationConsistencyEvaluator:
    """淘汰一致性评估器"""
    
    def __init__(self, estimator):
        self.estimator = estimator
        self.df = estimator.df
        self.result_df = estimator.result_df
        self.season_weeks = estimator.season_weeks
        self.truth_map = {}
        self.eval_week_df = None
        self.season_summary = None
        
        # 赛制规则配置（根据输入改进）
        self.max_season = int(self.result_df["season"].max())
        self.rank_seasons = set(range(1, 3)) | set(range(28, self.max_season + 1))
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
    
    def predict_elims_rank(self, g, k):
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
    
    def predict_elims_percent(self, g, k):
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
    
    def apply_twist(self, g, base_bottom2, k):
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
        
        # 新增：分Era的评估
        era_map = {**{s: 's1_2' for s in range(1,3)}, 
                  **{s: 's3_27' for s in range(3,28)},
                  **{s: 's28_plus' for s in range(28,35)}}
        self.eval_week_df['era'] = self.eval_week_df['season'].map(era_map)
        era_metrics = self.eval_week_df.groupby('era')['exact_match'].mean()
        print(f"\n  分Era精确匹配率：")
        for era, rate in era_metrics.items():
            print(f"    {era}: {rate:.4f}")
        
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
        
        print(f"\n[Metrics] 赛季级一致性汇总（前5行）：")
        print(self.season_summary.head().to_string())
        return self
    
    def plot_consistency(self):
        """绘制一致性图表"""
        # 图1: 赛季级淘汰一致率柱状图
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
        bars = ax.bar(self.season_summary["season"].astype(str), 
               self.season_summary["exact_elim"], color=COLORS["primary"], edgecolor='navy', alpha=0.85)
        
        # 高亮不同Era
        for i, (bar, season) in enumerate(zip(bars, self.season_summary["season"])):
            if season <= 2:
                bar.set_color(COLORS["accent"])  # S1-2
            elif season >= 28:
                bar.set_color(COLORS["success"])  # S28+
        
        ax.axhline(y=self.season_summary["exact_elim"].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Average: {self.season_summary["exact_elim"].mean():.3f}')
        ax.set_xlabel("Season", fontsize=12)
        ax.set_ylabel("Exact Match Rate", fontsize=12)
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig1_season_consistency_enhanced.png", dpi=300, bbox_inches='tight')
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
# PART 3: 不确定性度量分析（保持原有）
# ============================================================
class UncertaintyAnalyzer:
    """不确定性分析器：Bootstrap方法"""
    
    def __init__(self, estimator, B: int = 60, sigma_pool: float = 0.18):
        self.estimator = estimator
        self.B = B
        self.SIGMA_POOL = sigma_pool
        self.rng = np.random.default_rng(RANDOM_SEED)
        
        self.unc_df = None
        self.votes_boot = None
    
    def prepare_data(self):
        """准备不确定性分析数据"""
        print("\n" + "=" * 60)
        print("PART 3: 不确定性度量分析")
        print("=" * 60)
        
        self.unc_df = self.estimator.pred_df.copy()
        print(f"\n[Step 3-1] 数据准备完成: {len(self.unc_df)} 条记录")
        return self
    
    def run_bootstrap(self):
        """运行Bootstrap（简化版）"""
        print(f"\n[Step 3-2] 简化Bootstrap (B={self.B})...")
        
        n_samples = len(self.unc_df)
        self.votes_boot = np.zeros((self.B, n_samples), dtype=float)
        
        # 获取基础预测
        base_votes = self.unc_df['votes_hat'].values
        
        for b in range(self.B):
            # 添加对数正态噪声模拟不确定性
            noise = self.rng.lognormal(mean=0, sigma=self.SIGMA_POOL, size=n_samples)
            self.votes_boot[b, :] = base_votes * noise
            
            if (b + 1) % 20 == 0:
                print(f"  Bootstrap {b+1}/{self.B} 完成")
        
        print(f"\n[Step 3-2] Bootstrap完成")
        return self
    
    def compute_uncertainty_metrics(self):
        """计算不确定性指标"""
        self.unc_df["votes_q10"] = np.quantile(self.votes_boot, 0.10, axis=0)
        self.unc_df["votes_q50"] = np.quantile(self.votes_boot, 0.50, axis=0)
        self.unc_df["votes_q90"] = np.quantile(self.votes_boot, 0.90, axis=0)
        self.unc_df["votes_std"] = self.votes_boot.std(axis=0)
        
        self.unc_df["ci80_width"] = self.unc_df["votes_q90"] - self.unc_df["votes_q10"]
        self.unc_df["rel_ci80"] = self.unc_df["ci80_width"] / np.where(
            self.unc_df["votes_q50"] > 0, self.unc_df["votes_q50"], np.nan
        )
        
        print(f"\n[Step 3-3] 不确定性指标完成")
        print(f"  平均相对不确定性(rel_ci80): {self.unc_df['rel_ci80'].mean():.4f}")
        return self
    
    def plot_uncertainty(self):
        """绘制不确定性图表"""
        # 图: rel_ci80整体分布
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax.hist(self.unc_df["rel_ci80"].dropna(), bins=40, edgecolor="black", 
                color=COLORS["secondary"], alpha=0.7)
        ax.axvline(x=self.unc_df["rel_ci80"].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.unc_df["rel_ci80"].mean():.3f}')
        ax.set_xlabel("Relative CI80 Width", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig4_uncertainty_distribution_enhanced.png", dpi=300, bbox_inches='tight')
        plt.savefig(OUTPUT_DIR / "fig4_uncertainty_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 图5: 不确定性随周变化
        week_summary = (
            self.unc_df.groupby("week", as_index=False)
            .agg(rel_ci80_mean=("rel_ci80", "mean"), rel_ci80_std=("rel_ci80", "std"))
        )

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax.plot(week_summary["week"], week_summary["rel_ci80_mean"], 
                marker="o", linewidth=2, markersize=8, color='teal')
        ax.fill_between(
            week_summary["week"],
            week_summary["rel_ci80_mean"] - week_summary["rel_ci80_std"],
            week_summary["rel_ci80_mean"] + week_summary["rel_ci80_std"],
            alpha=0.25, color='teal', label="±1 Std"
        )
        ax.set_xlabel("Week", fontsize=12)
        ax.set_ylabel("Relative CI80 Width", fontsize=12)
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig5_uncertainty_by_week.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 图6: 不确定性最高的选手
        topk = 15
        cele_unc = (
            self.unc_df.groupby("celebrity_name", as_index=False)
            .agg(rel_ci80_mean=("rel_ci80", "mean"))
            .sort_values("rel_ci80_mean", ascending=False)
        )
        top_cele = cele_unc.head(topk)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        ax.bar(range(topk), top_cele["rel_ci80_mean"], color='coral', edgecolor='darkred', alpha=0.8)
        ax.set_xticks(range(topk))
        ax.set_xticklabels(top_cele["celebrity_name"], rotation=45, ha="right", fontsize=9)
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
        heat_cmap = LinearSegmentedColormap.from_list(
            "dwts_theme", [COLORS["light"], COLORS["primary"], COLORS["secondary"], COLORS["success"]]
        )
        im = ax.imshow(heat_u.values, aspect="auto", cmap=heat_cmap)
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
        ax.scatter(scatter_df["pred_total_votes"], scatter_df["placement"], 
                   alpha=0.5, c='steelblue', edgecolor='navy', s=50)
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
        
        print(f"\n[Plots] 不确定性图表已保存")
        return self
    
    def run(self):
        """运行完整分析"""
        return (self.prepare_data()
                .run_bootstrap()
                .compute_uncertainty_metrics()
                .plot_uncertainty())


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("MCM 2026 Problem C - Question 1: 增强版解决方案")
    print("改进：分Era建模 + 增强特征工程 + 两阶段估计")
    print("=" * 70)
    
    # Part 1: 增强版粉丝投票估算
    estimator = FanVoteEstimator()  # 已经是增强版
    estimator.run()
    
    # Part 2: 淘汰一致性评估
    evaluator = EliminationConsistencyEvaluator(estimator)
    evaluator.run()
    
    # Part 3: 不确定性分析
    analyzer = UncertaintyAnalyzer(estimator, B=60, sigma_pool=0.18)
    analyzer.run()
    
    # 汇总报告
    print("\n" + "=" * 70)
    print("完整分析报告（增强版）")
    print("=" * 70)
    
    print("\n【Part 1 - 粉丝投票估算（增强版）】")
    print(f"  - 预测样本数: {len(estimator.pred_df)}")
    print(f"  - 平均预测票数: {estimator.pred_df['votes_hat'].mean():,.0f}")
    print(f"  - 分Era建模: S1-2 (Rank), S3-27 (Percentage), S28+ (Rank+Twist)")
    print(f"  - 新增特征: judge_pctile, momentum, season_stage等")
    
    print("\n【Part 2 - 淘汰一致性评估】")
    mask_elim = evaluator.eval_week_df["true_k"] > 0
    print(f"  - 淘汰周精确匹配率: {evaluator.eval_week_df.loc[mask_elim, 'exact_match'].mean():.4f}")
    print(f"  - Bottom-2覆盖率: {evaluator.eval_week_df.loc[mask_elim, 'bottom2_cover_true'].mean():.4f}")
    
    print("\n【Part 3 - 不确定性度量】")
    print(f"  - 平均相对不确定性: {analyzer.unc_df['rel_ci80'].mean():.4f}")
    
    print(f"\n【输出文件】")
    print(f"  - q1_fan_vote_estimates_enhanced.csv (增强版票数估算)")
    print(f"  - {OUTPUT_DIR}/ (所有可视化图表)")
    
    return estimator, evaluator, analyzer


if __name__ == "__main__":
    estimator, evaluator, analyzer = main()