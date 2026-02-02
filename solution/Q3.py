# ============================================================
# MCM 2026 Problem C - Question 3: 特征影响分析
# 包含三个主要部分：
#   Part 1: 特征工程与XGBoost建模
#   Part 2: SHAP归因分析（评委得分 vs 粉丝投票）
#   Part 3: 偏好差异对比与可视化
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: shap库未安装，将使用XGBoost内置特征重要性代替SHAP分析")

warnings.filterwarnings('ignore')

# -----------------------------
# 全局配置 & 可视化风格
# -----------------------------
np.random.seed(2026)
RANDOM_SEED = 2026

# 主题配色 - 马卡龙风格（低饱和度柔和色调）
COLORS = {
    "primary": "#7BADDF",      # 浅蓝
    "secondary": "#B581B4",    # 薰衣草紫
    "accent": "#EAB170",       # 暖橙
    "success": "#DA8176",      # 珊瑚粉
    "neutral": "#B1A8D3",      # 淡紫
    "light": "#BADDF3",        # 极浅蓝
    "dark": "#4A5568",         # 深灰
    "judge": "#EAB170",        # 评委颜色（暖橙）
    "fan": "#7BADDF",          # 粉丝颜色（浅蓝）
}

# 扩展调色板
PALETTE = [
    '#BADDF3', '#C8C3E1', '#B581B4', '#B1A8D3', '#B5C3EA', 
    '#7FBDB0', '#F4E09B', '#EAB170', '#DA8176', '#7BADDF'
]

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
OUTPUT_DIR = Path("plots/q3_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUTPUT_DIR = Path("outputs/q3")
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PART 1: 特征工程与数据准备
# ============================================================
class FeatureEngineer:
    """特征工程处理器"""
    
    def __init__(self, 
                 long_data_path: str = "dwts_long_format.csv",
                 vote_data_path: str = "q1_fan_vote_estimates_enhanced.csv",
                 cleaned_data_path: str = "dwts_cleaned.csv"):
        self.long_data_path = long_data_path
        self.vote_data_path = vote_data_path
        self.cleaned_data_path = cleaned_data_path
        self.df = None
        self.feature_names = []
        self.cat_encoders = {}
        
    def load_data(self):
        """加载并合并数据"""
        print("=" * 60)
        print("PART 1: 特征工程与数据准备")
        print("=" * 60)
        
        # 加载长格式数据
        self.df = pd.read_csv(self.long_data_path)
        
        # 加载粉丝投票预估数据
        vote_df = pd.read_csv(self.vote_data_path)
        
        # 合并粉丝投票数据
        if "votes_hat" not in self.df.columns:
            merge_cols = ["season", "week", "celebrity_name"]
            vote_cols = merge_cols + ["votes_hat", "vote_share_hat"]
            available_cols = [c for c in vote_cols if c in vote_df.columns]
            self.df = self.df.merge(
                vote_df[available_cols],
                on=merge_cols,
                how="left"
            )
        
        print(f"\n[Step 1] 数据加载完成")
        print(f"  数据形状: {self.df.shape}")
        print(f"  赛季范围: {self.df['season'].min()} - {self.df['season'].max()}")
        return self
    
    def create_partner_features(self):
        """创建舞伴相关特征"""
        # 舞伴历史平均排名（反映舞伴实力）
        partner_stats = self.df.groupby("partner").agg({
            "placement": ["mean", "min", "count"],
            "judge_total": "mean"
        }).reset_index()
        partner_stats.columns = [
            "partner", "partner_avg_placement", "partner_best_placement",
            "partner_total_contestants", "partner_avg_judge_score"
        ]
        
        # 舞伴历史参与季数
        partner_seasons = self.df.groupby("partner")["season"].nunique().reset_index()
        partner_seasons.columns = ["partner", "partner_seasons_count"]
        
        partner_stats = partner_stats.merge(partner_seasons, on="partner", how="left")
        
        # 合并舞伴特征
        self.df = self.df.merge(partner_stats, on="partner", how="left")
        
        # 计算舞伴在当前赛季之前的经验
        def get_prior_experience(row):
            prior_seasons = self.df[
                (self.df["partner"] == row["partner"]) & 
                (self.df["season"] < row["season"])
            ]["season"].nunique()
            return prior_seasons
        
        # 使用更高效的方式计算
        partner_first_season = self.df.groupby("partner")["season"].min().to_dict()
        self.df["partner_first_season"] = self.df["partner"].map(partner_first_season)
        self.df["partner_prev_seasons"] = self.df["season"] - self.df["partner_first_season"]
        self.df["partner_prev_seasons"] = self.df["partner_prev_seasons"].clip(lower=0)
        
        # 舞伴人气分类（基于历史表现）
        placement_q33 = partner_stats["partner_avg_placement"].quantile(0.33)
        placement_q67 = partner_stats["partner_avg_placement"].quantile(0.67)
        
        def classify_partner_popularity(avg_placement):
            if pd.isna(avg_placement):
                return 0  # 新人
            if avg_placement <= placement_q33:
                return 2  # 高人气（排名靠前）
            elif avg_placement <= placement_q67:
                return 1  # 普通
            else:
                return 0  # 低人气
        
        self.df["partner_popularity"] = self.df["partner_avg_placement"].apply(
            classify_partner_popularity
        )
        
        # 舞伴历史平均排名（用于回归）
        self.df["partner_historical_avg_placement"] = self.df["partner_avg_placement"]
        
        print(f"\n[Step 2] 舞伴特征创建完成")
        print(f"  唯一舞伴数: {self.df['partner'].nunique()}")
        return self
    
    def create_celebrity_features(self):
        """创建明星特征"""
        # 年龄标准化
        age_mean = self.df["age"].mean()
        age_std = self.df["age"].std()
        self.df["age_normalized"] = (self.df["age"] - age_mean) / age_std
        
        # 年龄分组（如果不存在）
        if "age_group" not in self.df.columns:
            self.df["age_group"] = pd.cut(
                self.df["age"],
                bins=[0, 25, 35, 45, 55, 100],
                labels=["≤25", "26-35", "36-45", "46-55", "55+"]
            )
        
        # 行业分类（如果不存在）
        if "industry_category" not in self.df.columns:
            industry_mapping = {
                "Athlete": "Sports",
                "Actor/Actress": "Entertainment",
                "Singer/Rapper": "Entertainment",
                "TV Personality": "Media",
                "Model": "Media",
                "Social Media Personality": "Media",
                "Olympian": "Sports",
                "Dancer": "Entertainment",
                "Musician": "Entertainment",
            }
            self.df["industry_category"] = self.df["celebrity_industry"].map(
                lambda x: industry_mapping.get(x, "Other")
            )
        
        # 是否美国选手
        if "is_us" not in self.df.columns:
            self.df["is_us"] = (self.df["home_country"] == "United States").astype(int)
        self.df["is_us_contestant"] = self.df["is_us"]
        
        print(f"\n[Step 3] 明星特征创建完成")
        print(f"  行业分类: {self.df['industry_category'].unique().tolist()}")
        print(f"  年龄分组: {self.df['age_group'].unique().tolist()}")
        return self
    
    def create_interaction_features(self):
        """创建交互特征"""
        # 年龄×行业交互（使用标准化年龄）
        # 这里不直接做one-hot展开，让XGBoost自动处理
        
        # 舞伴人气×行业交互
        # 同样让XGBoost处理
        
        # 赛季归一化（捕捉时间趋势）
        max_season = self.df["season"].max()
        self.df["season_normalized"] = self.df["season"] / max_season
        
        # 周次进度（当周/总周数）
        season_max_week = self.df.groupby("season")["week"].transform("max")
        self.df["week_progress"] = self.df["week"] / season_max_week
        
        print(f"\n[Step 4] 交互特征创建完成")
        return self
    
    def encode_categorical_features(self):
        """编码分类特征"""
        cat_features = ["industry_category", "age_group"]
        
        for feat in cat_features:
            if feat in self.df.columns:
                le = LabelEncoder()
                self.df[f"{feat}_encoded"] = le.fit_transform(
                    self.df[feat].astype(str)
                )
                self.cat_encoders[feat] = le
        
        print(f"\n[Step 5] 分类特征编码完成")
        return self
    
    def prepare_modeling_data(self):
        """准备建模数据"""
        # 定义特征列
        self.feature_names = [
            "week",                              # 周次
            "season",                            # 赛季
            "age",                               # 年龄
            "partner_prev_seasons",              # 舞伴历史经验
            "partner_historical_avg_placement",  # 舞伴历史平均排名
            "industry_category_encoded",         # 行业分类
            "is_us_contestant",                  # 是否美国选手
        ]
        
        # 确保所有特征都存在
        available_features = [f for f in self.feature_names if f in self.df.columns]
        
        # 填充缺失值
        for feat in available_features:
            if self.df[feat].isna().any():
                if self.df[feat].dtype in ["float64", "int64"]:
                    self.df[feat] = self.df[feat].fillna(self.df[feat].median())
                else:
                    self.df[feat] = self.df[feat].fillna(0)
        
        # 定义目标变量
        self.df["target_judge"] = self.df["judge_total"]
        self.df["target_fan"] = self.df.get("votes_hat", self.df.get("vote_share_hat", 0))
        
        # 过滤有效数据
        valid_mask = (
            self.df["target_judge"].notna() & 
            self.df["target_fan"].notna() &
            (self.df["target_fan"] > 0)
        )
        self.df = self.df[valid_mask].copy()
        
        self.feature_names = available_features
        
        print(f"\n[Step 6] 建模数据准备完成")
        print(f"  特征数量: {len(self.feature_names)}")
        print(f"  特征列表: {self.feature_names}")
        print(f"  有效样本数: {len(self.df)}")
        return self
    
    def run(self):
        """运行完整特征工程"""
        return (self.load_data()
                .create_partner_features()
                .create_celebrity_features()
                .create_interaction_features()
                .encode_categorical_features()
                .prepare_modeling_data())


# ============================================================
# PART 2: XGBoost建模与RankSHAP分析
# ============================================================
class XGBoostRankSHAPAnalyzer:
    """XGBoost模型与RankSHAP归因分析
    
    核心改进（相比原始TreeSHAP）：
    1. 使用NDCG作为价值函数，衡量排名质量而非预测误差
    2. 基于排名的特征归因，更适合比赛排名预测场景
    3. 按赛季-周次分组计算，保持排名的局部性
    """
    
    def __init__(self, feature_engineer: FeatureEngineer):
        self.fe = feature_engineer
        self.df = feature_engineer.df
        self.feature_names = feature_engineer.feature_names
        
        self.model_judge = None
        self.model_fan = None
        self.shap_values_judge = None
        self.shap_values_fan = None
        self.rankshap_judge = None  # RankSHAP归因值
        self.rankshap_fan = None
        self.X_train = None
        self.X_test = None
        self.y_train_judge = None
        self.y_train_fan = None
        self.y_test_judge = None
        self.y_test_fan = None
        
        self.importance_judge = None
        self.importance_fan = None
        self.metrics = {}

        
    def prepare_train_test_split(self, test_size: float = 0.3):
        """按选手分组划分训练集和测试集"""
        print("\n" + "=" * 60)
        print("PART 2: XGBoost建模与SHAP分析")
        print("=" * 60)
        
        # 获取唯一选手-赛季组合
        self.df["contestant_key"] = (
            self.df["season"].astype(str) + "_" + 
            self.df["celebrity_name"].astype(str)
        )
        unique_contestants = self.df["contestant_key"].unique()
        
        # 随机划分
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(unique_contestants)
        
        n_test = int(len(unique_contestants) * test_size)
        test_contestants = set(unique_contestants[:n_test])
        train_contestants = set(unique_contestants[n_test:])
        
        train_mask = self.df["contestant_key"].isin(train_contestants)
        test_mask = self.df["contestant_key"].isin(test_contestants)
        
        self.X_train = self.df.loc[train_mask, self.feature_names].values
        self.X_test = self.df.loc[test_mask, self.feature_names].values
        
        self.y_train_judge = self.df.loc[train_mask, "target_judge"].values
        self.y_test_judge = self.df.loc[test_mask, "target_judge"].values
        
        self.y_train_fan = self.df.loc[train_mask, "target_fan"].values
        self.y_test_fan = self.df.loc[test_mask, "target_fan"].values
        
        print(f"\n[Step 1] 数据划分完成")
        print(f"  训练集: {len(self.X_train)} 样本")
        print(f"  测试集: {len(self.X_test)} 样本")
        return self
    
    def train_xgboost_models(self, do_grid_search: bool = False):
        """训练XGBoost模型"""
        # 基础参数
        base_params = {
            "objective": "reg:squarederror",
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_lambda": 1.0,
            "random_state": RANDOM_SEED,
            "n_jobs": -1
        }
        
        if do_grid_search:
            print("\n[Step 2] 执行网格搜索...")
            param_grid = {
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [100, 200, 300],
                "gamma": [0.01, 0.1, 1.0]
            }
            
            base_model = xgb.XGBRegressor(**base_params)
            gkf = GroupKFold(n_splits=5)
            groups = self.df.loc[
                self.df["contestant_key"].isin(
                    self.df[self.df["contestant_key"].isin(
                        set(self.df["contestant_key"].unique()[int(len(self.df["contestant_key"].unique()) * 0.3):])
                    )]["contestant_key"]
                ),
                "season"
            ].values
            
            # 只对评委分数做grid search（加速）
            grid_search = GridSearchCV(
                base_model, param_grid, 
                cv=3, scoring="r2", n_jobs=-1
            )
            grid_search.fit(self.X_train, self.y_train_judge)
            best_params = grid_search.best_params_
            print(f"  最优参数: {best_params}")
            
            for key, value in best_params.items():
                base_params[key] = value
        
        # 训练评委得分模型
        print("\n[Step 2] 训练评委得分模型...")
        self.model_judge = xgb.XGBRegressor(**base_params)
        self.model_judge.fit(self.X_train, self.y_train_judge)
        
        # 评估评委模型
        pred_judge_train = self.model_judge.predict(self.X_train)
        pred_judge_test = self.model_judge.predict(self.X_test)
        
        self.metrics["judge"] = {
            "r2_train": r2_score(self.y_train_judge, pred_judge_train),
            "r2_test": r2_score(self.y_test_judge, pred_judge_test),
            "mae_test": mean_absolute_error(self.y_test_judge, pred_judge_test),
            "rmse_test": np.sqrt(mean_squared_error(self.y_test_judge, pred_judge_test))
        }
        
        print(f"  训练R²: {self.metrics['judge']['r2_train']:.4f}")
        print(f"  测试R²: {self.metrics['judge']['r2_test']:.4f}")
        print(f"  测试MAE: {self.metrics['judge']['mae_test']:.4f}")
        
        # 训练粉丝投票模型
        print("\n[Step 3] 训练粉丝投票模型...")
        self.model_fan = xgb.XGBRegressor(**base_params)
        self.model_fan.fit(self.X_train, self.y_train_fan)
        
        # 评估粉丝模型
        pred_fan_train = self.model_fan.predict(self.X_train)
        pred_fan_test = self.model_fan.predict(self.X_test)
        
        self.metrics["fan"] = {
            "r2_train": r2_score(self.y_train_fan, pred_fan_train),
            "r2_test": r2_score(self.y_test_fan, pred_fan_test),
            "mae_test": mean_absolute_error(self.y_test_fan, pred_fan_test),
            "rmse_test": np.sqrt(mean_squared_error(self.y_test_fan, pred_fan_test))
        }
        
        print(f"  训练R²: {self.metrics['fan']['r2_train']:.4f}")
        print(f"  测试R²: {self.metrics['fan']['r2_test']:.4f}")
        print(f"  测试MAE: {self.metrics['fan']['mae_test']:.4f}")
        
        return self
    
    def compute_feature_importance(self):
        """计算特征重要性（使用SHAP或内置方法）"""
        print("\n[Step 4] 计算特征重要性...")
        
        if SHAP_AVAILABLE:
            print("  使用SHAP进行特征归因分析...")
            
            # 使用测试集的一部分进行SHAP分析
            sample_size = min(500, len(self.X_test))
            sample_idx = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test[sample_idx]
            
            # 评委模型SHAP
            explainer_judge = shap.TreeExplainer(self.model_judge)
            self.shap_values_judge = explainer_judge.shap_values(X_sample)
            
            # 粉丝模型SHAP
            explainer_fan = shap.TreeExplainer(self.model_fan)
            self.shap_values_fan = explainer_fan.shap_values(X_sample)
            
            # 计算全局重要性（|SHAP|均值）
            self.importance_judge = pd.DataFrame({
                "feature": self.feature_names,
                "importance": np.abs(self.shap_values_judge).mean(axis=0)
            }).sort_values("importance", ascending=False)
            
            self.importance_fan = pd.DataFrame({
                "feature": self.feature_names,
                "importance": np.abs(self.shap_values_fan).mean(axis=0)
            }).sort_values("importance", ascending=False)
            
            self.X_sample = X_sample
            
        else:
            print("  使用XGBoost内置特征重要性...")
            
            # 使用gain作为重要性度量
            self.importance_judge = pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.model_judge.feature_importances_
            }).sort_values("importance", ascending=False)
            
            self.importance_fan = pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.model_fan.feature_importances_
            }).sort_values("importance", ascending=False)
        
        # 归一化重要性
        self.importance_judge["importance_norm"] = (
            self.importance_judge["importance"] / 
            self.importance_judge["importance"].sum()
        )
        self.importance_fan["importance_norm"] = (
            self.importance_fan["importance"] / 
            self.importance_fan["importance"].sum()
        )
        
        print(f"\n  评委模型Top-5特征:")
        for _, row in self.importance_judge.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance_norm']:.4f}")
        
        print(f"\n  粉丝模型Top-5特征:")
        for _, row in self.importance_fan.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance_norm']:.4f}")
        
        return self
    
    def compute_ndcg(self, relevance: np.ndarray, order_scores: np.ndarray, k: int = None) -> float:
        """计算NDCG（归一化折损累计增益）
        
        Args:
            relevance: 真实相关性分数（越高越好）
            order_scores: 用于排序的分数（越高越排前）
            k: 截断位置，默认为全部
        
        Returns:
            NDCG score in [0, 1]
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
    
    def compute_rankshap_for_group(self, g: pd.DataFrame, target_col: str, model) -> dict:
        """计算单个赛季-周次组的RankSHAP归因值
        
        基于Q2的方差敏感性分析方法，扩展到多特征场景：
        - 计算每个特征对预测排名质量(NDCG)的边际贡献
        - 使用排列重要性作为Shapley值的近似
        
        Args:
            g: 单个周次的数据（包含多个选手）
            target_col: 目标列名 ('target_judge' 或 'target_fan')
            model: 训练好的XGBoost模型
        
        Returns:
            dict: 各特征的RankSHAP归因值
        """
        n = len(g)
        if n < 2:
            return {feat: 0.0 for feat in self.feature_names}
        
        X = g[self.feature_names].values
        y_true = g[target_col].values
        
        # 基准NDCG：使用完整特征的预测
        y_pred_base = model.predict(X)
        
        # 用实际值作为relevance，预测值作为ranking score
        ndcg_base = self.compute_ndcg(y_true, y_pred_base)
        
        # 计算每个特征的边际贡献（排列重要性）
        rankshap_values = {}
        n_permutations = 10  # 排列次数
        
        for feat_idx, feat_name in enumerate(self.feature_names):
            ndcg_drops = []
            
            for _ in range(n_permutations):
                # 打乱该特征的值
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, feat_idx])
                
                # 计算打乱后的NDCG
                y_pred_permuted = model.predict(X_permuted)
                ndcg_permuted = self.compute_ndcg(y_true, y_pred_permuted)
                
                # NDCG下降 = 该特征的贡献
                ndcg_drops.append(ndcg_base - ndcg_permuted)
            
            # 平均边际贡献作为RankSHAP值
            rankshap_values[feat_name] = np.mean(ndcg_drops)
        
        # 添加元信息
        rankshap_values["_ndcg_base"] = ndcg_base
        rankshap_values["_n_samples"] = n
        
        return rankshap_values
    
    def compute_rankshap(self):
        """计算全局RankSHAP归因值（基于NDCG）"""
        print("\n[Step 4.5] 计算RankSHAP归因值（基于NDCG）...")
        
        # 按赛季-周次分组计算
        self.df["season_week"] = (
            self.df["season"].astype(str) + "_" + 
            self.df["week"].astype(str)
        )
        
        # 评委模型RankSHAP
        print("  计算评委模型RankSHAP...")
        rankshap_judge_list = []
        for sw, g in self.df.groupby("season_week"):
            if len(g) >= 3:  # 至少3个选手才有意义
                rs = self.compute_rankshap_for_group(g, "target_judge", self.model_judge)
                rs["season_week"] = sw
                rankshap_judge_list.append(rs)
        
        self.rankshap_judge_raw = pd.DataFrame(rankshap_judge_list)
        
        # 聚合全局RankSHAP
        self.rankshap_judge = pd.DataFrame({
            "feature": self.feature_names,
            "rankshap": [
                self.rankshap_judge_raw[f].mean() 
                for f in self.feature_names
            ],
            "rankshap_std": [
                self.rankshap_judge_raw[f].std() 
                for f in self.feature_names
            ]
        }).sort_values("rankshap", ascending=False)
        
        # 归一化（相对贡献）
        total_rs = self.rankshap_judge["rankshap"].abs().sum()
        self.rankshap_judge["rankshap_norm"] = (
            self.rankshap_judge["rankshap"].abs() / total_rs if total_rs > 0 else 0
        )
        
        # 粉丝模型RankSHAP
        print("  计算粉丝模型RankSHAP...")
        rankshap_fan_list = []
        for sw, g in self.df.groupby("season_week"):
            if len(g) >= 3:
                rs = self.compute_rankshap_for_group(g, "target_fan", self.model_fan)
                rs["season_week"] = sw
                rankshap_fan_list.append(rs)
        
        self.rankshap_fan_raw = pd.DataFrame(rankshap_fan_list)
        
        # 聚合全局RankSHAP
        self.rankshap_fan = pd.DataFrame({
            "feature": self.feature_names,
            "rankshap": [
                self.rankshap_fan_raw[f].mean() 
                for f in self.feature_names
            ],
            "rankshap_std": [
                self.rankshap_fan_raw[f].std() 
                for f in self.feature_names
            ]
        }).sort_values("rankshap", ascending=False)
        
        # 归一化
        total_rs = self.rankshap_fan["rankshap"].abs().sum()
        self.rankshap_fan["rankshap_norm"] = (
            self.rankshap_fan["rankshap"].abs() / total_rs if total_rs > 0 else 0
        )
        
        # 计算平均NDCG
        avg_ndcg_judge = self.rankshap_judge_raw["_ndcg_base"].mean()
        avg_ndcg_fan = self.rankshap_fan_raw["_ndcg_base"].mean()
        
        print(f"\n  评委模型平均NDCG: {avg_ndcg_judge:.4f}")
        print(f"  粉丝模型平均NDCG: {avg_ndcg_fan:.4f}")
        
        print(f"\n  评委模型Top-5 RankSHAP:")
        for _, row in self.rankshap_judge.head(5).iterrows():
            print(f"    {row['feature']}: {row['rankshap']:.6f} (±{row['rankshap_std']:.6f})")
        
        print(f"\n  粉丝模型Top-5 RankSHAP:")
        for _, row in self.rankshap_fan.head(5).iterrows():
            print(f"    {row['feature']}: {row['rankshap']:.6f} (±{row['rankshap_std']:.6f})")
        
        # 保存NDCG指标
        self.metrics["ndcg_judge"] = avg_ndcg_judge
        self.metrics["ndcg_fan"] = avg_ndcg_fan
        
        return self
    
    def compute_rankshap_ratio(self):
        """计算评委vs粉丝的RankSHAP比例"""
        print("\n[Step 5.5] 计算RankSHAP偏好差异...")
        
        # 合并两个RankSHAP表
        rankshap_merged = self.rankshap_judge.merge(
            self.rankshap_fan,
            on="feature",
            suffixes=("_judge", "_fan")
        )
        
        # 计算比例 (粉丝/评委)
        rankshap_merged["ratio"] = (
            rankshap_merged["rankshap_norm_fan"] / 
            rankshap_merged["rankshap_norm_judge"].replace(0, np.nan)
        )
        
        # 计算差异 (粉丝 - 评委)
        rankshap_merged["diff"] = (
            rankshap_merged["rankshap_norm_fan"] - 
            rankshap_merged["rankshap_norm_judge"]
        )
        
        self.rankshap_ratio = rankshap_merged.sort_values(
            "rankshap_norm_judge", ascending=False
        )
        
        print(f"\n  RankSHAP偏好差异分析:")
        print(f"  {'特征':<35} {'评委':<10} {'粉丝':<10} {'比例':<10} {'解读'}")
        print("  " + "-" * 80)
        
        for _, row in self.rankshap_ratio.iterrows():
            ratio = row["ratio"]
            if pd.isna(ratio):
                interpretation = "N/A"
            elif ratio > 1.5:
                interpretation = "粉丝更敏感"
            elif ratio < 0.67:
                interpretation = "评委更敏感"
            else:
                interpretation = "影响相近"
            
            print(f"  {row['feature']:<35} {row['rankshap_norm_judge']:.4f}    "
                  f"{row['rankshap_norm_fan']:.4f}    {ratio:.2f}       {interpretation}")
        
        return self
    
    def compute_importance_ratio(self):
        """计算评委vs粉丝的重要性比例"""
        print("\n[Step 5] 计算偏好差异...")
        
        # 合并两个重要性表
        importance_merged = self.importance_judge.merge(
            self.importance_fan,
            on="feature",
            suffixes=("_judge", "_fan")
        )
        
        # 计算比例 (粉丝/评委)
        importance_merged["ratio"] = (
            importance_merged["importance_norm_fan"] / 
            importance_merged["importance_norm_judge"].replace(0, np.nan)
        )
        
        # 计算差异 (粉丝 - 评委)
        importance_merged["diff"] = (
            importance_merged["importance_norm_fan"] - 
            importance_merged["importance_norm_judge"]
        )
        
        self.importance_ratio = importance_merged.sort_values(
            "importance_norm_judge", ascending=False
        )
        
        print(f"\n  偏好差异分析:")
        print(f"  {'特征':<35} {'评委':<10} {'粉丝':<10} {'比例':<10} {'解读'}")
        print("  " + "-" * 80)
        
        for _, row in self.importance_ratio.iterrows():
            ratio = row["ratio"]
            if pd.isna(ratio):
                interpretation = "N/A"
            elif ratio > 1.5:
                interpretation = "粉丝更敏感"
            elif ratio < 0.67:
                interpretation = "评委更敏感"
            else:
                interpretation = "影响相近"
            
            print(f"  {row['feature']:<35} {row['importance_norm_judge']:.4f}    "
                  f"{row['importance_norm_fan']:.4f}    {ratio:.2f}       {interpretation}")
        
        return self
    
    def analyze_category_effects(self):
        """分析分类变量的效应"""
        print("\n[Step 6] 分析分类变量效应...")
        
        self.category_effects = {}
        
        # 舞伴效应分析
        if "partner" in self.df.columns:
            # 限制舞伴数量（只分析出场次数较多的舞伴）
            partner_counts = self.df["partner"].value_counts()
            top_partners = partner_counts[partner_counts >= 30].index.tolist()
            
            partner_effect = self.df[self.df["partner"].isin(top_partners)].groupby("partner").apply(
                lambda g: pd.Series({
                    "count": len(g),
                    "judge_pred_mean": self.model_judge.predict(
                        g[self.feature_names].values
                    ).mean() if len(g) > 0 else np.nan,
                    "fan_pred_mean": self.model_fan.predict(
                        g[self.feature_names].values
                    ).mean() if len(g) > 0 else np.nan,
                    "judge_actual_mean": g["target_judge"].mean(),
                    "fan_actual_mean": g["target_fan"].mean()
                })
            ).reset_index()
            
            # 添加Other类别（其他舞伴）
            other_partners_df = self.df[~self.df["partner"].isin(top_partners)]
            if len(other_partners_df) > 0:
                other_row = pd.DataFrame([{
                    "partner": "Other",
                    "count": len(other_partners_df),
                    "judge_pred_mean": self.model_judge.predict(
                        other_partners_df[self.feature_names].values
                    ).mean(),
                    "fan_pred_mean": self.model_fan.predict(
                        other_partners_df[self.feature_names].values
                    ).mean(),
                    "judge_actual_mean": other_partners_df["target_judge"].mean(),
                    "fan_actual_mean": other_partners_df["target_fan"].mean()
                }])
                partner_effect = pd.concat([partner_effect, other_row], ignore_index=True)
            
            self.category_effects["ballroom_partner"] = partner_effect
            
            print(f"\n  舞伴效应 (Top 10):")
            top10 = partner_effect.nlargest(10, "fan_pred_mean")
            for _, row in top10.iterrows():
                print(f"    {row['partner']}: 评委预测={row['judge_pred_mean']:.1f}, "
                      f"粉丝预测={row['fan_pred_mean']:.0f}")
        
        # 行业效应分析
        if "industry_category" in self.df.columns:
            industry_effect = self.df.groupby("industry_category").apply(
                lambda g: pd.Series({
                    "count": len(g),
                    "judge_pred_mean": self.model_judge.predict(
                        g[self.feature_names].values
                    ).mean() if len(g) > 0 else np.nan,
                    "fan_pred_mean": self.model_fan.predict(
                        g[self.feature_names].values
                    ).mean() if len(g) > 0 else np.nan,
                    "judge_actual_mean": g["target_judge"].mean(),
                    "fan_actual_mean": g["target_fan"].mean()
                })
            ).reset_index()
            
            self.category_effects["industry_category"] = industry_effect
            
            print(f"\n  行业效应:")
            for _, row in industry_effect.iterrows():
                print(f"    {row['industry_category']}: 评委预测={row['judge_pred_mean']:.1f}, "
                      f"粉丝预测={row['fan_pred_mean']:.0f}")
        
        # 年龄组效应分析
        if "age_group" in self.df.columns:
            age_effect = self.df.groupby("age_group").apply(
                lambda g: pd.Series({
                    "count": len(g),
                    "judge_pred_mean": self.model_judge.predict(
                        g[self.feature_names].values
                    ).mean() if len(g) > 0 else np.nan,
                    "fan_pred_mean": self.model_fan.predict(
                        g[self.feature_names].values
                    ).mean() if len(g) > 0 else np.nan,
                    "judge_actual_mean": g["target_judge"].mean(),
                    "fan_actual_mean": g["target_fan"].mean()
                })
            ).reset_index()
            
            self.category_effects["age_group"] = age_effect
            
            print(f"\n  年龄组效应:")
            for _, row in age_effect.iterrows():
                print(f"    {row['age_group']}: 评委预测={row['judge_pred_mean']:.1f}, "
                      f"粉丝预测={row['fan_pred_mean']:.0f}")
        
        return self
    
    def save_results(self):
        """保存分析结果"""
        print("\n[Step 7] 保存分析结果...")
        
        # 保存模型指标
        metrics_df = pd.DataFrame([
            {"model": "judge_score", "r2": self.metrics["judge"]["r2_test"],
             "mae": self.metrics["judge"]["mae_test"],
             "n_train": len(self.X_train), "n_test": len(self.X_test)},
            {"model": "fan_votes", "r2": self.metrics["fan"]["r2_test"],
             "mae": self.metrics["fan"]["mae_test"],
             "n_train": len(self.X_train), "n_test": len(self.X_test)}
        ])
        metrics_df.to_csv(CSV_OUTPUT_DIR / "q3_model_metrics.csv", index=False)
        
        # 保存特征重要性
        self.importance_judge.to_csv(
            CSV_OUTPUT_DIR / "q3_feature_importance_judge.csv", index=False
        )
        self.importance_fan.to_csv(
            CSV_OUTPUT_DIR / "q3_feature_importance_fan.csv", index=False
        )
        self.importance_ratio.to_csv(
            CSV_OUTPUT_DIR / "q3_feature_importance_ratio.csv", index=False
        )
        
        # 保存分类效应
        for name, effect_df in self.category_effects.items():
            # 限制保存的舞伴数量
            if name == "ballroom_partner":
                effect_df.to_csv(
                    CSV_OUTPUT_DIR / f"q3_category_effect_{name}_capped.csv", index=False
                )
            else:
                effect_df.to_csv(
                    CSV_OUTPUT_DIR / f"q3_category_effect_{name}.csv", index=False
                )
        
        print(f"  结果已保存至 {CSV_OUTPUT_DIR}")
        return self
    
    def run(self):
        """运行完整分析"""
        return (self.prepare_train_test_split()
                .train_xgboost_models()
                .compute_feature_importance()
                .compute_importance_ratio()
                .analyze_category_effects()
                .save_results())


# ============================================================
# PART 3: 可视化
# ============================================================
class Q3Visualizer:
    """问题三可视化器"""
    
    def __init__(self, analyzer: XGBoostRankSHAPAnalyzer):
        self.analyzer = analyzer
        self.fe = analyzer.fe
        self.df = analyzer.df
        
    def plot_global_importance(self):
        """图1: 全局特征重要性对比（双柱状图）"""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        # 准备数据
        importance_df = self.analyzer.importance_ratio.copy()
        features = importance_df["feature"].tolist()
        
        # 特征名称映射（更友好的显示名称）
        feature_display = {
            "week": "Week Number",
            "season": "Season",
            "age": "Celebrity Age",
            "partner_prev_seasons": "Partner Experience",
            "partner_historical_avg_placement": "Partner Avg Placement",
            "industry_category_encoded": "Industry Category",
            "is_us_contestant": "US Contestant",
            "partner_popularity": "Partner Popularity",
            "age_normalized": "Age (Normalized)",
        }
        
        features_display = [feature_display.get(f, f) for f in features]
        
        x = np.arange(len(features))
        width = 0.35
        
        # 绘制双柱状图
        bars1 = ax.bar(x - width/2, importance_df["importance_norm_judge"], width,
                       label="Judge Score", color=COLORS["judge"], alpha=0.85,
                       edgecolor="white")
        bars2 = ax.bar(x + width/2, importance_df["importance_norm_fan"], width,
                       label="Fan Votes", color=COLORS["fan"], alpha=0.85,
                       edgecolor="white")
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel("Feature", fontsize=11)
        ax.set_ylabel("Normalized Importance", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(features_display, rotation=30, ha="right", fontsize=9)
        ax.legend()
        ax.set_ylim(0, max(importance_df["importance_norm_judge"].max(),
                          importance_df["importance_norm_fan"].max()) * 1.15)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig1_global_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Fig1] 全局特征重要性对比图已保存")
    
    def plot_importance_ratio(self):
        """图2: 评委vs粉丝偏好差异（雷达图/条形图）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        
        importance_df = self.analyzer.importance_ratio.copy()
        
        # 左图: 重要性比例条形图
        ax1 = axes[0]
        
        features = importance_df["feature"].tolist()
        ratios = importance_df["ratio"].fillna(0).tolist()
        
        # 特征名称映射
        feature_display = {
            "week": "Week",
            "season": "Season",
            "age": "Age",
            "partner_prev_seasons": "Partner Exp",
            "partner_historical_avg_placement": "Partner Rank",
            "industry_category_encoded": "Industry",
            "is_us_contestant": "US Contestant",
        }
        features_display = [feature_display.get(f, f) for f in features]
        
        # 根据比例着色
        colors = [COLORS["fan"] if r > 1 else COLORS["judge"] for r in ratios]
        
        bars = ax1.barh(features_display, ratios, color=colors, alpha=0.85, edgecolor="white")
        ax1.axvline(x=1, color="black", linestyle="--", linewidth=1.5, label="Equal Importance")
        ax1.axvline(x=1.5, color=COLORS["fan"], linestyle=":", linewidth=1, alpha=0.7)
        ax1.axvline(x=0.67, color=COLORS["judge"], linestyle=":", linewidth=1, alpha=0.7)
        
        ax1.set_xlabel("Importance Ratio (Fan / Judge)", fontsize=11)
        ax1.set_ylabel("Feature", fontsize=11)
        
        # 添加区域标注
        ax1.fill_betweenx([-0.5, len(features)-0.5], 0, 0.67, alpha=0.1, 
                         color=COLORS["judge"], label="Judge-favored")
        ax1.fill_betweenx([-0.5, len(features)-0.5], 1.5, max(ratios)*1.1, alpha=0.1,
                         color=COLORS["fan"], label="Fan-favored")
        
        ax1.legend(loc="upper right", fontsize=9)
        ax1.set_xlim(0, max(ratios) * 1.1)
        
        # 右图: 差异瀑布图
        ax2 = axes[1]
        
        diffs = importance_df["diff"].tolist()
        colors_diff = [COLORS["fan"] if d > 0 else COLORS["judge"] for d in diffs]
        
        ax2.barh(features_display, diffs, color=colors_diff, alpha=0.85, edgecolor="white")
        ax2.axvline(x=0, color="black", linestyle="-", linewidth=1)
        
        ax2.set_xlabel("Importance Difference (Fan - Judge)", fontsize=11)
        ax2.set_ylabel("Feature", fontsize=11)
        
        # 添加标注
        ax2.text(0.95, 0.95, "Fan-favored →", transform=ax2.transAxes,
                ha="right", va="top", fontsize=10, color=COLORS["fan"], fontweight="bold")
        ax2.text(0.05, 0.95, "← Judge-favored", transform=ax2.transAxes,
                ha="left", va="top", fontsize=10, color=COLORS["judge"], fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig2_importance_ratio.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Fig2] 偏好差异分析图已保存")
    
    def plot_partner_effect(self):
        """图3: 舞伴效应分析"""
        if "ballroom_partner" not in self.analyzer.category_effects:
            return
        
        partner_df = self.analyzer.category_effects["ballroom_partner"].copy()
        partner_df = partner_df[partner_df["count"] >= 30]  # 只显示样本量足够的
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300)
        
        # 按粉丝预测排序
        partner_df = partner_df.sort_values("fan_pred_mean", ascending=True)
        
        # 限制显示数量
        if len(partner_df) > 20:
            partner_df = partner_df.tail(20)
        
        # 左图: 粉丝投票预测
        ax1 = axes[0]
        partners = partner_df["partner"].tolist()
        
        # 修改partner为更简短的名称
        partners_short = [p if len(p) <= 15 else p[:12] + "..." for p in partners]
        
        y_pos = np.arange(len(partners))
        
        ax1.barh(y_pos, partner_df["fan_pred_mean"], 
                color=COLORS["fan"], alpha=0.85, edgecolor="white", height=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(partners_short, fontsize=9)
        ax1.set_xlabel("Predicted Fan Votes", fontsize=11)
        ax1.set_ylabel("Professional Partner", fontsize=11)
        
        # 标注Top3
        for i in range(-3, 0):
            ax1.annotate(f'Top {-i}', 
                        xy=(partner_df["fan_pred_mean"].iloc[i], y_pos[i]),
                        xytext=(10, 0), textcoords="offset points",
                        fontsize=9, fontweight="bold", color=COLORS["dark"])
        
        # 右图: 评委得分预测
        ax2 = axes[1]
        
        partner_df_j = partner_df.sort_values("judge_pred_mean", ascending=True)
        partners_j = partner_df_j["partner"].tolist()
        partners_j_short = [p if len(p) <= 15 else p[:12] + "..." for p in partners_j]
        
        ax2.barh(y_pos, partner_df_j["judge_pred_mean"],
                color=COLORS["judge"], alpha=0.85, edgecolor="white", height=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(partners_j_short, fontsize=9)
        ax2.set_xlabel("Predicted Judge Score", fontsize=11)
        ax2.set_ylabel("Professional Partner", fontsize=11)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig3_partner_effect.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Fig3] 舞伴效应分析图已保存")
    
    def plot_industry_effect(self):
        """图4: 行业效应分析"""
        if "industry_category" not in self.analyzer.category_effects:
            return
        
        industry_df = self.analyzer.category_effects["industry_category"].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        
        # 归一化（按最大值缩放，避免最小值变为0导致“空柱”）
        industry_df["fan_pred_norm"] = (
            industry_df["fan_pred_mean"] / industry_df["fan_pred_mean"].max()
        )
        industry_df["judge_pred_norm"] = (
            industry_df["judge_pred_mean"] / industry_df["judge_pred_mean"].max()
        )
        
        industries = industry_df["industry_category"].tolist()
        x = np.arange(len(industries))
        width = 0.35
        
        # 左图: 原始预测值
        ax1 = axes[0]
        
        # 使用双Y轴
        ax1_twin = ax1.twinx()
        
        bars1 = ax1.bar(x - width/2, industry_df["judge_pred_mean"], width,
                       label="Judge Score", color=COLORS["judge"], alpha=0.85)
        bars2 = ax1_twin.bar(x + width/2, industry_df["fan_pred_mean"], width,
                            label="Fan Votes", color=COLORS["fan"], alpha=0.85)
        
        ax1.set_xlabel("Industry Category", fontsize=11)
        ax1.set_ylabel("Judge Score", fontsize=11, color=COLORS["judge"])
        ax1_twin.set_ylabel("Fan Votes", fontsize=11, color=COLORS["fan"])
        ax1.set_xticks(x)
        ax1.set_xticklabels(industries, rotation=30, ha="right", fontsize=9)
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        
        # 右图: 归一化比较
        ax2 = axes[1]
        
        ax2.bar(x - width/2, industry_df["judge_pred_norm"], width,
               label="Judge (Normalized)", color=COLORS["judge"], alpha=0.85)
        ax2.bar(x + width/2, industry_df["fan_pred_norm"], width,
               label="Fan (Normalized)", color=COLORS["fan"], alpha=0.85)
        
        ax2.set_xlabel("Industry Category", fontsize=11)
        ax2.set_ylabel("Normalized Score (max=1)", fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels(industries, rotation=30, ha="right", fontsize=9)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig4_industry_effect.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Fig4] 行业效应分析图已保存")
    
    def plot_age_group_effect(self):
        """图5: 年龄组效应分析"""
        if "age_group" not in self.analyzer.category_effects:
            return
        
        age_df = self.analyzer.category_effects["age_group"].copy()
        
        # 按年龄组排序
        age_order = ["≤25", "26-35", "36-45", "46-55", "55+"]
        age_df["age_group"] = pd.Categorical(
            age_df["age_group"], categories=age_order, ordered=True
        )
        age_df = age_df.sort_values("age_group")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        
        age_groups = age_df["age_group"].tolist()
        x = np.arange(len(age_groups))
        
        # 左图: 评委得分趋势
        ax1 = axes[0]
        
        ax1.plot(x, age_df["judge_pred_mean"], "o-", 
                color=COLORS["judge"], linewidth=2.5, markersize=10,
                label="Predicted")
        ax1.plot(x, age_df["judge_actual_mean"], "s--",
                color=COLORS["judge"], linewidth=2, markersize=8, alpha=0.6,
                label="Actual")
        
        ax1.fill_between(x, age_df["judge_pred_mean"], age_df["judge_actual_mean"],
                        alpha=0.2, color=COLORS["judge"])
        
        ax1.set_xlabel("Age Group", fontsize=11)
        ax1.set_ylabel("Judge Score", fontsize=11)
        ax1.set_xticks(x)
        ax1.set_xticklabels(age_groups, fontsize=10)
        ax1.legend()
        
        # 右图: 粉丝投票趋势
        ax2 = axes[1]
        
        ax2.plot(x, age_df["fan_pred_mean"], "o-",
                color=COLORS["fan"], linewidth=2.5, markersize=10,
                label="Predicted")
        ax2.plot(x, age_df["fan_actual_mean"], "s--",
                color=COLORS["fan"], linewidth=2, markersize=8, alpha=0.6,
                label="Actual")
        
        ax2.fill_between(x, age_df["fan_pred_mean"], age_df["fan_actual_mean"],
                        alpha=0.2, color=COLORS["fan"])
        
        ax2.set_xlabel("Age Group", fontsize=11)
        ax2.set_ylabel("Fan Votes", fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels(age_groups, fontsize=10)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig5_age_group_effect.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Fig5] 年龄组效应分析图已保存")
    
    def plot_shap_summary(self):
        """图6: SHAP Summary Plot (如果SHAP可用)"""
        if not SHAP_AVAILABLE or self.analyzer.shap_values_judge is None:
            print("  [Fig6] SHAP不可用，跳过SHAP摘要图")
            return
        
        # 评委模型SHAP
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        
        # 左图: 评委模型
        plt.sca(axes[0])
        shap.summary_plot(
            self.analyzer.shap_values_judge,
            self.analyzer.X_sample,
            feature_names=self.analyzer.feature_names,
            show=False,
            plot_size=None
        )
        # MCM图表不显示标题
        
        # 右图: 粉丝模型
        plt.sca(axes[1])
        shap.summary_plot(
            self.analyzer.shap_values_fan,
            self.analyzer.X_sample,
            feature_names=self.analyzer.feature_names,
            show=False,
            plot_size=None
        )
        # MCM图表不显示标题
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig6_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Fig6] SHAP摘要图已保存")
    
    def plot_model_performance(self):
        """图7: 模型性能对比"""
        metrics = self.analyzer.metrics
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        
        # 左图: R² 对比
        ax1 = axes[0]
        models = ["Judge Score", "Fan Votes"]
        r2_train = [metrics["judge"]["r2_train"], metrics["fan"]["r2_train"]]
        r2_test = [metrics["judge"]["r2_test"], metrics["fan"]["r2_test"]]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, r2_train, width, label="Train R²",
               color=COLORS["neutral"], alpha=0.85)
        ax1.bar(x + width/2, r2_test, width, label="Test R²",
               color=COLORS["primary"], alpha=0.85)
        
        ax1.set_ylabel("R² Score", fontsize=11)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontsize=11)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for i, (tr, te) in enumerate(zip(r2_train, r2_test)):
            ax1.text(i - width/2, tr + 0.02, f"{tr:.3f}", ha="center", fontsize=9)
            ax1.text(i + width/2, te + 0.02, f"{te:.3f}", ha="center", fontsize=9)
        
        # 右图: MAE/RMSE 对比（使用对数尺度，避免数量级差异导致不可读）
        ax2 = axes[1]
        
        mae = [metrics["judge"]["mae_test"], metrics["fan"]["mae_test"]]
        rmse = [metrics["judge"]["rmse_test"], metrics["fan"]["rmse_test"]]

        ax2.bar(x - width/2, mae, width, label="MAE",
               color=COLORS["accent"], alpha=0.85)
        ax2.bar(x + width/2, rmse, width, label="RMSE",
               color=COLORS["success"], alpha=0.85)

        ax2.set_ylabel("Error (log scale)", fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, fontsize=11)
        ax2.set_yscale("log")
        ax2.yaxis.set_major_formatter(ScalarFormatter())
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        ax2.legend()
        
        # 添加原始值标签
        for i, (m, r) in enumerate(zip(mae, rmse)):
            ax2.text(i - width/2, m * 1.15, f"{m:.1f}", ha="center", fontsize=8)
            ax2.text(i + width/2, r * 1.15, f"{r:.1f}", ha="center", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig7_model_performance.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Fig7] 模型性能对比图已保存")
    
    def plot_rankshap_importance(self):
        """图8: RankSHAP全局特征重要性对比（基于NDCG）"""
        if self.analyzer.rankshap_judge is None:
            print("  [Fig8] RankSHAP不可用，跳过")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        const_rank_features = {"week", "season"}
        
        # 左图: RankSHAP柱状图对比（按最大重要性排序）
        ax1 = axes[0]
        
        # 合并两个RankSHAP表
        rankshap_df = self.analyzer.rankshap_ratio.copy()
        rankshap_df["rankshap_max"] = rankshap_df[
            ["rankshap_norm_judge", "rankshap_norm_fan"]
        ].max(axis=1)
        rankshap_df = rankshap_df.sort_values("rankshap_max", ascending=False)
        features = rankshap_df["feature"].tolist()
        
        # 特征名称映射
        feature_display = {
            "week": "Week Number",
            "season": "Season",
            "age": "Celebrity Age",
            "partner_prev_seasons": "Partner Experience",
            "partner_historical_avg_placement": "Partner Avg Placement",
            "industry_category_encoded": "Industry Category",
            "is_us_contestant": "US Contestant",
        }
        features_display = [
            f"{feature_display.get(f, f)} (const)" if f in const_rank_features
            else feature_display.get(f, f)
            for f in features
        ]
        
        y = np.arange(len(features))
        height = 0.35
        
        bars1 = ax1.barh(y - height/2, rankshap_df["rankshap_norm_judge"], height,
                       label="Judge Score (NDCG)", color=COLORS["judge"], alpha=0.85,
                       edgecolor="white")
        bars2 = ax1.barh(y + height/2, rankshap_df["rankshap_norm_fan"], height,
                       label="Fan Votes (NDCG)", color=COLORS["fan"], alpha=0.85,
                       edgecolor="white")
        
        ax1.set_xlabel("RankSHAP Importance (NDCG-based)", fontsize=11)
        ax1.set_ylabel("Feature", fontsize=11)
        # MCM图表不显示标题
        ax1.set_yticks(y)
        ax1.set_yticklabels(features_display, fontsize=9)
        ax1.invert_yaxis()
        ax1.grid(axis="x", linestyle="--", alpha=0.4)
        ax1.legend()

        # 数值标签
        for bar in list(bars1) + list(bars2):
            w = bar.get_width()
            ax1.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{w:.2f}", va="center", fontsize=8, color=COLORS["dark"])
        
        # 右图: RankSHAP vs TreeSHAP 对比
        ax2 = axes[1]
        
        # 使用评委模型作为对比
        treeshap_norm = self.analyzer.importance_judge.set_index("feature")["importance_norm"]
        rankshap_norm = self.analyzer.rankshap_judge.set_index("feature")["rankshap_norm"]
        
        comparison_df = pd.DataFrame({
            "feature": self.analyzer.feature_names,
            "TreeSHAP": [treeshap_norm.get(f, 0) for f in self.analyzer.feature_names],
            "RankSHAP": [rankshap_norm.get(f, 0) for f in self.analyzer.feature_names]
        })
        comparison_df["importance_max"] = comparison_df[["TreeSHAP", "RankSHAP"]].max(axis=1)
        comparison_df = comparison_df.sort_values("importance_max", ascending=False)
        
        features_cmp = comparison_df["feature"].tolist()
        features_cmp_display = [
            f"{feature_display.get(f, f)} (const)" if f in const_rank_features
            else feature_display.get(f, f)
            for f in features_cmp
        ]
        
        y2 = np.arange(len(features_cmp))
        
        ax2.barh(y2 - height/2, comparison_df["TreeSHAP"], height,
               label="TreeSHAP (Prediction)", color=COLORS["neutral"], alpha=0.85)
        ax2.barh(y2 + height/2, comparison_df["RankSHAP"], height,
               label="RankSHAP (NDCG)", color=COLORS["primary"], alpha=0.85)
        
        ax2.set_xlabel("Normalized Importance", fontsize=11)
        ax2.set_ylabel("Feature", fontsize=11)
        # MCM图表不显示标题
        ax2.set_yticks(y2)
        ax2.set_yticklabels(features_cmp_display, fontsize=9)
        ax2.invert_yaxis()
        ax2.grid(axis="x", linestyle="--", alpha=0.4)
        ax2.legend()

        # 数值标签
        for bar in ax2.patches:
            w = bar.get_width()
            ax2.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{w:.2f}", va="center", fontsize=8, color=COLORS["dark"])
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig8_rankshap_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Fig8] RankSHAP特征重要性图已保存")
    
    def run(self):
        """生成所有图表"""
        print("\n" + "=" * 60)
        print("PART 3: 可视化")
        print("=" * 60)
        
        self.plot_global_importance()
        self.plot_importance_ratio()
        self.plot_partner_effect()
        self.plot_industry_effect()
        self.plot_age_group_effect()
        self.plot_shap_summary()
        self.plot_model_performance()
        self.plot_rankshap_importance()  # 新增：RankSHAP可视化
        
        print(f"\n  所有图表已保存至: {OUTPUT_DIR}")
        return self


# ============================================================
# 汇总报告生成
# ============================================================
class Q3ReportGenerator:
    """问题三报告生成器"""
    
    def __init__(self, analyzer: XGBoostRankSHAPAnalyzer):
        self.analyzer = analyzer
        
    def generate_summary(self):
        """生成汇总报告"""
        print("\n" + "=" * 70)
        print("Q3 分析结果汇总 (含RankSHAP)")
        print("=" * 70)
        
        # 模型性能
        print("\n【模型性能】")
        print(f"  评委得分模型 Test R²: {self.analyzer.metrics['judge']['r2_test']:.4f}")
        print(f"  粉丝投票模型 Test R²: {self.analyzer.metrics['fan']['r2_test']:.4f}")
        
        # NDCG指标（RankSHAP）
        if 'ndcg_judge' in self.analyzer.metrics:
            print(f"\n【排名质量 (NDCG)】")
            print(f"  评委模型 NDCG: {self.analyzer.metrics['ndcg_judge']:.4f}")
            print(f"  粉丝模型 NDCG: {self.analyzer.metrics['ndcg_fan']:.4f}")
        
        # 特征重要性排序
        print("\n【特征重要性排序】")
        print("\n  评委得分模型:")
        for i, (_, row) in enumerate(self.analyzer.importance_judge.head(5).iterrows(), 1):
            print(f"    {i}. {row['feature']}: {row['importance_norm']:.4f}")
        
        print("\n  粉丝投票模型:")
        for i, (_, row) in enumerate(self.analyzer.importance_fan.head(5).iterrows(), 1):
            print(f"    {i}. {row['feature']}: {row['importance_norm']:.4f}")
        
        # 偏好差异
        print("\n【评委vs粉丝偏好差异】")
        print(f"  {'特征':<30} {'评委重要性':<12} {'粉丝重要性':<12} {'比例':<8} {'结论'}")
        print("  " + "-" * 75)
        
        for _, row in self.analyzer.importance_ratio.iterrows():
            ratio = row["ratio"]
            if pd.isna(ratio):
                conclusion = "N/A"
            elif ratio > 1.5:
                conclusion = "粉丝更敏感 ⬆"
            elif ratio < 0.67:
                conclusion = "评委更敏感 ⬇"
            else:
                conclusion = "影响相近 ≈"
            
            print(f"  {row['feature']:<30} {row['importance_norm_judge']:.4f}       "
                  f"{row['importance_norm_fan']:.4f}       {ratio:.2f}    {conclusion}")
        
        # 关键发现
        print("\n【关键发现】")
        
        # 找出差异最大的特征
        ratio_df = self.analyzer.importance_ratio
        fan_favored = ratio_df[ratio_df["ratio"] > 1.5]
        judge_favored = ratio_df[ratio_df["ratio"] < 0.67]
        
        if len(fan_favored) > 0:
            print(f"\n  粉丝更关注的特征:")
            for _, row in fan_favored.iterrows():
                print(f"    - {row['feature']} (比例: {row['ratio']:.2f})")
        
        if len(judge_favored) > 0:
            print(f"\n  评委更关注的特征:")
            for _, row in judge_favored.iterrows():
                print(f"    - {row['feature']} (比例: {row['ratio']:.2f})")
        
        # 输出文件列表
        print("\n【输出文件】")
        print(f"  CSV文件: {CSV_OUTPUT_DIR}/")
        for f in CSV_OUTPUT_DIR.glob("*.csv"):
            print(f"    - {f.name}")
        
        print(f"\n  图表文件: {OUTPUT_DIR}/")
        for f in OUTPUT_DIR.glob("*.png"):
            print(f"    - {f.name}")
        
        print("\n" + "=" * 70)
        
        # 保存汇总CSV
        summary_data = []
        summary_data.append({"metric": "Judge Model R²", "value": f"{self.analyzer.metrics['judge']['r2_test']:.4f}"})
        summary_data.append({"metric": "Fan Model R²", "value": f"{self.analyzer.metrics['fan']['r2_test']:.4f}"})
        summary_data.append({"metric": "Judge Model MAE", "value": f"{self.analyzer.metrics['judge']['mae_test']:.2f}"})
        summary_data.append({"metric": "Fan Model MAE", "value": f"{self.analyzer.metrics['fan']['mae_test']:.2f}"})
        
        # 添加NDCG指标
        if 'ndcg_judge' in self.analyzer.metrics:
            summary_data.append({"metric": "Judge Model NDCG", "value": f"{self.analyzer.metrics['ndcg_judge']:.4f}"})
            summary_data.append({"metric": "Fan Model NDCG", "value": f"{self.analyzer.metrics['ndcg_fan']:.4f}"})
        
        for _, row in self.analyzer.importance_ratio.iterrows():
            summary_data.append({
                "metric": f"TreeSHAP Ratio: {row['feature']}",
                "value": f"{row['ratio']:.4f}" if not pd.isna(row['ratio']) else "N/A"
            })
        
        # 添加RankSHAP比例
        if hasattr(self.analyzer, 'rankshap_ratio') and self.analyzer.rankshap_ratio is not None:
            for _, row in self.analyzer.rankshap_ratio.iterrows():
                summary_data.append({
                    "metric": f"RankSHAP Ratio: {row['feature']}",
                    "value": f"{row['ratio']:.4f}" if not pd.isna(row['ratio']) else "N/A"
                })
        
        pd.DataFrame(summary_data).to_csv(CSV_OUTPUT_DIR / "q3_summary.csv", index=False)
        
        # 保存RankSHAP详细结果
        if hasattr(self.analyzer, 'rankshap_judge') and self.analyzer.rankshap_judge is not None:
            self.analyzer.rankshap_judge.to_csv(CSV_OUTPUT_DIR / "q3_rankshap_judge.csv", index=False)
            self.analyzer.rankshap_fan.to_csv(CSV_OUTPUT_DIR / "q3_rankshap_fan.csv", index=False)
            print(f"\n  RankSHAP结果已保存至: {CSV_OUTPUT_DIR}")
        
        return self


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数：运行完整Q3分析（含RankSHAP）"""
    print("=" * 70)
    print("MCM 2026 Problem C - Question 3: 特征影响分析 (RankSHAP)")
    print("=" * 70)
    
    # Part 1: 特征工程
    fe = FeatureEngineer()
    fe.run()
    
    # Part 2: XGBoost建模与RankSHAP分析
    analyzer = XGBoostRankSHAPAnalyzer(fe)
    analyzer.prepare_train_test_split()
    analyzer.train_xgboost_models()
    analyzer.compute_feature_importance()  # TreeSHAP（作为对比）
    analyzer.compute_rankshap()           # RankSHAP（基于NDCG）
    analyzer.compute_importance_ratio()   # TreeSHAP偏好差异
    analyzer.compute_rankshap_ratio()     # RankSHAP偏好差异
    analyzer.analyze_category_effects()
    
    # Part 3: 可视化
    visualizer = Q3Visualizer(analyzer)
    visualizer.run()
    
    # 生成报告
    reporter = Q3ReportGenerator(analyzer)
    reporter.generate_summary()
    
    print("\n分析完成！")
    
    return fe, analyzer, visualizer


if __name__ == "__main__":
    fe, analyzer, visualizer = main()

