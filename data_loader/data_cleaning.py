"""
MCM 2026 Problem C - DWTS Data Cleaning Script
================================================
数据清洗脚本：处理缺失值、转换数据格式、创建衍生特征

输入: 2026_MCM_Problem_C_Data.csv
输出: 
  - dwts_cleaned.csv (清洗后的宽格式数据)
  - dwts_long_format.csv (长格式数据，每行一个选手-周次)
  - dwts_summary.csv (选手汇总数据)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. 加载原始数据
df = pd.read_csv('2026_MCM_Problem_C_Data.csv')
print(f"原始数据维度: {df.shape}, 选手数: {len(df)}")

# 2. 基本信息清洗
# 2.1 统一列名（去除空格，转小写）
df.columns = df.columns.str.strip().str.lower().str.replace('/', '_')

# 2.2 处理名人姓名（去除多余空格）
df['celebrity_name'] = df['celebrity_name'].str.strip()

# 2.3 处理家乡州（填充缺失值）
df['celebrity_homestate'] = df['celebrity_homestate'].fillna('Unknown')

# 2.4 处理国家（标准化）
df['celebrity_homecountry_region'] = df['celebrity_homecountry_region'].fillna('Unknown')

# 2.5 处理职业类型（标准化）
df['celebrity_industry'] = df['celebrity_industry'].str.strip()
# 合并相似类别
industry_mapping = {
    'Social media personality': 'Social Media Personality',
    'Beauty Pagent': 'Beauty Pageant'
}
df['celebrity_industry'] = df['celebrity_industry'].replace(industry_mapping)

# 2.6 处理年龄（检查异常值）
df['celebrity_age_during_season'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce')
age_median = df['celebrity_age_during_season'].median()
df['celebrity_age_during_season'] = df['celebrity_age_during_season'].fillna(age_median)

# 3. 解析淘汰信息
def parse_results(results):
    """解析results列，返回淘汰周次和最终名次"""
    if pd.isna(results):
        return None, None
    
    results = str(results).strip()
    
    if 'Eliminated Week' in results:
        week = int(results.split('Week ')[-1])
        return week, None
    elif '1st Place' in results:
        return None, 1
    elif '2nd Place' in results:
        return None, 2
    elif '3rd Place' in results:
        return None, 3
    elif '4th Place' in results:
        return None, 4
    elif '5th Place' in results:
        return None, 5
    elif 'Withdrew' in results:
        return -1, None  # -1表示退赛
    else:
        return None, None

# 应用解析
df['elimination_week'], df['final_place'] = zip(*df['results'].apply(parse_results))

# 计算参与周数
def calculate_weeks_participated(row):
    if row['elimination_week'] is not None and row['elimination_week'] > 0:
        return row['elimination_week']
    elif row['final_place'] is not None:
        # 决赛选手，计算实际参与周数
        max_week = 11
        for week in range(max_week, 0, -1):
            col = f'week{week}_judge1_score'
            if col in row.index:
                val = row[col]
                if pd.notna(val) and str(val) != 'N/A' and val != 0:
                    return week
        return max_week
    elif row['elimination_week'] == -1:  # 退赛
        # 找最后一个有分数的周
        for week in range(11, 0, -1):
            col = f'week{week}_judge1_score'
            if col in row.index:
                val = row[col]
                if pd.notna(val) and str(val) != 'N/A' and val != 0:
                    return week
        return 1
    return 1

df['weeks_participated'] = df.apply(calculate_weeks_participated, axis=1)

# 4. 清洗评委得分
def clean_judge_score(val):
    """清洗单个评委得分"""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        if val == 0:
            return np.nan  # 0表示未参赛
        return float(val)
    val_str = str(val).strip()
    if val_str in ['N/A', 'NA', '', '0', '0.0']:
        return np.nan
    try:
        score = float(val_str)
        return score if score > 0 else np.nan
    except:
        return np.nan

# 获取所有评委得分列
judge_cols = [col for col in df.columns if 'judge' in col and 'score' in col]
print(f"    - 评委得分列数: {len(judge_cols)}")

# 清洗每个评委得分列
for col in judge_cols:
    df[col] = df[col].apply(clean_judge_score)

def normalize_judge_scores(df, weeks=range(1, 12)):
    """
    将每个赛季-周次的评委打分归一到 1-10 区间。
    返回: (df_normalized, scale_map)
    scale_map[(season, week)] = scale_factor，用于还原。
    """
    df = df.copy()
    scale_map = {}

    for week in weeks:
        week_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        existing_cols = [c for c in week_cols if c in df.columns]
        if not existing_cols:
            continue

        for season, season_df in df.groupby('season'):
            max_score = season_df[existing_cols].max().max()
            if pd.isna(max_score) or max_score <= 0:
                scale_map[(season, week)] = 1.0
                continue

            scale_factor = 10.0 / max_score if max_score > 10 else 1.0
            scale_map[(season, week)] = scale_factor
            if scale_factor != 1.0:
                season_mask = df['season'] == season
                df.loc[season_mask, existing_cols] = df.loc[season_mask, existing_cols] * scale_factor

    return df, scale_map

def restore_judge_scores(df, scale_map, weeks=range(1, 12)):
    """
    将归一化后的评委打分还原到原始尺度。
    """
    df = df.copy()
    for week in weeks:
        week_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        existing_cols = [c for c in week_cols if c in df.columns]
        if not existing_cols:
            continue

        for season in df['season'].dropna().unique():
            scale_factor = scale_map.get((season, week), 1.0)
            if scale_factor != 1.0:
                season_mask = df['season'] == season
                df.loc[season_mask, existing_cols] = df.loc[season_mask, existing_cols] / scale_factor

    return df

# 计算每周的评委总分和平均分
for week in range(1, 12):
    week_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
    existing_cols = [c for c in week_cols if c in df.columns]
    
    if existing_cols:
        # 每周评委总分（忽略缺失）
        df[f'week{week}_judge_total'] = df[existing_cols].sum(axis=1, skipna=True)
        # 每周评委平均分
        df[f'week{week}_judge_mean'] = df[existing_cols].mean(axis=1, skipna=True)
        # 每周有效评委数
        df[f'week{week}_judge_count'] = df[existing_cols].notna().sum(axis=1)
        
        # 将总分为0的设为NaN（表示该周未参赛）
        df.loc[df[f'week{week}_judge_total'] == 0, f'week{week}_judge_total'] = np.nan
        df.loc[df[f'week{week}_judge_mean'] == 0, f'week{week}_judge_mean'] = np.nan

# 5. 计算选手汇总统计
# 5.1 整体平均评委得分
total_cols = [f'week{w}_judge_total' for w in range(1, 12)]
existing_total_cols = [c for c in total_cols if c in df.columns]
df['avg_judge_total'] = df[existing_total_cols].mean(axis=1, skipna=True)

# 5.2 最高/最低周得分
df['max_judge_total'] = df[existing_total_cols].max(axis=1, skipna=True)
df['min_judge_total'] = df[existing_total_cols].min(axis=1, skipna=True)

# 5.3 得分标准差（稳定性指标）
df['std_judge_total'] = df[existing_total_cols].std(axis=1, skipna=True)

# 5.4 得分趋势（最后三周平均 - 前三周平均）
def calculate_trend(row):
    scores = [row[f'week{w}_judge_total'] for w in range(1, 12) 
              if f'week{w}_judge_total' in row.index and pd.notna(row[f'week{w}_judge_total'])]
    if len(scores) >= 4:
        early = np.mean(scores[:3])
        late = np.mean(scores[-3:])
        return late - early
    return np.nan

df['score_trend'] = df.apply(calculate_trend, axis=1)

# 6. 创建衍生特征
# 6.1 舞伴经验（该舞伴在该季之前的总参赛次数）
df = df.sort_values(['ballroom_partner', 'season'])
df['partner_prev_seasons'] = df.groupby('ballroom_partner').cumcount()

# 6.2 舞伴历史平均名次
partner_avg_placement = df.groupby('ballroom_partner')['placement'].transform(
    lambda x: x.expanding().mean().shift(1)
)
df['partner_historical_avg_placement'] = partner_avg_placement.fillna(df['placement'].mean())

# 6.3 是否来自美国
df['is_us_contestant'] = (df['celebrity_homecountry_region'] == 'United States').astype(int)

# 6.4 年龄分组
df['age_group'] = pd.cut(df['celebrity_age_during_season'], 
                         bins=[0, 25, 35, 45, 55, 100], 
                         labels=['≤25', '26-35', '36-45', '46-55', '>55'])

# 6.5 职业大类
industry_category = {
    'Actor/Actress': 'Entertainment',
    'Singer/Rapper': 'Entertainment',
    'Comedian': 'Entertainment',
    'Musician': 'Entertainment',
    'Magician': 'Entertainment',
    'Athlete': 'Sports',
    'Racing Driver': 'Sports',
    'TV Personality': 'Media',
    'News Anchor': 'Media',
    'Sports Broadcaster': 'Media',
    'Radio Personality': 'Media',
    'Journalist': 'Media',
    'Social Media Personality': 'Media',
    'Model': 'Model',
    'Beauty Pageant': 'Model',
    'Politician': 'Other',
    'Entrepreneur': 'Other',
    'Astronaut': 'Other',
    'Fashion Designer': 'Other',
    'Motivational Speaker': 'Other',
    'Military': 'Other',
    'Producer': 'Other',
    'Fitness Instructor': 'Other',
    'Con artist': 'Other',
    'Conservationist': 'Other'
}
df['industry_category'] = df['celebrity_industry'].map(industry_category).fillna('Other')

# 7. 创建长格式数据
long_records = []
for _, row in df.iterrows():
    for week in range(1, 12):
        total_col = f'week{week}_judge_total'
        if total_col in row.index and pd.notna(row[total_col]) and row[total_col] > 0:
            # 获取单个评委得分
            judge_scores = []
            for j in range(1, 5):
                col = f'week{week}_judge{j}_score'
                if col in row.index and pd.notna(row[col]):
                    judge_scores.append(row[col])
            
            # 判断是否是淘汰周
            is_elimination_week = (row['elimination_week'] == week)
            
            long_records.append({
                'celebrity_name': row['celebrity_name'],
                'season': row['season'],
                'week': week,
                'judge_total': row[total_col],
                'judge_mean': row[f'week{week}_judge_mean'],
                'judge_count': row[f'week{week}_judge_count'],
                'judge1_score': judge_scores[0] if len(judge_scores) > 0 else np.nan,
                'judge2_score': judge_scores[1] if len(judge_scores) > 1 else np.nan,
                'judge3_score': judge_scores[2] if len(judge_scores) > 2 else np.nan,
                'judge4_score': judge_scores[3] if len(judge_scores) > 3 else np.nan,
                'placement': row['placement'],
                'elimination_week': row['elimination_week'],
                'is_elimination_week': is_elimination_week,
                'celebrity_industry': row['celebrity_industry'],
                'industry_category': row['industry_category'],
                'age': row['celebrity_age_during_season'],
                'age_group': row['age_group'],
                'partner': row['ballroom_partner'],
                'partner_experience': row['partner_prev_seasons'],
                'home_country': row['celebrity_homecountry_region'],
                'home_state': row['celebrity_homestate'],
                'is_us': row['is_us_contestant']
            })

df_long = pd.DataFrame(long_records)

# 添加每周每季排名
df_long['weekly_rank'] = df_long.groupby(['season', 'week'])['judge_total'].rank(ascending=False, method='min')
df_long['weekly_rank_pct'] = df_long.groupby(['season', 'week'])['judge_total'].rank(ascending=False, pct=True)

# 8. 创建选手汇总数据
summary_cols = [
    'celebrity_name', 'ballroom_partner', 'celebrity_industry', 'industry_category',
    'celebrity_homestate', 'celebrity_homecountry_region', 'celebrity_age_during_season',
    'age_group', 'season', 'results', 'placement', 'elimination_week', 'final_place',
    'weeks_participated', 'avg_judge_total', 'max_judge_total', 'min_judge_total',
    'std_judge_total', 'score_trend', 'partner_prev_seasons', 
    'partner_historical_avg_placement', 'is_us_contestant'
]
df_summary = df[[c for c in summary_cols if c in df.columns]].copy()

# 9. 保存数据
df.to_csv('dwts_cleaned.csv', index=False)
df_long.to_csv('dwts_long_format.csv', index=False)
df_summary.to_csv('dwts_summary.csv', index=False)
print("数据清洗并保存完成。")

# 10. 关键统计输出
print(f"\n【DWTS 数据统计摘要】")
print(f"  赛季范围: S{df['season'].min()} - S{df['season'].max()} ({df['season'].nunique()}季)")
print(f"  总选手数: {len(df)}")
print(f"  总周次记录: {len(df_long)}")
print(f"  职业类型: {df['celebrity_industry'].nunique()}种")
print(f"  舞伴数量: {df['ballroom_partner'].nunique()}人")
print(f"  平均参赛周数: {df['weeks_participated'].mean():.1f}周")
print(f"  冠军数: {(df['final_place'] == 1).sum()}")
