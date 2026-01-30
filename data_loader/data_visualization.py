"""
MCM 2026 Problem C - DWTS Data Visualization Script
====================================================
数据可视化脚本：生成精简且专业的论文图表

输出: 
  - fig1_score_heatmap.png (评分热力图)
  - fig2_controversy_analysis.png (争议选手个案分析)
  - fig3_key_insights_summary.png (六合一综合洞察大图)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 主题配色
COLORS = {
    'primary': '#7BADDF',      # 浅蓝
    'secondary': '#B581B4',    # 薰衣草紫
    'accent': '#EAB170',       # 暖橙
    'success': '#DA8176',      # 珊瑚粉
    'neutral': '#B1A8D3',      # 淡紫
    'light': '#BADDF3'         # 极浅蓝
}

PALETTE = [
    '#BADDF3', '#C8C3E1', '#B581B4', '#B1A8D3', '#B5C3EA', 
    '#7FBDB0', '#F4E09B', '#EAB170', '#DA8176', '#7BADDF'
]


df = pd.read_csv('dwts_cleaned.csv')
df_long = pd.read_csv('dwts_long_format.csv')
df_summary = pd.read_csv('dwts_summary.csv')

# 图 1: 评分热力图 (保留核心)

fig, ax = plt.subplots(figsize=(14, 8))
custom_heatmap_cmap = LinearSegmentedColormap.from_list("dwts_theme", ["#BADDF3", "#7BADDF", "#B581B4", "#DA8176"])
pivot_data = df_long.pivot_table(values='judge_total', index='season', columns='week', aggfunc='mean')
sns.heatmap(pivot_data, cmap=custom_heatmap_cmap, ax=ax, linewidths=0.5, linecolor='white', cbar_kws={'label': 'Average Judge Total Score'})
ax.set_title('Average Judge Total Score by Season and Week')
plt.savefig('fig1_score_heatmap.png')
plt.close()


# 图 2: 争议选手分析 (保留核心)
controversy_names = ['Jerry Rice', 'Bristol Palin', 'Bobby Bones', 'Billy Ray Cyrus']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for idx, name in enumerate(controversy_names):
    ax = axes[idx]
    contestant_data = df_long[df_long['celebrity_name'] == name].sort_values('week')
    if len(contestant_data) == 0: continue
    season = contestant_data['season'].iloc[0]
    placement = contestant_data['placement'].iloc[0]
    season_avg = df_long[df_long['season'] == season].groupby('week')['judge_total'].mean()
    ax.plot(contestant_data['week'], contestant_data['judge_total'], 'o-', color=COLORS['success'], linewidth=2, markersize=8, label=name)
    ax.plot(season_avg.index, season_avg.values, 's--', color=COLORS['neutral'], alpha=0.5, label='Season Average')
    for _, row in contestant_data.iterrows():
        ax.annotate(f"#{int(row['weekly_rank'])}", (row['week'], row['judge_total']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    ax.set_title(f'{name} (Season {season}, Final: #{int(placement)})')
    ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('fig2_controversy_analysis.png')
plt.close()

# 图 3: 综合洞察大图 (6合1聚合)

fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# (a) 职业与排名
ax = axes[0, 0]
industry_stats = df.groupby('celebrity_industry').agg({'placement': 'mean', 'celebrity_name': 'count'}).rename(columns={'celebrity_name': 'count'})
industry_stats = industry_stats[industry_stats['count'] >= 5].sort_values('placement')
colors = [COLORS['success'] if p < industry_stats['placement'].median() else COLORS['primary'] for p in industry_stats['placement'].values]
ax.barh(range(len(industry_stats)), industry_stats['placement'].values, color=colors)
ax.set_yticks(range(len(industry_stats)))
ax.set_yticklabels(industry_stats.index)
ax.invert_yaxis()
ax.set_title('(a) Average Placement by Industry (n≥5)')

# (b) 年龄与排名
ax = axes[0, 1]
age_placement = df.groupby('age_group')['placement'].mean().dropna()
ax.bar(range(len(age_placement)), age_placement.values, color=PALETTE[2:2+len(age_placement)])
ax.set_xticks(range(len(age_placement)))
ax.set_xticklabels(age_placement.index)
z = np.polyfit(range(len(age_placement)), age_placement.values, 1)
p = np.poly1d(z)
ax.plot(range(len(age_placement)), p(range(len(age_placement))), color=COLORS['success'], linestyle='--', label='Trend')
ax.set_title('(b) Average Placement by Age Group')

# (c) 舞伴经验影响
ax = axes[1, 0]
exp_groups = df.groupby('partner_prev_seasons').agg({'placement': 'mean', 'celebrity_name': 'count'}).rename(columns={'celebrity_name': 'count'})
exp_groups = exp_groups[exp_groups['count'] >= 5]
ax.scatter(exp_groups.index, exp_groups['placement'], s=exp_groups['count']*10, c=COLORS['secondary'], alpha=0.6)
z = np.polyfit(exp_groups.index, exp_groups['placement'], 1)
p = np.poly1d(z)
ax.plot(exp_groups.index, p(exp_groups.index), color=COLORS['accent'], linestyle='--')
ax.set_title('(c) Partner Experience vs Placement')

# (d) 不同名次选手的得分趋势
ax = axes[1, 1]
for place, color in zip([1, 2, 3], [COLORS['success'], COLORS['accent'], COLORS['primary']]):
    place_avg = df_long[df_long['placement'] == place].groupby('week')['judge_total'].mean()
    ax.plot(place_avg.index, place_avg.values, 'o-', color=color, label=f'Rank #{place}')
ax.set_title('(d) Score Trajectory by Final Rank')
ax.legend()

# (e) 相关性热力图
ax = axes[2, 0]
numeric_cols = ['celebrity_age_during_season', 'placement', 'weeks_participated', 'avg_judge_total', 'std_judge_total', 'partner_prev_seasons']
corr_data = df[numeric_cols].dropna()
corr_data.columns = ['Age', 'Place', 'Weeks', 'AvgSc', 'StdSc', 'PrtExp']
sns.heatmap(corr_data.corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, square=True, cbar=False)
ax.set_title('(e) Correlation Matrix of Key Variables')

# (f) 淘汰选手的平均分分布
ax = axes[2, 1]
elim_scores = df[df['elimination_week'] > 0].groupby('elimination_week')['avg_judge_total'].mean()
ax.bar(elim_scores.index, elim_scores.values, color=COLORS['secondary'], alpha=0.8)
ax.set_title('(f) Avg Score of Eliminated Contestants by Week')

plt.tight_layout()
plt.savefig('fig3_key_insights_summary.png')
plt.close()

print(f"\n可视化完成。生成文件: fig1_score_heatmap.png, fig2_controversy_analysis.png, fig3_key_insights_summary.png")
