"""
MCM 2026 Problem C - DWTS Data Visualization Script
====================================================
数据可视化脚本：生成论文所需的各类图表

输入: 
  - dwts_cleaned.csv
  - dwts_long_format.csv
  - dwts_summary.csv
输出: 
  - 各类PNG图表文件
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

# MCM竞赛推荐配色
COLORS = {
    'primary': '#2E86AB',      # 蓝色
    'secondary': '#A23B72',    # 紫红
    'accent': '#F18F01',       # 橙色
    'success': '#C73E1D',      # 红色
    'neutral': '#3B3B3B',      # 深灰
    'light': '#E8E8E8'         # 浅灰
}

PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4226', '#1B998B', '#E55934', '#7D70BA']

# 加载数据
try:
    df = pd.read_csv('dwts_cleaned.csv')
    df_long = pd.read_csv('dwts_long_format.csv')
    df_summary = pd.read_csv('dwts_summary.csv')
except FileNotFoundError:
    print("Error: 清洗后的数据文件未找到，请先运行 data_cleaning.py")
    exit(1)

# 图1: 数据集概览
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1a: 每季选手数
season_counts = df.groupby('season').size()
ax1 = axes[0]
bars = ax1.bar(season_counts.index, season_counts.values, color=COLORS['primary'], alpha=0.8, edgecolor='white')
ax1.set_xlabel('Season')
ax1.set_ylabel('Number of Contestants')
ax1.set_title('(a) Contestants per Season')
ax1.axhline(y=season_counts.mean(), color=COLORS['accent'], linestyle='--', linewidth=2, label=f'Average: {season_counts.mean():.1f}')
ax1.legend()
ax1.set_xticks(range(1, 35, 3))

# 1b: 每季最大周数
max_weeks = df.groupby('season')['weeks_participated'].max()
ax2 = axes[1]
ax2.bar(max_weeks.index, max_weeks.values, color=COLORS['secondary'], alpha=0.8, edgecolor='white')
ax2.set_xlabel('Season')
ax2.set_ylabel('Maximum Weeks')
ax2.set_title('(b) Competition Length per Season')
ax2.axhline(y=max_weeks.mean(), color=COLORS['accent'], linestyle='--', linewidth=2, label=f'Average: {max_weeks.mean():.1f}')
ax2.legend()
ax2.set_xticks(range(1, 35, 3))

plt.tight_layout()
plt.savefig('fig1_dataset_overview.png')
plt.close()

# 图2: 评委得分分布热力图
fig, ax = plt.subplots(figsize=(14, 8))

# 创建每周每季的平均得分矩阵
pivot_data = df_long.pivot_table(
    values='judge_total', 
    index='season', 
    columns='week', 
    aggfunc='mean'
)

# 绘制热力图
sns.heatmap(pivot_data, cmap='YlOrRd', annot=False, fmt='.1f', 
            cbar_kws={'label': 'Average Judge Total Score'},
            ax=ax, linewidths=0.5, linecolor='white')
ax.set_xlabel('Week')
ax.set_ylabel('Season')
ax.set_title('Average Judge Total Score by Season and Week')

plt.tight_layout()
plt.savefig('fig2_score_heatmap.png')
plt.close()

# 图3: 职业类型分析
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 3a: 职业分布
industry_counts = df['celebrity_industry'].value_counts().head(10)
ax1 = axes[0]
bars = ax1.barh(range(len(industry_counts)), industry_counts.values, color=PALETTE[:len(industry_counts)])
ax1.set_yticks(range(len(industry_counts)))
ax1.set_yticklabels(industry_counts.index)
ax1.set_xlabel('Number of Contestants')
ax1.set_title('(a) Top 10 Celebrity Industries')
ax1.invert_yaxis()

# 添加数值标签
for i, v in enumerate(industry_counts.values):
    ax1.text(v + 1, i, str(v), va='center', fontsize=9)

# 3b: 职业与平均排名
industry_stats = df.groupby('celebrity_industry').agg({
    'placement': 'mean',
    'celebrity_name': 'count'
}).rename(columns={'celebrity_name': 'count'})
industry_stats = industry_stats[industry_stats['count'] >= 5].sort_values('placement')

ax2 = axes[1]
colors = [COLORS['success'] if p < industry_stats['placement'].median() else COLORS['primary'] 
          for p in industry_stats['placement'].values]
bars = ax2.barh(range(len(industry_stats)), industry_stats['placement'].values, color=colors)
ax2.set_yticks(range(len(industry_stats)))
ax2.set_yticklabels(industry_stats.index)
ax2.set_xlabel('Average Final Placement (lower is better)')
ax2.set_title('(b) Average Placement by Industry (n≥5)')
ax2.invert_yaxis()
ax2.axvline(x=industry_stats['placement'].median(), color=COLORS['neutral'], linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('fig3_industry_analysis.png')
plt.close()

# 图4: 年龄分析
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 4a: 年龄分布
ax1 = axes[0]
ax1.hist(df['celebrity_age_during_season'].dropna(), bins=20, color=COLORS['primary'], 
         alpha=0.7, edgecolor='white')
ax1.axvline(x=df['celebrity_age_during_season'].median(), color=COLORS['accent'], 
            linestyle='--', linewidth=2, label=f'Median: {df["celebrity_age_during_season"].median():.0f}')
ax1.set_xlabel('Age')
ax1.set_ylabel('Number of Contestants')
ax1.set_title('(a) Age Distribution of Contestants')
ax1.legend()

# 4b: 年龄与排名的关系
ax2 = axes[1]
age_placement = df.groupby('age_group')['placement'].mean().dropna()
ax2.bar(range(len(age_placement)), age_placement.values, color=PALETTE[:len(age_placement)])
ax2.set_xticks(range(len(age_placement)))
ax2.set_xticklabels(age_placement.index)
ax2.set_xlabel('Age Group')
ax2.set_ylabel('Average Placement (lower is better)')
ax2.set_title('(b) Average Placement by Age Group')

# 添加趋势线
z = np.polyfit(range(len(age_placement)), age_placement.values, 1)
p = np.poly1d(z)
ax2.plot(range(len(age_placement)), p(range(len(age_placement))), 
         color=COLORS['success'], linestyle='--', linewidth=2, label='Trend')
ax2.legend()

plt.tight_layout()
plt.savefig('fig4_age_analysis.png')
plt.close()

# 图5: 舞伴分析
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 5a: 最成功的舞伴（按平均排名）
partner_stats = df.groupby('ballroom_partner').agg({
    'placement': 'mean',
    'celebrity_name': 'count',
    'avg_judge_total': 'mean'
}).rename(columns={'celebrity_name': 'count'})
partner_stats = partner_stats[partner_stats['count'] >= 5].sort_values('placement').head(10)

ax1 = axes[0]
bars = ax1.barh(range(len(partner_stats)), partner_stats['placement'].values, color=COLORS['primary'])
ax1.set_yticks(range(len(partner_stats)))
ax1.set_yticklabels([f"{name} (n={int(partner_stats.loc[name, 'count'])})" 
                     for name in partner_stats.index])
ax1.set_xlabel('Average Placement (lower is better)')
ax1.set_title('(a) Top 10 Most Successful Partners')
ax1.invert_yaxis()

# 5b: 舞伴经验与结果
ax2 = axes[1]
exp_groups = df.groupby('partner_prev_seasons').agg({
    'placement': 'mean',
    'celebrity_name': 'count'
}).rename(columns={'celebrity_name': 'count'})
exp_groups = exp_groups[exp_groups['count'] >= 5]

ax2.scatter(exp_groups.index, exp_groups['placement'], 
            s=exp_groups['count']*10, c=COLORS['secondary'], alpha=0.6)
ax2.set_xlabel("Partner's Previous Seasons")
ax2.set_ylabel('Average Placement')
ax2.set_title('(b) Partner Experience vs Contestant Placement')

# 添加趋势线
z = np.polyfit(exp_groups.index, exp_groups['placement'], 1)
p = np.poly1d(z)
ax2.plot(exp_groups.index, p(exp_groups.index), color=COLORS['accent'], 
         linestyle='--', linewidth=2, label=f'Trend (slope={z[0]:.2f})')
ax2.legend()

plt.tight_layout()
plt.savefig('fig5_partner_analysis.png')
plt.close()

# 图6: 争议选手分析
controversy_names = ['Jerry Rice', 'Bristol Palin', 'Bobby Bones', 'Billy Ray Cyrus']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, name in enumerate(controversy_names):
    ax = axes[idx]
    
    # 获取选手数据
    contestant_data = df_long[df_long['celebrity_name'] == name].sort_values('week')
    
    if len(contestant_data) == 0:
        ax.text(0.5, 0.5, f'{name}\nNo data found', ha='center', va='center', fontsize=12)
        ax.set_title(name)
        continue
    
    season = contestant_data['season'].iloc[0]
    placement = contestant_data['placement'].iloc[0]
    
    # 获取该季所有选手的周平均分（用于对比）
    season_data = df_long[df_long['season'] == season]
    season_avg = season_data.groupby('week')['judge_total'].mean()
    
    # 绘制
    ax.plot(contestant_data['week'], contestant_data['judge_total'], 
            'o-', color=COLORS['success'], linewidth=2, markersize=8, label=name)
    ax.plot(season_avg.index, season_avg.values, 
            's--', color=COLORS['neutral'], alpha=0.5, label='Season Average')
    
    # 添加排名信息
    for _, row in contestant_data.iterrows():
        rank_text = f"#{int(row['weekly_rank'])}"
        ax.annotate(rank_text, (row['week'], row['judge_total']), 
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    ax.set_xlabel('Week')
    ax.set_ylabel('Judge Total Score')
    ax.set_title(f'{name} (Season {season}, Final: #{int(placement)})')
    ax.legend(loc='lower right')
    ax.set_xlim(0.5, contestant_data['week'].max() + 0.5)

plt.tight_layout()
plt.savefig('fig6_controversy_analysis.png')
plt.close()

# 图7: 评委得分趋势分析
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 7a: 整体周得分趋势
ax1 = axes[0]
weekly_stats = df_long.groupby('week').agg({
    'judge_total': ['mean', 'std', 'count']
}).round(2)
weekly_stats.columns = ['mean', 'std', 'count']

ax1.errorbar(weekly_stats.index, weekly_stats['mean'], 
             yerr=weekly_stats['std']/2, fmt='o-', color=COLORS['primary'],
             capsize=3, capthick=1, linewidth=2, markersize=8)
ax1.fill_between(weekly_stats.index, 
                 weekly_stats['mean'] - weekly_stats['std']/2,
                 weekly_stats['mean'] + weekly_stats['std']/2,
                 alpha=0.2, color=COLORS['primary'])
ax1.set_xlabel('Week')
ax1.set_ylabel('Judge Total Score')
ax1.set_title('(a) Average Judge Score by Week')

# 7b: 不同名次选手的得分曲线
ax2 = axes[1]
for place, color in zip([1, 2, 3], [COLORS['success'], COLORS['accent'], COLORS['primary']]):
    place_data = df_long[df_long['placement'] == place]
    place_avg = place_data.groupby('week')['judge_total'].mean()
    ax2.plot(place_avg.index, place_avg.values, 'o-', color=color, 
             linewidth=2, markersize=6, label=f'{place}st/nd/rd Place' if place <= 3 else f'{place}th Place')

# 添加淘汰选手平均
elim_data = df_long[df_long['placement'] > 5]
elim_avg = elim_data.groupby('week')['judge_total'].mean()
ax2.plot(elim_avg.index, elim_avg.values, 's--', color=COLORS['neutral'], 
         linewidth=1.5, markersize=5, alpha=0.7, label='Others (>5th)')

ax2.set_xlabel('Week')
ax2.set_ylabel('Judge Total Score')
ax2.set_title('(b) Score Trajectory by Final Placement')
ax2.legend()

plt.tight_layout()
plt.savefig('fig7_score_trends.png')
plt.close()

# 图8: 赛季演变分析
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 8a: 平均年龄演变
ax1 = axes[0, 0]
season_age = df.groupby('season')['celebrity_age_during_season'].mean()
ax1.plot(season_age.index, season_age.values, 'o-', color=COLORS['primary'], linewidth=2)
z = np.polyfit(season_age.index, season_age.values, 1)
p = np.poly1d(z)
ax1.plot(season_age.index, p(season_age.index), '--', color=COLORS['accent'], 
         label=f'Trend (slope={z[0]:.2f})')
ax1.set_xlabel('Season')
ax1.set_ylabel('Average Age')
ax1.set_title('(a) Average Contestant Age Over Seasons')
ax1.legend()

# 8b: 平均得分演变
ax2 = axes[0, 1]
season_score = df.groupby('season')['avg_judge_total'].mean()
ax2.plot(season_score.index, season_score.values, 'o-', color=COLORS['secondary'], linewidth=2)
ax2.set_xlabel('Season')
ax2.set_ylabel('Average Judge Total')
ax2.set_title('(b) Average Judge Score Over Seasons')

# 8c: 职业多样性演变
ax3 = axes[1, 0]
season_diversity = df.groupby('season')['celebrity_industry'].nunique()
ax3.bar(season_diversity.index, season_diversity.values, color=COLORS['primary'], alpha=0.7)
ax3.set_xlabel('Season')
ax3.set_ylabel('Number of Industries')
ax3.set_title('(c) Industry Diversity Over Seasons')

# 8d: 美国vs非美国选手比例
ax4 = axes[1, 1]
us_ratio = df.groupby('season')['is_us_contestant'].mean() * 100
ax4.bar(us_ratio.index, us_ratio.values, color=COLORS['accent'], alpha=0.7)
ax4.axhline(y=us_ratio.mean(), color=COLORS['neutral'], linestyle='--', 
            label=f'Average: {us_ratio.mean():.1f}%')
ax4.set_xlabel('Season')
ax4.set_ylabel('US Contestants (%)')
ax4.set_title('(d) Percentage of US Contestants')
ax4.legend()

plt.tight_layout()
plt.savefig('fig8_season_evolution.png')
plt.close()

# 图9: 相关性分析
fig, ax = plt.subplots(figsize=(10, 8))

# 选择数值特征
numeric_cols = ['celebrity_age_during_season', 'placement', 'weeks_participated',
                'avg_judge_total', 'std_judge_total', 'partner_prev_seasons']
corr_data = df[numeric_cols].dropna()
corr_data.columns = ['Age', 'Placement', 'Weeks', 'Avg Score', 'Score Std', 'Partner Exp']

corr_matrix = corr_data.corr()

# 绘制热力图
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=1, cbar_kws={'shrink': 0.8},
            ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlation Matrix of Key Variables')

plt.tight_layout()
plt.savefig('fig9_correlation.png')
plt.close()

# 图10: 每周淘汰分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 10a: 淘汰周分布
ax1 = axes[0]
elim_dist = df[df['elimination_week'] > 0]['elimination_week'].value_counts().sort_index()
ax1.bar(elim_dist.index, elim_dist.values, color=COLORS['primary'], alpha=0.8)
ax1.set_xlabel('Elimination Week')
ax1.set_ylabel('Number of Contestants')
ax1.set_title('(a) Distribution of Eliminations by Week')

# 10b: 淘汰选手的平均评委得分 vs 周数
ax2 = axes[1]
elim_scores = df[df['elimination_week'] > 0].groupby('elimination_week')['avg_judge_total'].mean()
ax2.bar(elim_scores.index, elim_scores.values, color=COLORS['secondary'], alpha=0.8)
ax2.set_xlabel('Elimination Week')
ax2.set_ylabel('Average Judge Total Score')
ax2.set_title('(b) Average Score of Eliminated Contestants by Week')

plt.tight_layout()
plt.savefig('fig10_elimination_analysis.png')
plt.close()

# 生成汇总报告
import os
output_dir = '.'
figures = [f for f in os.listdir(output_dir) if f.endswith('.png')]
print(f"\n可视化完成，共生成 {len(figures)} 个图表文件。")

# 输出关键发现摘要
print(f"\n【DWTS 可视化发现摘要】")

# 年龄发现
age_corr = df['celebrity_age_during_season'].corr(df['placement'])
print(f"\n1. 年龄与排名相关性: r={age_corr:.3f}")
print(f"   → 年轻选手表现更好 (≤25岁平均排名: {df[df['age_group']=='≤25']['placement'].mean():.1f})")

# 职业发现
best_industry = df.groupby('celebrity_industry')['placement'].mean().idxmin()
print(f"\n2. 最成功的职业类型: {best_industry}")

# 舞伴发现
best_partner = partner_stats.index[0]
print(f"\n3. 最成功的舞伴: {best_partner} (平均排名: {partner_stats['placement'].iloc[0]:.1f})")

# 争议选手
print(f"\n4. 争议选手案例:")
for name in controversy_names:
    contestant = df[df['celebrity_name'] == name]
    if len(contestant) > 0:
        c = contestant.iloc[0]
        print(f"   - {name} (S{int(c['season'])}): 最终#{int(c['placement'])}, 平均得分{c['avg_judge_total']:.1f}")
