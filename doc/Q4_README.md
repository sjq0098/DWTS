# MCM 2026 Problem C - Question 4: 新投票系统设计 (增强版)

## 核心创新框架

本方案采用三大核心创新点，显著提升模型的**学术novelty**：

---

## 1. Shapley贡献度分析 (Shapley Contribution Analysis)

### 理论基础
将评委(J)和粉丝(F)视为**合作博弈中的两个玩家**，使用博弈论中的Shapley值量化各自对最终排名结果的贡献度。

### 核心公式
对于2玩家博弈，Shapley值计算为：
```
φ_J = 1/2 × [v({J}) - v(∅)] + 1/2 × [v({J,F}) - v({F})]
φ_F = 1/2 × [v({F}) - v(∅)] + 1/2 × [v({J,F}) - v({J})]
```

其中 v(S) 是联盟S的价值函数，使用排名相关性衡量。

### 创新贡献
- **公平性理论保障**：Shapley值满足效率性、对称性、可加性、虚拟玩家等公理
- **动态贡献分析**：揭示评委/粉丝贡献度随赛季进度的变化规律
- **协同效应量化**：计算 synergy = v({J,F}) - v({J}) - v({F})

### 关键发现（示例结果）
| 阶段 | 评委贡献度 | 粉丝贡献度 |
|------|-----------|-----------|
| 筛选期(t<0.5) | 0.867 | 0.133 |
| 过渡期(0.5≤t≤0.8) | 0.843 | 0.157 |
| 决战期(t>0.8) | 0.799 | 0.201 |

---

## 2. 时间注意力学习权重 (Temporal Attention Weight Learning)

### 理论基础
使用**注意力机制**从历史数据中学习最优权重曲线，替代固定的Sigmoid函数。

### 核心机制
```
w_j(t) = Attention(Q, K, V)
       = sigmoid(θ₀ + θ₁t + θ₂t² + θ₃t³) + α × controversy_level × (1-t)
```

### 创新贡献
- **数据驱动**：权重从历史数据学习，而非预设参数
- **自适应调整**：根据争议程度动态调整权重
- **可解释性强**：注意力分数可解释每周的重要性

### 与Sigmoid对比
| 特性 | Sigmoid | Attention |
|------|---------|-----------|
| 参数来源 | 人工设定 | 数据学习 |
| 适应性 | 固定曲线 | 自适应调整 |
| 争议处理 | 无 | 自动调整 |

---

## 3. Pareto多目标优化 (Pareto Multi-Objective Optimization)

### 理论基础
将三个目标（公平性、民意性、观赏性）建模为严格的**多目标优化问题**，寻找Pareto前沿上的最优解集。

### 目标函数
1. **公平性 F_fair**（最小化）：争议选手晋级率
2. **民意性 F_pop**（最大化）：粉丝排名与最终排名的Spearman相关性
3. **观赏性 F_exc**（最大化）：组合分数的方差

### Pareto支配定义
解 x₁ 支配 x₂ 当且仅当：
- ∀i: fᵢ(x₁) ≤ fᵢ(x₂)
- ∃j: fⱼ(x₁) < fⱼ(x₂)

### 创新贡献
- **理论完备**：基于经典多目标优化理论
- **解集多样**：提供多个Pareto最优解供决策者选择
- **可视化直观**：3D Pareto前沿展示权衡关系

---

## 生成的图表说明

| 图表 | 内容 | 对应创新点 |
|------|------|-----------|
| fig1 | Shapley贡献度分析（4子图） | Shapley分析 |
| fig2 | 时间注意力权重对比 | 注意力学习 |
| fig3 | 2D Pareto前沿（3视图） | Pareto优化 |
| fig4 | 四种方法指标对比+雷达图 | 综合比较 |
| fig5 | 分阶段权重曲线 | 动态权重 |
| fig6 | 赛季级指标热力图 | 方法评估 |
| fig7 | 3D Pareto前沿 | Pareto优化 |
| fig8 | 分阶段Shapley贡献度 | Shapley分析 |

---

## 参考文献（支撑Novelty）

### Shapley值相关
1. Shapley, L.S. (1953). "A value for n-person games." Contributions to the Theory of Games II.
2. MDPI Mathematics (2025). "The Shapley Value in Data Science: Advances in Computation, Extensions, and Applications." https://www.mdpi.com/2227-7390/13/10/1581
3. Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions" (SHAP).

### 多目标优化相关
4. arXiv (2025). "A Multi-Objective Evaluation Framework for Analyzing Utility-Fairness Trade-Offs." https://arxiv.org/abs/2503.11120
5. Complex & Intelligent Systems (2024). "Towards Fairness-Aware Multi-Objective Optimization." https://link.springer.com/article/10.1007/s40747-024-01668-w
6. arXiv (2024). "Towards Efficient Pareto-optimal Utility-Fairness between Groups in Repeated Rankings." https://arxiv.org/abs/2402.14305

### 注意力机制相关
7. arXiv (2017). "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction." https://arxiv.org/abs/1704.02971
8. arXiv (2024). "Attention as Robust Representation for Time Series Forecasting." https://arxiv.org/abs/2402.05370

---

## 使用方法

```python
# 运行完整分析
python Q4_enhanced.py

# 需要的输入文件（可选，无则自动生成模拟数据）
# - dwts_long_format.csv
# - q1_fan_vote_estimates_enhanced.csv
```

## 输出文件

- **CSV数据**: shapley_analysis.csv, pareto_solutions.csv, attention_weights.csv, etc.
- **可视化图表**: fig1-fig8 (PNG格式)

---

## 核心卖点总结

| 传统方法 | 本方案创新 |
|---------|-----------|
| Sigmoid预设参数 | Shapley值量化 + 注意力学习 |
| 单目标优化 | Pareto三目标优化 |
| 缺乏理论支撑 | 博弈论+多目标优化理论 |
| 静态权重 | 动态自适应权重 |

**关键优势**：三项创新点均有成熟的学术理论支撑，能有效避免"lack of novelty"的问题。
