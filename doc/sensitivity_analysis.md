# 敏感性分析说明文档

本文档说明敏感性分析脚本的功能、输出内容以及每张图/表的含义，并给出最终结论解读。  
脚本入口：`solution/sensitivity_analysis.py`

---

## 1. 分析目标与范围

本次敏感性分析聚焦三个关键参数族，目的是检验模型输出对参数扰动的敏感程度，从而评估模型稳健性。

- **Q1 融合权重**：`p_survive_next` 与评委评分（`judge_total_week_z`）的融合权重  
- **Q4 gamma**：SHAP趋势权重与熵权的融合比例  
- **Q3 XGBoost超参数**：`max_depth`、`learning_rate`、`n_estimators`

核心问题：**这些参数变化是否显著改变结果，模型是否稳健？**

---

## 2. 程序做了什么

脚本执行流程概览：

1. **Q1**  
   - 训练增强版投票估计模型  
   - 改变融合权重 `w ∈ [0,1]`（以 0.1 为步长）  
   - 重新计算 `vote_share_hat` 和 `votes_hat`  
   - 计算淘汰一致性指标（Exact/Bottom-2/Hit）  

2. **Q4**  
   - 训练 RankSHAP 趋势权重  
   - 遍历 gamma 值并重建权重  
   - 计算三指标（公平性、民意性、观赏性）  
   - **Bootstrap** 估计三指标均值的置信区间  
   - 绘制 Pareto 前沿（公平性 vs 民意性 vs 观赏性）  

3. **Q3**  
   - 固定数据与特征工程  
   - 单参数扫描超参数  
   - 记录 Judge/Fan 两个目标的 R²  
   - 记录 **NRMSE** 与 **MAPE**（统一量纲，避免RMSE不可比）  
   - 给出敏感度指数并添加阈值线（低敏感性基准）  

---

## 3. 输出文件说明

### 3.1 CSV 结果表（在 `outputs/sensitivity_analysis/`）

- `q1_fusion_sensitivity.csv`  
  不同融合权重下的淘汰一致性指标：
  - `exact_elim`：淘汰周精确匹配率  
  - `bottom2_cover`：淘汰者是否落在底部2的覆盖率  
  - `hit_all_true`：淘汰者全覆盖命中率  

- `q4_gamma_sensitivity.csv`  
  不同 gamma 下的三指标 + 置信区间：
  - `fairness_rate`：公平性（越低越好）  
  - `popularity_fan_ndcg`：民意性（NDCG@K，越高越好）  
  - `excitement_reversal_rate`：观赏性（逆转率，越高越好）  
  - `*_ci_low/high`：bootstrap 置信区间  

- `q3_xgb_sensitivity.csv`  
  单参数扫描结果：
  - `judge_r2`, `fan_r2`  
  - `judge_nrmse`, `fan_nrmse`  
  - `judge_mape`, `fan_mape`  

- `sensitivity_summary.txt`  
  自动生成的结论摘要，便于直接引用到报告正文。

---

## 4. 图表逐张说明（在 `plots/sensitivity_analysis/`）

### 图 1：`q1_fusion_sensitivity.png`

左图：不同融合权重下的三项淘汰一致性指标曲线  
右图：相对中位权重（w≈0.5）变化幅度（%）

解读方式：
- 曲线越平坦 ⇒ 对权重不敏感 ⇒ **模型稳健**  
- 右图变化幅度若长期接近 0 ⇒ **稳定性强**

### 图 2：`q3_xgb_sensitivity_r2.png`

上排：Judge/Fan 两个模型的 R² 随超参数变化  
下排：Judge/Fan 的 NRMSE（已归一化，量纲可比）

解读方式：
- 若 R² 变化不大 ⇒ 对超参数不敏感  
- NRMSE 已消除量纲差异，避免 RMSE 误导  

### 图 3：`q3_xgb_sensitivity_index.png`

超参数敏感度指数柱状图  
虚线阈值=0.10（经验上低敏感）

解读方式：
- 小于阈值 ⇒ **低敏感 / 稳健**  
- 明显高于阈值 ⇒ **该参数更关键**

### 图 4：`q4_gamma_sensitivity.png`

左图：三指标归一化后的曲线 + bootstrap 置信带  
右图：三指标随 gamma 的归一化热力图

解读方式：
- 指标曲线若平滑且 CI 窄 ⇒ **稳健**  
- 热力图展示整体趋势一致性  

### 图 5：`q4_gamma_pareto.png`

二维投影：X=民意性，Y=观赏性，颜色=公平性  
黑边点为 Pareto 前沿（不可被同时超越）

解读方式：
- Pareto 点代表多目标折中最优  
- 若 Pareto 点集中在狭窄区域 ⇒ 权重选择对整体结论影响有限  

---

## 5. 总体结论（可直接写入正文）

1. **Q1 融合权重对淘汰一致性影响较小**  
   三项指标曲线整体平坦，相对变化幅度有限，说明模型对融合权重具有较强稳健性。

2. **Q3 中 learning_rate 更敏感，max_depth 与 n_estimators 影响较弱**  
   敏感度指数显示 learning_rate 明显高于其他参数，但整体变化仍处于可控范围，模型训练稳定。

3. **Q4 gamma 的三指标变化平滑且置信区间窄**  
   bootstrap CI 带较紧，表明结果对 gamma 的依赖不强；Pareto 前沿集中，显示多目标权衡稳定。

4. **总体结论：模型稳健性良好**  
   在合理参数范围内，核心结论与指标变化不剧烈，可认为模型对参数扰动具有鲁棒性。

---


