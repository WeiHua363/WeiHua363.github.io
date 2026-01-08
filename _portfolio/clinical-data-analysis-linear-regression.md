---
title: "PICU数据分析完整项目"
collection: portfolio
type: "Data Analysis"
permalink: /portfolio/picu-data-complete
date: 2026-01-08
excerpt: "本项目通过Python的Pandas、Matplotlib、Seaborn和Scikit-learn库，对PICU数据进行全面的分析和建模，包括数据读取、预处理、可视化、模型构建和评估等环节。"
header:
  teaser: /images/portfolio/picu-data-complete/teaser.png
tags:
  - PICU数据分析
  - Python
  - 机器学习
tech_stack:
  - name: Python
  - name: Pandas
  - name: Matplotlib
  - name: Seaborn
  - name: Scikit-learn
---

## 项目背景
本项目是一个完整的PICU数据分析项目，旨在通过Python的Pandas、Matplotlib、Seaborn和Scikit-learn库，对PICU数据进行全面的分析和建模，包括数据读取、预处理、可视化、模型构建和评估等环节。

## 核心实现
### 数据读取与探索
```python
# 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 读取Excel数据文件
path = 'data.xlsx'
picu_data = pd.read_excel(path)

# 查看数据基本信息
print(picu_data.shape)
print(picu_data.columns)
print(picu_data.isnull().sum())
print(Counter(picu_data["HOSPITAL_EXPIRE_FLAG"]))

### 数据探索 - 描述性统计与可视化
print(picu_data.mean())
print(picu_data.median())
print(picu_data.var())

# 绘制直方图
picu_data.hist(figsize=(12, 8))
plt.suptitle('直方图')
plt.show()
![直方图](../images/portfolio/picu-data-complete/histogram.png)

# 绘制箱线图
plt.figure(figsize=(12, 8))
sns.boxplot(data=picu_data)
plt.title('箱线图')
plt.show()


### 模型构建与评估
```python
# 导入机器学习相关模块
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.cluster import KMeans

# 数据预处理
# 填充缺失值
imputer = SimpleImputer(strategy='median')
picu_data_imputed = imputer.fit_transform(picu_data)

# 标准化数据
scaler = StandardScaler()
picu_data_scaled = scaler.fit_transform(picu_data_imputed)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(picu_data_scaled, picu_data['HOSPITAL_EXPIRE_FLAG'], test_size=0.2, random_state=42)

# 构建逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# 绘制混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('逻辑回归模型 混淆矩阵')
plt.xticks([0.5, 1.5], ['存活(0)', '死亡(1)'])
plt.yticks([0.5, 1.5], ['存活(0)', '死亡(1)'])
plt.show()

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('逻辑回归模型 ROC曲线')
plt.legend()
plt.show()
