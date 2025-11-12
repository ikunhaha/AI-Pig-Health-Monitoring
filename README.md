markdown
# 基于AI的养猪场疫病预警模拟系统

## 项目简介
本项目是一个基于Python和机器学习的数据驱动方案，旨在模拟和验证AI技术在养猪场疫病早期预警中的应用。通过分析模拟的猪只行为数据（躺卧时间、进食次数、体温），构建随机森林分类模型，实现对健康与风险猪只的自动判别。

## 技术栈
*   **编程语言**: Python
*   **核心库**: Pandas, NumPy, Scikit-learn, Matplotlib, Joblib
*   **算法**: 随机森林 (Random Forest)

## 文件结构
AI-Pig-Health-Monitoring/
│
├── pig_farm_data.csv # 模拟生成的猪场数据集
├── pig_health_predictor.pkl # 训练好的AI模型文件
├── create_data.py # 数据生成脚本
├── train_ai_model.py # 模型训练与评估脚本
├── create_visualizations.py # 结果可视化脚本
└── README.md # 项目说明文件 (本文件)


## 快速开始
1.  **克隆本项目**: `git clone https://github.com/[你的用户名]/AI-Pig-Health-Monitoring.git`
2.  **安装依赖**: `pip install pandas numpy scikit-learn matplotlib`
3.  **运行流程**:
    - 首先运行 `python create_data.py` 生成数据。
    - 然后运行 `python train_ai_model.py` 训练并保存模型。
    - 最后运行 `python create_visualizations.py` 生成分析图表。

## 项目背景
此项目作为【你的论文题目】的支撑材料，演示了如何将畜牧学知识转化为数据标准，并利用入门级AI工具实现疫病风险预警的逻辑。
