# 1. 引入工具库
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 设置中文字体（避免显示乱码，如果运行出错可以删除这行或告诉我）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 加载数据
df = pd.read_csv('pig_farm_data.csv')
df['date'] = pd.to_datetime(df['date'])  # 确保日期为正确的格式

print("开始生成可视化图表...")

# 3. 创建一个大画布，包含多个子图表
fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # 2行2列，共4个图表
fig.suptitle('🐷 养殖场AI疫病预警分析报告', fontsize=16, fontweight='bold')

# ==================== 图表1：整体健康状况分布 ====================
print("生成图表1：健康状况分布...")
health_counts = df['health_status'].value_counts()
axes[0, 0].pie(health_counts.values, labels=health_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66c2a5', '#fc8d62'])
axes[0, 0].set_title('健康与风险猪只比例分布')

# ==================== 图表2：关键指标随时间变化（重点看异常猪） ====================
print("生成图表2：关键指标趋势...")
# 提取Pig_03和Pig_01的数据进行对比
pig_03_data = df[df['pig_id'] == 'Pig_03']
pig_01_data = df[df['pig_id'] == 'Pig_01']

# 绘制躺卧时间对比
axes[0, 1].plot(pig_01_data['date'], pig_01_data['lying_time'], marker='o', label='Pig_01 (健康)', color='green')
axes[0, 1].plot(pig_03_data['date'], pig_03_data['lying_time'], marker='s', label='Pig_03 (风险)', color='red')
axes[0, 1].set_ylabel('躺卧时间 (小时)')
axes[0, 1].set_title('异常与健康猪只行为对比（躺卧时间）')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, linestyle='--', alpha=0.7)

# ==================== 图表3：体温分布箱型图 ====================
print("生成图表3：体温分布...")
health_data = df[df['health_status'] == 'healthy']['temperature']
risk_data = df[df['health_status'] == 'risk']['temperature']
box_plot_data = [health_data, risk_data]
box = axes[1, 0].boxplot(box_plot_data, labels=['健康', '风险'], patch_artist=True)
# 给箱型图上色
colors = ['#66c2a5', '#fc8d62']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
axes[1, 0].set_ylabel('体温 (°C)')
axes[1, 0].set_title('健康与风险猪只体温分布对比')
axes[1, 0].grid(True, linestyle='--', alpha=0.7)

# ==================== 图表4：多特征散点图 ====================
print("生成图表4：多特征散点图...")
scatter = axes[1, 1].scatter(df['lying_time'], df['temperature'], c=df['eating_count'], s=df['temperature']*10,
            cmap='viridis', alpha=0.6)
axes[1, 1].set_xlabel('躺卧时间 (小时)')
axes[1, 1].set_ylabel('体温 (°C)')
axes[1, 1].set_title('多维度特征分析（点大小=体温，颜色=进食次数）')

# 添加颜色条
cbar = plt.colorbar(scatter, ax=axes[1, 1])
cbar.set_label('进食次数')

# 标记出异常区域
axes[1, 1].axvspan(16, 20, alpha=0.2, color='red', label='异常躺卧区')
axes[1, 1].axhspan(40.0, 41.0, alpha=0.2, color='orange', label='异常体温区')
axes[1, 1].legend()

# 4. 自动调整布局，保存图表
plt.tight_layout()
chart_filename = 'ai_farming_analysis_report.png'
plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
print(f"\n✅ 所有图表已生成并保存为 '{chart_filename}'！")

# 5. 在PyCharm中显示图表（可选）
plt.show()

print("\n" + "="*60)
print("🎉 恭喜！你的AI疫病预警项目全部完成！")
print("="*60)
print("\n你现在拥有：")
print("1. 📊 模拟数据集 (pig_farm_data.csv)")
print("2. 🤖 训练好的AI模型 (pig_health_predictor.pkl)")
print("3. 📈 专业的分析报告图表 (ai_farming_analysis_report.png)")
print("\n这些成果完全可以写入你的论文中！")