# 1. 引入所需的工具库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 2. 加载我们上一步生成的模拟数据
print("🚀 开始加载猪场数据...")
df = pd.read_csv('pig_farm_data.csv')  # 读取CSV文件
print("数据加载成功！")
print("数据前几行：")
print(df.head())

# 3. 准备“特征”和“标签”
# “特征”：模型用来做判断的依据（X）
# “标签”：我们想要模型预测的结果（y）
print("\n🔧 准备训练数据...")
X = df[['lying_time', 'eating_count', 'temperature']]  # 选择这三个列作为特征
y = df['health_status']  # 将健康状态作为我们要预测的标签

print("特征数据 (X):")
print(X.head())
print("\n标签数据 (y):")
print(y.head())

# 4. 分割数据集
# 将数据分为“训练集”和“测试集”，用大部分数据训练模型，留下一小部分检验它学得好不好
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n📊 数据分割完成：")
print(f"  训练集样本数：{len(X_train)}")
print(f"  测试集样本数：{len(X_test)}")

# 5. 创建并训练AI模型
# 我们使用“随机森林”算法，它是一种强大且易于使用的分类器
print("\n🤖 开始训练AI预警模型...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # 这就是“训练”的过程，模型从数据中学习规律
print("模型训练完成！")

# 6. 评估模型性能：看看它在测试集上的表现
print("\n📈 评估模型性能...")
y_pred = model.predict(X_test)  # 让模型对没见过的测试数据进行预测

accuracy = accuracy_score(y_test, y_pred)  # 计算准确率
print(f"模型准确率：{accuracy:.2%}")  # 格式化输出为百分比

print("\n详细分类报告：")
print(classification_report(y_test, y_pred))

# 7. 保存训练好的模型，这样以后就可以直接使用，无需重新训练
model_filename = 'pig_health_predictor.pkl'
joblib.dump(model, model_filename)
print(f"\n💾 模型已保存为 '{model_filename}'")

# 8. 【核心演示】模拟真实预警场景！
print("\n" + "="*50)
print("🚨 AI疫病预警系统模拟演示 🚨")
print("="*50)

# 模拟一批新的、模型没见过的猪只数据
new_pigs_data = {
    'lying_time': [12.5, 18.2, 13.1, 9.8],  # 躺卧时间：第2头猪异常
    'eating_count': [5, 1, 4, 2],           # 进食次数：第2头猪异常
    'temperature': [39.1, 40.8, 39.2, 39.5] # 体温：第2头猪异常
}
new_pigs_df = pd.DataFrame(new_pigs_data)

print("接收到新的猪只监测数据：")
print(new_pigs_df)

# 使用训练好的模型进行预测
predictions = model.predict(new_pigs_df)
risk_proba = model.predict_proba(new_pigs_df)  # 获取预测的概率

# 打印预警结果
print("\n--- AI预警报告 ---")
for i, (pred, prob) in enumerate(zip(predictions, risk_proba)):
    risk_prob = prob[1] if model.classes_[1] == 'risk' else prob[0]  # 获取“风险”的概率
    status = "🔴 风险预警！" if pred == 'risk' else "🟢 健康"
    print(f"猪只 #{i+1}: {status} (风险概率：{risk_prob:.1%})")
    if pred == 'risk':
        print(f"       建议：立即隔离并检查猪只 #{i+1}！")