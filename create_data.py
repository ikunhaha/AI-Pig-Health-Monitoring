# 引入我们安装好的“工具箱”
import pandas as pd
import numpy as np

# 1. 设置随机种子，这样每次生成的数据都一样，便于你复现结果
np.random.seed(42)

# 2. 定义基础参数：我们模拟5头猪，连续7天的数据
num_pigs = 5
num_days = 7
pig_ids = [f'Pig_{i+1:02d}' for i in range(num_pigs)]  # 生成猪只编号：Pig_01, Pig_02...
dates = pd.date_range(start='2024-10-01', periods=num_days)  # 生成连续7天的日期

# 3. 创建一个空列表来存放所有数据
data = []

# 4. 循环生成每一头猪、每一天的数据
for date in dates:
    for pig_id in pig_ids:
        # 为健康猪只生成随机数据（基于《猪生产学》教材标准）
        base_lying_time = np.random.normal(13, 1)  # 躺卧时间：围绕13小时正常波动
        base_eating_count = np.random.randint(4, 6) # 进食次数：4到5次之间
        base_temperature = np.random.normal(39.0, 0.2) # 体温：围绕39.0°C正常波动

        # 5. 特意为“Pig_03”和“Pig_05”制造一些异常数据，用来给AI预警
        if pig_id == 'Pig_03' and date.day > 3:  # 从10月4号开始，Pig_03开始异常
            lying_time = np.random.normal(18, 1)   # 躺卧时间暴增到18小时左右
            eating_count = np.random.randint(0, 2) # 进食次数骤减到0-1次
            temperature = np.random.normal(40.5, 0.2) # 体温升高到40.5°C左右
            health_status = 'risk' # 标记为“风险”
        elif pig_id == 'Pig_05' and date.day > 5: # 从10月6号开始，Pig_05开始异常
            lying_time = np.random.normal(16, 1)
            eating_count = np.random.randint(1, 3)
            temperature = np.random.normal(40.0, 0.2)
            health_status = 'risk'
        else:
            # 其他情况，都是健康数据
            lying_time = max(0, base_lying_time)  # 确保时间不为负数
            eating_count = base_eating_count
            temperature = base_temperature
            health_status = 'healthy' # 标记为“健康”

        # 6. 将生成好的这条数据记录加到列表里
        data.append({
            'date': date,
            'pig_id': pig_id,
            'lying_time': round(lying_time, 2),
            'eating_count': eating_count,
            'temperature': round(temperature, 2),
            'health_status': health_status
        })

# 7. 将列表转换为Pandas DataFrame（这就是我们核心的表格数据结构）
df = pd.DataFrame(data)

# 8. 将表格数据保存到本地的CSV文件
df.to_csv('pig_farm_data.csv', index=False)

# 9. 打印前10行数据，让我们看看生成的数据长什么样
print("模拟猪场数据已生成并保存为 'pig_farm_data.csv'！")
print("\n数据预览（前10行）:")
print(df.head(10))

# 10. 简单查看一下数据的统计信息
print("\n数据统计信息：")
print(df.describe())

# 11. 特别查看一下风险猪只的数据
print("\n----- 风险个体筛查 -----")
risk_pigs = df[df['health_status'] == 'risk']
print(risk_pigs)