import pandas as pd
import numpy as np

# 读取预测结果
pred_df = pd.read_csv('notebooks/pred.csv')
predictions = pred_df['tested_positive'].values

# 读取真实答案
true_df = pd.read_csv('Data/covid.test.true.csv')
# 最后一列是真实的tested_positive值
true_values = true_df.iloc[:, -1].values

# 确保数据长度一致
print(f"预测值数量: {len(predictions)}")
print(f"真实值数量: {len(true_values)}")

if len(predictions) != len(true_values):
    min_len = min(len(predictions), len(true_values))
    predictions = predictions[:min_len]
    true_values = true_values[:min_len]
    print(f"调整后数量: {min_len}")

# 计算MSE
mse = np.mean((predictions - true_values) ** 2)

print(f"\n=== MSE 计算结果 ===")
print(f"均方误差 (MSE): {mse:.6f}")
print(f"均方根误差 (RMSE): {np.sqrt(mse):.6f}")

# 显示一些统计信息
print(f"\n=== 统计信息 ===")
print(f"预测值范围: {predictions.min():.2f} ~ {predictions.max():.2f}")
print(f"真实值范围: {true_values.min():.2f} ~ {true_values.max():.2f}")
print(f"预测值平均: {predictions.mean():.2f}")
print(f"真实值平均: {true_values.mean():.2f}")

# 计算平均绝对误差
mae = np.mean(np.abs(predictions - true_values))
print(f"平均绝对误差 (MAE): {mae:.6f}")

# 保存详细结果
results_df = pd.DataFrame({
    'id': range(len(predictions)),
    'predicted': predictions,
    'true': true_values,
    'error': predictions - true_values,
    'abs_error': np.abs(predictions - true_values),
    'squared_error': (predictions - true_values) ** 2
})

results_df.to_csv('mse_analysis.csv', index=False)
print(f"\n详细分析结果已保存到 mse_analysis.csv")