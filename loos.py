import pandas as pd
import matplotlib.pyplot as plt

# 修复字体（避免中文/符号显示问题）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 读取训练日志
df = pd.read_csv('D:\\project\\ultralytics-main\\runs\\detect\\drone_vehicle_train\\yolov8_drone_crossmodal2\\results.csv')

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train/box_loss'], label='训练框损失', color='blue')
plt.plot(df['epoch'], df['val/box_loss'], label='验证框损失', color='red')
plt.xlabel('Epoch')
plt.ylabel('损失值')
plt.title('训练/验证框损失曲线')
plt.legend()
plt.savefig('loss_curve_fixed.png')
plt.show()

# 绘制 mAP 曲线（模型精度）
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5-0.95', color='orange')
plt.xlabel('Epoch')
plt.ylabel('mAP 值（越高越好）')
plt.title('模型精度曲线')
plt.legend()
plt.savefig('map_curve.png')
plt.show()