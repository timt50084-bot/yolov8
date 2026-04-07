# Stage 3 RGB-IR Train-Assist OBB Model

## 1. 本阶段目标

阶段3只新增一条显式 opt-in 的训练期 RGB-IR 协同分支：

- 训练时消费 `img_ir`
- 主部署路径仍然是 RGB-only OBB
- 暂时不消费 `img_prev`
- 不改默认 OBBTrainer / OBBModel / track 主线

## 2. 为什么采用训练期 RGB-IR 协同，而不是部署期双输入

本项目的硬约束是：

- IR 只用于训练期协同
- 推理、验证、部署只依赖 RGB

因此阶段3不把模型改成 6 通道或双图强依赖检测器，而是保留原生 RGB OBB 图作为真正部署主线，只在训练时引入轻量 IR 辅助。

## 3. 新模型总体结构

Stage 3 新增的是一个独立模型子类：

- [rgbir_obb_train.py](D:/project/ultralytics-main/ultralytics/models/yolo/obb/rgbir_obb_train.py)

它继承原生 `OBBModel`，保留原始 backbone / neck / head 图结构不变。训练时在选定 RGB stage 上：

1. 从 `img_ir` 通过轻量 stage adapter 提取对应尺寸的 IR feature  
2. 计算一个很小的 RGB-IR 对齐辅助损失  
3. 按配置执行轻量 gated residual fusion 或保守旁路  

验证 / 推理时：

- `img_ir` 不参与前向
- 模型自动退化为原生 RGB-only OBB 路径

## 4. IR 分支的职责

IR 分支不是新的部署 backbone，而是训练期辅助支路：

- 输入：`img_ir`
- 输出：与指定 RGB stage 对齐的轻量 IR 特征
- 作用：
  - 给 RGB feature 提供保守 residual 辅助
  - 提供小权重对齐损失，约束 RGB 表征吸收 IR 线索

这条支路在 eval / predict 中自动旁路。

## 5. 融合或辅助监督发生在什么位置

当前阶段默认在 `ir_feature_stages: [6, 9]` 执行训练期协同，属于中高层特征位点。

支持的配置包括：

- `fusion_type: gated_add`
- `fusion_type: weighted_sum`
- `fusion_type: align_only`
- `fusion_type: none`

其中：

- `gated_add` 是默认策略，风险最低
- `align_only` 只保留辅助对齐损失，不改主前向特征
- `none` 完全关闭协同，退化为 RGB-only

## 6. 为什么当前阶段不消费 img_prev

`img_prev` 已经由阶段2 dataset 稳定提供，但阶段3只解决 RGB-IR 协同，不做时序建模。

因此当前策略是：

- dataset / batch 中允许存在 `img_prev`
- trainer 会把它规范化并透传
- model 当前不消费它，也不会因为它存在而报错

时序消费留到阶段5实现，避免本阶段把时序、融合、训练数值风险同时耦合进来。

## 7. 如何显式启用新 trainer / 新脚本 / 新配置

### 自定义 trainer

```python
from ultralytics.models.yolo.obb.rgbir_train import RGBIROBBTrainer

trainer = RGBIROBBTrainer(
    overrides={
        "model": r"D:\project\ultralytics-main\ultralytics\cfg\models\v8\yolov8-rgbir-obb.yaml",
        "data": r"D:\project\ultralytics-main\runs\stage1_preprocess_smoke\prepared_sample\data\uav_rgb_obb.yaml",
        "epochs": 1,
        "batch": 2,
        "imgsz": 640,
        "device": "cpu",
        "use_rgbir_train_assist": True,
    }
)
trainer.train()
```

### 训练脚本

```bash
python tools/train_uav_rgbir_obb.py ^
  --data D:\project\ultralytics-main\runs\stage1_preprocess_smoke\prepared_sample\data\uav_rgb_obb.yaml ^
  --model D:\project\ultralytics-main\ultralytics\cfg\models\v8\yolov8-rgbir-obb.yaml ^
  --epochs 1 ^
  --batch 2 ^
  --device cpu ^
  --val
```

## 8. baseline 与新路径的关系

- baseline `OBBTrainer` 不变
- baseline `OBBModel` 不变
- baseline `build_yolo_dataset()` 不变
- baseline `yolo obb train/val/predict/track` 默认语义不变

阶段3新增的是：

- 显式自定义 model 子类
- 显式自定义 trainer
- 显式训练脚本

只有主动使用这条分支时，阶段2 dataset 和 `img_ir` 才会被消费。

## 9. 后续阶段如何在此基础上加小目标优化、时序和 tracking

后续扩展建议：

- 阶段4小目标优化：
  - 在当前 RGB 主干和 OBB loss 主线基础上做更小范围改进
- 阶段5轻量时序：
  - 开始实际消费 `img_prev`
  - 在当前 dataset batch 契约上增加 temporal refine
- 阶段6 tracking：
  - 继续复用 RGB-only 推理主线输出
  - 再做时序/轨迹级工程扩展

这样可以保持：

- 训练期 RGB-IR 协同
- 推理期 RGB-only
- 每个阶段只新增一类主要风险
