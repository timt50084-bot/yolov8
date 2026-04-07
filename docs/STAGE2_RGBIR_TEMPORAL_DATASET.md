# Stage 2 RGBIRTemporal Dataset

## 1. 本阶段目标

阶段2只落地一个可独立启用的 `RGBIRTemporalOBBDataset` 数据层，用于把阶段1已经标准化好的 RGB、IR、temporal index 和 canonical OBB 标签接成真实可读的数据集对象。当前阶段不改模型、不改 trainer、不改 loss。

## 2. 为什么当前仓库自带 dataset 不够用

当前仓库原生 `YOLODataset` 和 `YOLOMultiModalDataset` 主要围绕单图或图像+文本的默认训练入口设计，不能直接按阶段1索引稳定读取：

- 当前帧 RGB
- 当前帧 IR
- 上一帧 RGB
- canonical `labels/obb`
- `index/train_temporal.json` / `index/val_temporal.json`

如果把这些逻辑硬塞进默认 dataset builder，会污染 baseline OBB 路径。阶段2因此采用独立 dataset 文件和显式 builder。

## 3. 新 dataset 依赖的阶段1产物

`RGBIRTemporalOBBDataset` 以阶段1产物为唯一输入来源，优先使用：

- `images/rgb/{train,val}`
- `images/ir/{train,val}`
- `labels/obb/{train,val}`
- `index/train_pairs.json`
- `index/val_pairs.json`
- `index/train_temporal.json`
- `index/val_temporal.json`
- `data/uav_rgb_obb.yaml`

其中：

- `labels/obb` 是 canonical OBB 标签来源。
- `labels/rgb` 仍然保留给当前 baseline 兼容路径使用。
- 阶段2 dataset 不再依赖 `labels/rgb`，而是直接读取 canonical `labels/obb`。

## 4. 样本字段结构

单个样本当前返回的核心字段如下：

- `img`: 当前帧 RGB，`torch.uint8`，`C x H x W`
- `img_ir`: 当前帧 IR，`torch.uint8`，`C x H x W`
- `img_prev`: 上一帧 RGB，`torch.uint8`，`C x H x W`
- `cls`: `N x 1`
- `bboxes`: `N x 5`，归一化 `xywhr`
- `segments`: `N x 4 x 2`，归一化 polygon 四点
- `batch_idx`: 每样本局部 target 索引，供 collate 统一偏移
- `temporal_valid`: 当前样本是否拥有真实上一帧
- `im_file`, `im_file_ir`, `im_file_prev`, `label_file`
- `sample_id`, `frame_id`, `seq_id`
- `ori_shape`, `ori_shape_ir`, `ori_shape_prev`
- `resized_shape`, `ratio_pad`, `ratio_pad_ir`, `ratio_pad_prev`

## 5. temporal 缺失帧策略

当前阶段先支持 one-step temporal。

当 `train_temporal.json` 或 `val_temporal.json` 中某个样本没有上一帧时，默认策略是：

- `img_prev` 回退为当前帧 RGB
- `temporal_valid=False`

这样做的目的不是伪造时序信息，而是保证 batch 结构稳定，同时把边界帧状态显式暴露给后续阶段模型。

## 6. 同步变换策略

阶段2只保留保守、可解释的同步几何：

- letterbox 到统一 `imgsz`
- 可选水平翻转
- 可选垂直翻转

当前帧 RGB 与当前帧 IR 共享完全相同的 letterbox 参数，以保证双模态当前帧严格对齐。

上一帧 RGB 也会被变换到相同输出尺寸，并共享同一组 flip 决策。这样可以保证 temporal 输入张量形状稳定；当前阶段不做更重的时序增强，也不做颜色类专属增强。

标签几何通过 `Instances` 同步更新，最后输出：

- polygon `segments`
- 与当前仓库 OBB 习惯一致的归一化 `xywhr` `bboxes`

## 7. collate 组织方式

`RGBIRTemporalOBBDataset.collate_fn` 会：

- `stack`:
  - `img`
  - `img_ir`
  - `img_prev`
  - `temporal_valid`
- `cat`:
  - `cls`
  - `bboxes`
  - `segments`
- 重新生成全 batch 的 `batch_idx`
- 保留路径和 frame metadata 为 Python 列表

这保证后续阶段模型即使当前还不用 `img_ir` 或 `img_prev`，也能稳定拿到结构正确的 batch。

## 8. 如何显式启用该 dataset

当前阶段不改默认训练主线，只提供显式 opt-in 入口。

### 直接实例化

```python
from ultralytics.data.rgbir_temporal_obb_dataset import RGBIRTemporalOBBDataset

dataset = RGBIRTemporalOBBDataset(
    data=r"D:\project\ultralytics-main\runs\stage1_preprocess_smoke\prepared_sample\data\uav_rgb_obb.yaml",
    mode="train",
    imgsz=640,
    augment=False,
)
```

### 通过 opt-in builder

```python
from ultralytics.data.build import build_rgbir_temporal_obb_dataset

dataset = build_rgbir_temporal_obb_dataset(
    data=r"D:\project\ultralytics-main\runs\stage1_preprocess_smoke\prepared_sample\data\uav_rgb_obb.yaml",
    mode="train",
    imgsz=640,
)
```

### inspect 工具

```bash
python tools/inspect_rgbir_temporal_dataset.py ^
  --data D:\project\ultralytics-main\runs\stage1_preprocess_smoke\prepared_sample\data\uav_rgb_obb.yaml ^
  --mode train ^
  --use-builder
```

## 9. 为什么本阶段仍然不改训练主线

阶段2的目标是把阶段1标准化资产接成一个正确、稳定、可检查的 dataset 层。如果现在把它提前接进 trainer 或 model，会同时引入：

- 新 dataset 行为
- 新 batch 结构
- 新 forward 消费约定

这会显著放大 baseline 回归风险。因此本阶段坚持：

- 默认 `build_yolo_dataset()` 不变
- 默认 `yolo obb train/val/predict/track` 行为不变
- 新能力只在显式调用 `RGBIRTemporalOBBDataset` 或 `build_rgbir_temporal_obb_dataset()` 时启用

## 10. 后续阶段如何使用本阶段产物

阶段3及后续阶段可以直接复用本阶段 dataset 输出：

- 阶段3训练期 RGB-IR 协同模型：
  - 消费 `img` + `img_ir`
- 阶段5轻量时序：
  - 消费 `img` + `img_prev` + `temporal_valid`
- 阶段4小目标优化：
  - 继续复用 canonical `labels/obb`
- 阶段6多目标追踪前置时序输入：
  - 复用 `frame_id`、`seq_id`、`img_prev`

如果后续需要多帧历史，优先在当前 temporal index 和 dataset 样本结构上扩展，不建议回退到猜目录或猜相邻文件名的方式。
