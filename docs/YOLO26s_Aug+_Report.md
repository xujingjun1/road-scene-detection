# YOLO26s-Aug+ 增强训练报告

> **核心模型**: [runs3/detect/rod_yolo26s_aug](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/)
> 源模型: [yolo26s.pt](file:///d:/ROD-Dataset(1)/ROD-Dataset/yolo26s.pt) (CAEv2 Backbone) | 训练脚本: [train_augmented_v3.py](file:///d:/ROD-Dataset(1)/ROD-Dataset/train_augmented_v3.py)

---

## YOLO26s-Aug+ 核心优势速览

| 维度 | 优势 | 数据 |
|------|------|:----:|
| **鲁棒性** | 5 模型中退化最小 | Corruption 平均退化仅 **-0.4108**（第 1 名） |
| **模糊不敏感** | Blur k3 下几乎无退化 | mAP50-95 仅降 **-0.0155** |
| **噪声不敏感** | Noise σ=15 下几乎无退化 | mAP50-95 仅降 **-0.0296** |
| **尾类建模** | 5 个稀有类 3× 过采样 | 数据量 19186 → **23611** (+23%) |
| **零参数增量** | 不改网络，推理速度不变 | 参数数 = 原版 yolo26s |
| **EMA 推理** | best_ema.pt 更稳定 | decay=0.9999 逐 batch 平滑 |
| **深度收敛** | 196 轮 vs 其他模型 80 轮 | mAP50-95 最优 epoch=166 |
| **Downscale 调度** | 前期学小目标，后期关掉冲峰值 | 40%→0% 概率衰减 |

---

## 一、数据集信息

### 1.1 基础统计

| 项目 | 数量 |
|------|:-----:|
| 类别数 | **25** |
| 训练集（原始） | **19,186** 张 |
| 验证集 | **3,511** 张 |
| 测试集 | 独立测试集 |
| 图片尺寸 | 统一 640×640 |

### 1.2 25 类列表

| ID | 类别 | ID | 类别 | ID | 类别 | ID | 类别 | ID | 类别 |
|:--:|------|:--:|------|:--:|------|:--:|------|:--:|------|
| 0 | Bike | 5 | Traffic sign | 10 | Dog | 15 | Truck | 20 | Traffic Barrel |
| 1 | Building | 6 | Electrical Pole | 11 | Manhole | 16 | Bus | 21 | Plant Pot |
| 2 | Car | 7 | Road | 12 | Tree | 17 | Bench | 22 | Electrical Box |
| 3 | Person | 8 | Motorcycle | 13 | Guard rail | 18 | Traffic Cone | 23 | Chair |
| 4 | Stairs | 9 | Dustbin | 14 | Crosswalk | 19 | Fire hydrant | 24 | Bicycle Rack |

### 1.3 YOLO26s-Aug+ 独有的尾类过采样

自动检测 bbox 数量最少的 5 个尾部类别，3 倍复制其所在图像：

| Class ID | 类别 | 过采样倍率 | 原因 |
|:--------:|------|:---------:|------|
| 19 | Fire Hydrant | ×3 | 数据集中 bbox 最少 |
| 24 | Bicycle Rack | ×3 | 数据集中 bbox 最少 |
| 0 | Bike | ×3 | bbox 数量较少 |
| 22 | Electrical Box | ×3 | bbox 数量较少 |
| 20 | Traffic Barrel | ×3 | bbox 数量较少 |

**YOLO26s-Aug+ 实际训练数据量: 19,186 → 23,611 (+4,425 张，+23.1%)**

> 其他 4 个消融模型均使用原始 19,186 张，未做过采样。

---

## 二、模型架构对比

### 2.1 网络结构

| 模型 | 网络结构 | 额外参数 |
|------|:---:|:---:|
| CA + WIoU (runs1) | yolo26s + 3 个 CoordAtt 模块 | 有 |
| CA + EMA (runs1) | yolo26s + 3 个 CoordAtt 模块 | 有 |
| CA + EMA + WIoU (runs1) | yolo26s + 3 个 CoordAtt 模块 | 有 |
| YOLO26s-Aug (runs2) | 原版 yolo26s | 无 |
| **YOLO26s-Aug+** (runs3) | **原版 yolo26s** | **无** |

**YOLO26s-Aug+ 不改网络结构，所有提升来自训练策略，推理速度无任何损失。**

### 2.2 各模块持有矩阵

| 模块 | CA+WIoU | CA+EMA | CA+EMA+WIoU | YOLO26s-Aug | **YOLO26s-Aug+** |
|------|:---:|:---:|:---:|:---:|:---:|
| CoordAtt 网络注入 | ✓ | ✓ | ✓ | ✗ | ✗ |
| EMA (decay=0.9999) | ✗ | ✓ | ✓ | ✗ | **✓** |
| WIoU Loss | ✓ | ✗ | ✓ | ✗ | ✗ |
| DownscaleAugmentation | ✗ | ✗ | ✗ | ✗ | **✓** |
| Downscale 调度 | ✗ | ✗ | ✗ | ✗ | **✓** |
| Blur + Noise (30%) | ✗ | ✗ | ✗ | ✗ | **✓** |
| 尾类过采样 (×3) | ✗ | ✗ | ✗ | ✗ | **✓** |
| Mixup=0.15 | ✗ | ✗ | ✗ | ✗ | **✓** |
| RandAugment | ✗ | ✗ | ✗ | ✗ | **✓** |
| Erasing=0.4 | ✗ | ✗ | ✗ | ✗ | **✓** |
| Epochs | 80 | 80 | 80 | 80 | **196** |

---

## 三、YOLO26s-Aug+ 新增模块详解

### 3.1 EMA (指数移动平均)

- **实现文件**: [augmentations.py](file:///d:/ROD-Dataset(1)/ROD-Dataset/augmentations.py#L263-L318)
- **decay**: 0.9999（极高平滑系数）
- **注入方式**: `on_train_batch_end` 回调，每个 batch 后更新
- **原理**: `shadow = 0.9999 × shadow + 0.0001 × current`
- **输出**: 训练后保存 `best_ema.pt`（推理推荐使用）

### 3.2 DownscaleAugmentation (缩小增强)

- **实现文件**: [augmentations.py](file:///d:/ROD-Dataset(1)/ROD-Dataset/augmentations.py#L76-L261)
- **作用**: 随机缩小图片到 45%-80%，模拟小目标（ROD 数据集中大量小目标）
- **核心机制**:
  - 图片随机缩小 + 随机贴到黑色画布
  - **bbox 坐标自动映射**，含 zero-error 验证 (max_err < 0.00000001)
  - 过滤缩后宽/高 < 2px 的无效框
- **调度策略** (YOLO26s-Aug+ 独有):

  | 阶段 | Epoch | 作用概率 | 目的 |
  |------|:-----:|:-------:|------|
  | 全量学习期 | 0-80 | **40%** | 充分接触小目标 |
  | 线性退火期 | 80-120 | **40%→0%** | 逐步过渡到真实尺度 |
  | 精度冲刺期 | 120-200 | **0%** | 关闭人工缩小，专注真实分布 |

### 3.3 高斯模糊 + 高斯噪声

| 增强项 | 触发概率 | 参数 | 注入时机 |
|--------|:---:|------|------|
| 高斯模糊 | 30% (每张图独立) | kernel∈{3,5,7}, sigma 随机 | on_train_batch_begin |
| 高斯噪声 | 30% (每张图独立) | std=0.02 × random(0.5,1.5) | on_train_batch_begin |

> 模糊和噪声对每张图独立判断触发，可能同时施加、只施加一种、或都不施加。

---

## 四、训练配置

### 4.1 训练超参

| 参数 | YOLO26s-Aug+ | 其他 4 模型 |
|------|:---:|:---:|
| 源权重 | yolo26s.pt | yolo26s.pt |
| Epochs | **196** (实际运行) | 80 |
| Batch Size | 16 | 16 |
| Image Size | 640 | 640 |
| 训练数据量 | **23,611** (过采样后) | 19,186 (原始) |
| Workers | 2 | 2 |
| Optimizer | auto (AdamW) | auto (AdamW) |
| LR Schedule | **cos_lr** | cos_lr |
| lr0 / lrf | 0.01 / 0.01 | 0.01 / 0.01 |
| Warmup | 3 epochs | 3 epochs |
| AMP | True | True |
| Seed | **42** (deterministic) | — |
| Early Stopping | 30 | 30 |

### 4.2 数据增强配置

| 增强项 | YOLO26s-Aug+ | 标准 YOLO 默认 | 其他 4 模型 |
|--------|:---:|:---:|:---:|
| Mosaic 1.0 | ✓ | ✓ | ✓ |
| close_mosaic | **10** | 10 | — |
| Mixup | **0.15** | 0.0 | 0.0 |
| AutoAugment | **randaugment** | none | none |
| Erasing | **0.4** | 0.0 | 0.0 |
| HSV-H/S/V | 0.015/0.7/0.4 | 0.015/0.7/0.4 | 相同 |
| Degrees | 10° | 0° | 0° |
| Translate | 0.1 | 0.0 | 0.0 |
| Scale | 0.5 | 0.5 | 0.5 |
| Flip LR | 0.5 | 0.5 | 0.5 |
| **Downscale (在线回调)** | **40%→0% 调度** | 无 | 无 |
| **Blur (在线回调)** | **30%** | 无 | 无 |
| **Noise (在线回调)** | **30%** | 无 | 无 |

---

## 五、训练结果

### 5.1 训练指标

| 指标 | 最佳值 | 最佳 Epoch | 最终 (Epoch 196) |
|------|:------:|:--------:|:------:|
| **mAP50** | **0.9189** | 118 | 0.9127 |
| **mAP50-95** | **0.7631** | 166 | 0.7556 |
| Precision | 0.9169 | — | 0.9166 |
| Recall | 0.8773 | — | 0.8773 |
| Box Loss | — | — | 0.4750 |
| Cls Loss | — | — | 0.2193 |
| DFL Loss | — | — | 0.0097 |

### 5.2 模型权重文件

| 权重文件 | 用途 | 路径 |
|---------|------|------|
| **best_ema.pt** (推荐) | EMA 平滑推理，精度更高 | [weights/best_ema.pt](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/weights/best_ema.pt) |
| best.pt | 标准 best checkpoint | [weights/best.pt](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/weights/best.pt) |
| last.pt | 最终 epoch 权重 | [weights/last.pt](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/weights/last.pt) |

### 5.3 训练时间与收敛

| 指标 | 值 |
|------|:-----:|
| 总训练时间 | **~42,542 秒 (11.8 小时)** |
| 平均每 epoch | ~217 秒 |
| 总训练图片 | 23,611×196 = ~462 万次曝光 |
| 实际运行轮次 | 196 (200 计划, early stop 未触发) |

---

## 六、训练过程可视化

### 6.1 损失函数与指标变化

![训练曲线](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/results.png)

*Box/Cls/DFL Loss 收敛曲线 + mAP50/mAP50-95/Precision/Recall 变化。可见 mAP50-95 在 epoch 166 达到峰值 0.7631，后期平稳维持。*

### 6.2 YOLO 性能指标曲线

| F1 曲线 | P-R 曲线 | Precision 曲线 | Recall 曲线 |
|:---:|:---:|:---:|:---:|
| ![F1](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/BoxF1_curve.png) | ![PR](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/BoxPR_curve.png) | ![P](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/BoxP_curve.png) | ![R](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/BoxR_curve.png) |

### 6.3 混淆矩阵

| 标准混淆矩阵 | 归一化混淆矩阵 |
|:---:|:---:|
| ![CM](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/confusion_matrix.png) | ![CMN](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/confusion_matrix_normalized.png) |

### 6.4 类别分布

![Labels](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/labels.jpg)

*25 类标签分布（原始训练集统计）。尾部类别（Fire Hydrant、Bicycle Rack 等）严重稀缺，YOLO26s-Aug+ 独有的过采样针对性地缓解了这一问题。*

### 6.5 训练批次示例

| Batch 0 | Batch 1 | Batch 2 |
|:---:|:---:|:---:|
| ![t0](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/train_batch0.jpg) | ![t1](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/train_batch1.jpg) | ![t2](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/train_batch2.jpg) |

*Mosaic 4 合 1 + Mixup + RandAugment + Erasing 增强后的训练样本。*

### 6.6 验证集预测 vs 真实标签

| | Batch 0 | Batch 1 | Batch 2 |
|---|:---:|:---:|:---:|
| **预测** | ![vp0](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/val_batch0_pred.jpg) | ![vp1](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/val_batch1_pred.jpg) | ![vp2](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/val_batch2_pred.jpg) |
| **真值** | ![vl0](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/val_batch0_labels.jpg) | ![vl1](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/val_batch1_labels.jpg) | ![vl2](file:///d:/ROD-Dataset(1)/ROD-Dataset/runs3/detect/rod_yolo26s_aug/val_batch2_labels.jpg) |

---

## 七、鲁棒性评估 — YOLO26s-Aug+ 的核心优势

### 7.1 评估设置

对验证集 3,511 张图片施加 10 种 corruption，评估 5 个模型的鲁棒性：

| Corruption | 类型 | 参数 |
|-----------|:---:|------|
| Rot 90 / 180 / 270 | 旋转 | 90° / 180° / 270° |
| Scale 0.5 / 0.75 | 缩小 | 50% / 75% 缩放 |
| Flip H | 水平翻转 | 镜像 |
| Blur k3 / k7 | 高斯模糊 | kernel=3 / kernel=7 |
| Noise σ=15/35 | 高斯噪声 | std=15/255 / 35/255 |

### 7.2 YOLO26s-Aug+ 逐项鲁棒性

| Corruption | mAP50 | mAP50-95 | vs Clean 退化 | 评估 |
|-----------|:-----:|:--------:|:------------:|:----:|
| **Clean** | **0.9167** | **0.7625** | — | 基准 |
| **Blur k3** | 0.9010 | 0.7470 | **-0.0155** | 几乎无退化 |
| **Noise σ=15** | 0.8909 | 0.7329 | **-0.0296** | 几乎无退化 |
| Blur k7 | 0.8385 | 0.6890 | -0.0735 | 轻微退化 |
| Noise σ=35 | 0.8085 | 0.6358 | -0.1267 | 中度退化 |
| Scale 0.75 | 0.6124 | 0.1082 | -0.6543 | 大幅退化 |
| Flip H | 0.5122 | 0.3091 | -0.4534 | 大幅退化 |
| Rot 180 | 0.2956 | 0.1486 | -0.6139 | 严重退化 |
| Rot 270 | 0.1839 | 0.0737 | -0.6888 | 几乎失效 |
| Rot 90 | 0.1821 | 0.0730 | -0.6895 | 几乎失效 |
| Scale 0.5 | 0.0000 | 0.0000 | -0.7625 | 完全失效 |

> **关键优势**: Blur k3 下仅退化 0.0155，Noise σ=15 下仅退化 0.0296，证明了 Blur+Noise 在线增强的有效性。

### 7.3 5 模型鲁棒性排名

| 排名 | 模型 | Clean mAP50-95 | 10 种 Corruption 均值 | **退化幅度 ↓** |
|:----:|------|:------:|:------:|:------:|
| **1** | **YOLO26s-Aug+** | 0.7625 | 0.3517 | **-0.4108** (最优) |
| 2 | CA + EMA + WIoU | 0.7695 | 0.3342 | -0.4353 |
| 3 | CA + WIoU | 0.7661 | 0.3276 | -0.4385 |
| 4 | YOLO26s-Aug | 0.7606 | 0.3194 | -0.4412 |
| 5 | CA + EMA | 0.7724 | 0.3296 | -0.4427 (最差) |

**YOLO26s-Aug+ 在 Clean 精度并非最高的情况下，鲁棒性排名第 1，退化幅度比第 5 名少 7.2%。**

### 7.4 鲁棒性可视化

#### 柱状图 — 5 模型 × 11 Corruptions 全景

![鲁棒性柱状图](file:///d:/ROD-Dataset(1)/ROD-Dataset/ablation_analysis/robustness_bars.png)

#### 退化曲线 — mAP50-95 随 corruption 加剧的下降趋势

![退化曲线](file:///d:/ROD-Dataset(1)/ROD-Dataset/ablation_analysis/robustness_degradation.png)

> YOLO26s-Aug+（紫色）在 Blur/Noise 区段明显高于其他模型

#### 雷达图 — 按 corruption 类别聚合

![雷达图](file:///d:/ROD-Dataset(1)/ROD-Dataset/ablation_analysis/robustness_radar.png)

#### 模糊/噪声专项 — 最关键的鲁棒性维度

![模糊噪声对比](file:///d:/ROD-Dataset(1)/ROD-Dataset/ablation_analysis/robustness_blur_noise.png)

> YOLO26s-Aug+ 在 Clean → Blur k3 的过渡中几乎无下降，显著优于其他模型。

---

## 八、消融对比 — 5 模型全景

### 8.1 训练指标对比

| 模型 | Epochs | Best mAP50 | Best mAP50-95 | 训练数据量 | 网络参数 |
|------|:-----:|:----------:|:--------:|:------:|:---:|
| CA + WIoU | 80 | 0.9227 | 0.7662 | 19,186 | +CoordAtt |
| CA + EMA | 80 | 0.9256 | **0.7729** | 19,186 | +CoordAtt |
| CA + EMA + WIoU | 80 | **0.9264** | 0.7697 | 19,186 | +CoordAtt |
| YOLO26s-Aug | 80 | 0.9231 | 0.7609 | 19,186 | 原版 |
| **YOLO26s-Aug+** | **196** | 0.9189 | **0.7631** | **23,611** | **原版 (零额外)** |

### 8.2 消融对比训练曲线

![训练消融曲线](file:///d:/ROD-Dataset(1)/ROD-Dataset/ablation_analysis/training_curves.png)

> YOLO26s-Aug+（紫色）训练 196 轮，mAP50 在 epoch 118 触顶后因重度增强略有波动，但 mAP50-95 在 epoch 166 达到最优，展现了长期训练的收敛收益。

### 8.3 综合对比: Clean 精度 vs 鲁棒性

| 模型 | Clean mAP50-95 | 鲁棒性退化 | 网络修改 | 综合评估 |
|------|:------:|:------:|:---:|------|
| CA + EMA | **0.7729** | -0.4427 (差) | 有 | Clean 最优但鲁棒性最差 |
| CA + EMA + WIoU | 0.7697 | -0.4353 | 有 | 综合较好 |
| CA + WIoU | 0.7662 | -0.4385 | 有 | 中等 |
| **YOLO26s-Aug+** | **0.7631** | **-0.4108 (最佳)** | **无** | **鲁棒性+简洁性最优** |
| YOLO26s-Aug | 0.7609 | -0.4412 | 无 | Clean 最低 |

---

## 九、总结

### YOLO26s-Aug+ 的核心创新

1. **不改网络结构，零额外参数** — 所有提升来自训练策略，推理速度不变
2. **EMA 平滑推理** — decay=0.9999，best_ema.pt 提供稳定输出
3. **Downscale 三阶段调度** — 0-80 学习小目标 / 80-120 退火 / 120+ 关闭冲刺
4. **Blur + Noise 在线增强 (30%)** — 直接解释了鲁棒性评估中模糊/噪声几乎无退化的原因
5. **尾类 3× 过采样** — 5 个稀有类针对性缓解，数据量 +23%
6. **Mixup + RandAugment + Erasing** — 三重泛化增强
7. **196 轮深度训练** — 远超其他模型 80 轮，mAP50-95 在 epoch 166 才收敛

### 数据总结

| 指标 | 值 |
|------|:-----:|
| 原始训练集 | 19,186 张, 25 类 |
| 过采样后训练集 | **23,611 张 (+23%)** |
| 验证集 | 3,511 张 |
| Best mAP50 | **0.9189** (epoch 118) |
| Best mAP50-95 | **0.7631** (epoch 166) |
| 鲁棒性退化 (5 模型比较) | **-0.4108 (最优)** |
| Blur k3 退化 | **-0.0155 (几乎不受影响)** |
| Noise σ=15 退化 | **-0.0296 (几乎不受影响)** |
| 训练参数增量 | **0** |
| 推理速度影响 | **0** |

### 使用建议

- **推理务必使用 `best_ema.pt`**，精度高于标准 `best.pt`
- 适用于中等尺度目标的实际场景，尤其是存在轻微模糊/噪声的环境
- 对极端旋转 (90°/270°) 和大幅缩小 (50%) 场景建议在训练中额外加入对应增强
