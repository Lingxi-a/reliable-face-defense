# v0.7 NullSwap 保底补实验

本补丁不会修改现有 `reliable_paper_main_v07`。它在独立目录中验证一个候选策略：

1. 每个 source 以已经保存的 NullSwap PNG 为锚点；
2. 仅使用训练侧 ArcFace、FaceNet，在 Clean、JPEG Q60、Blur 1.0、OSN-QW 下进行小幅鲁棒残差搜索；
3. VGGFace、SFace 不参与生成或选择；
4. 候选必须同时满足：
   - 八个“识别模型 × 传播环境”相似度均不比本图 NullSwap 锚点高；
   - 最坏相似度至少下降 `0.0005`；
   - 保存为 8-bit PNG 后，PSNR、SSIM、LPIPS 均不比本图 NullSwap 差；
5. 没有候选满足全部条件时，逐像素回退到原 NullSwap 图。

因此代码可以保证选择阶段和视觉指标不存在主动劣化，但不能预先保证 VGGFace、SFace 和三种换脸模型的外部结果必然更好。是否超过 NullSwap 只由新实验表判定。

## 安装

将压缩包放到 `/root/autodl-tmp`，然后执行：

```bash
cd /root/autodl-tmp
tar -xzf nullswap_floor_rerun_patch_v01.tar.gz -C reliable_paper_experiments
cd reliable_paper_experiments
chmod +x run_nullswap_floor_probe.sh run_nullswap_floor_full.sh
```

前置目录保持与原实验一致：

```text
/root/autodl-tmp/reliable_face_defense_module05_visualft
/root/autodl-tmp/reliable_paper_experiments
/root/autodl-tmp/reliable_paper_main_v07
```

原实验目录还必须包含已经完成的四评价器补充结果：

```text
/root/autodl-tmp/reliable_paper_main_v07/four_evaluator_supplement/per_pair_four_evaluators.jsonl
```

## 第一步：20个身份探测实验

```bash
cd /root/autodl-tmp/reliable_paper_experiments
nohup bash ./run_nullswap_floor_probe.sh \
  > nullswap_floor_probe_launcher.out 2>&1 &

echo $! | tee nullswap_floor_probe.pid
tail -f nullswap_floor_probe_launcher.out
```

完成文件：

```text
/root/autodl-tmp/reliable_paper_v07_nullswap_floor_probe/summary.json
```

重点查看：

- `selection.selected_refinement_count`：20张中实际找到严格支配候选的数量；
- `main_comparison.better/equal/worse`：主实验各单元格相对 NullSwap 的数量；
- `robustness_comparison.better/equal/worse`：鲁棒性各单元格相对 NullSwap 的数量；
- `positive_probe_signal=true`：两张表都没有变差单元格、都至少有一个更优单元格，且确实选中了改进候选；这是继续完整实验的保守条件；
- `strictly_better_everywhere` 只有全部单元格严格更低时才为 `true`。

如果 `selected_refinement_count=0`，完整实验只会复现 NullSwap，不应继续跑200张。若探测集上 `worse` 明显大于0，也应先停下分析，而不是修改统计口径。

## 第二步：完整200个身份

只有探测实验出现真实正向信号后再运行：

```bash
cd /root/autodl-tmp/reliable_paper_experiments
nohup bash ./run_nullswap_floor_full.sh \
  > nullswap_floor_full_launcher.out 2>&1 &

echo $! | tee nullswap_floor_full.pid
tail -f nullswap_floor_full_launcher.out
```

完整结果目录：

```text
/root/autodl-tmp/reliable_paper_v07_nullswap_floor_rerun
```

关键结果：

```text
summary.json
protections/summary.json
main/visual_table_vs_nullswap.csv
main/main_table_vs_nullswap.csv
main/cellwise_comparison.csv
robustness/robustness_table_vs_nullswap.csv
robustness/cellwise_comparison.csv
```

脚本采用 JSONL 逐项写入并支持原目录断点续跑。探测实验和完整实验必须使用不同输出目录，不能把20张的中间结果混入200张正式结果。
每次启动还会先执行冻结协议、依赖路径、传播算子、CUDA 与核心保底逻辑的自检；看到 `NULLSWAP_FLOOR_SELFCHECK=PASS` 后才会进入候选生成。

## 判定口径

- Acc@1、Acc@5：越低越好；
- PSNR、SSIM：越高越好；
- LPIPS：越低越好；
- `equal` 只表示持平，不能写成优于；
- 最终论文仍以200个身份的完整冻结协议为准，20个身份只用于判断该候选方案是否值得继续。

