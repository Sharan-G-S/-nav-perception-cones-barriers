# Edge optimisation: before and after

Everything on this page comes from one command,

```
uv run navdet optimize --weights runs/train/yolov8n/weights/best.pt --finetune-epochs 2 --runs 40 --backbones yolov8n.pt yolov8s.pt yolov8m.pt
```

run on an Apple M1 (8 cores, 8 GB) with PyTorch 2.15.0, Ultralytics 8.4.215 and ONNX Runtime
1.32.0. "Latency" and "FPS" are end to end for one 640 px frame: letterbox preprocessing,
inference, and non-maximum suppression, averaged over 40 real validation frames after 10
warm-up frames. mAP is re-measured on the 180-image validation split for every variant, so each
speed number has its accuracy next to it. The raw rows are in
[../results/optimization/optimization.json](../results/optimization/optimization.json).

| Variant | Device | Precision | Size (MB) | mAP50 | mAP50-95 | Latency (ms) | FPS |
|---|---|---|---|---|---|---|---|
| pytorch fp32 | cpu | fp32 | 6.2 | 0.596 | 0.392 | 32.5 | 30.7 |
| onnx fp32 | cpu | fp32 | 12.2 | 0.592 | 0.389 | 47.2 | 21.2 |
| onnx int8 (static, calibrated) | cpu | int8 | 3.4 | 0.575 | 0.364 | 35.2 | 28.4 |
| pytorch fp32 | mps | fp32 | 6.2 | 0.596 | 0.392 | 12.5 | 79.9 |
| pytorch fp16 | mps | fp16 | 6.2 | 0.596 | 0.392 | 14.4 | 69.5 |
| pruned 30% (L1, no finetune) | cpu | fp32 | 6.2 | 0.571 | 0.360 | 31.9 | 31.4 |
| pruned 30% + 2 ep sparse finetune | cpu | fp32 | 6.2 | 0.547 | 0.352 | 30.6 | 32.7 |
| backbone yolov8n (COCO weights, speed only) | cpu | fp32 | 6.2 | n/a | n/a | 37.1 | 26.9 |
| backbone yolov8n (COCO weights, speed only) | mps | fp32 | 6.2 | n/a | n/a | 20.5 | 48.7 |
| backbone yolov8s (COCO weights, speed only) | cpu | fp32 | 21.5 | n/a | n/a | 64.9 | 15.4 |
| backbone yolov8s (COCO weights, speed only) | mps | fp32 | 21.5 | n/a | n/a | 25.0 | 40.0 |
| backbone yolov8m (COCO weights, speed only) | cpu | fp32 | 49.7 | n/a | n/a | 111.1 | 9.0 |
| backbone yolov8m (COCO weights, speed only) | mps | fp32 | 49.7 | n/a | n/a | 47.8 | 20.9 |
| backbone yolo11n (COCO weights, speed only) | cpu | fp32 | 5.3 | n/a | n/a | 35.8 | 28.0 |
| backbone yolo11n (COCO weights, speed only) | mps | fp32 | 5.3 | n/a | n/a | 25.7 | 38.9 |
| backbone yolo26n (COCO weights, speed only) | cpu | fp32 | 5.3 | n/a | n/a | 37.9 | 26.4 |
| backbone yolo26n (COCO weights, speed only) | mps | fp32 | 5.3 | n/a | n/a | 18.3 | 54.7 |

## What the rows say

**PyTorch FP32 on CPU is the baseline**: 30.7 FPS (32.5 ms a frame) at mAP50 0.596. The
checkpoint is 6.2 MB because Ultralytics stores weights in FP16.

**ONNX FP32 on CPU is slower than PyTorch here** (21.2 FPS). PyTorch on Apple silicon runs its
convolutions through Accelerate/oneDNN kernels that are tuned for the M1, while ONNX Runtime's
default CPU execution provider is not as tuned for arm64. The 12.2 MB file is the same network
stored in FP32. On an x86 CPU or on the Jetson's ARM cores with the CUDA or TensorRT execution
provider the picture usually flips; the harness is the same, only the numbers change. mAP drops
by 0.004, which is tracing noise (fixed input shape, fused batch norm).

**ONNX INT8 is the smallest and the fastest ONNX variant**: 3.4 MB (3.6 times smaller than the
FP32 ONNX, 1.8 times smaller than the FP16 checkpoint), 28.4 FPS, a 34 percent latency cut
against ONNX FP32, for a cost of 0.021 mAP50 and 0.028 mAP50-95. Calibration used the 180
validation images that Ultralytics collected from the dataset yaml (it warns that 300 or more
is recommended); calibrating on the training split or on frames from the robot's own camera is
the first thing to try to win part of that accuracy back. On a Jetson, INT8 in TensorRT runs on
the DLA and tensor cores and is where the large gain is; on this CPU it mostly buys size.

**GPU (MPS) FP32 gives 79.9 FPS**, 2.6 times the CPU, with identical mAP. **FP16 on MPS is not
faster** (69.5 FPS): the M1's Metal path converts in and out of half precision and the model is
too small for the memory bandwidth saving to matter. On CUDA tensor cores FP16 is the normal
2x; the row is here because the measurement says what this hardware does rather than what the
datasheet promises.

**Pruning.** L1 unstructured pruning at 30 percent sparsity on every learned convolution (the
fixed DFL decoder convolution in the head is excluded; pruning it zeroes the box decoder and
sends mAP to zero, which the test suite now guards against) costs 0.025 mAP50 and 0.032
mAP50-95 with no fine-tuning at all, and the checkpoint compresses from 5.63 MB to 4.68 MB
with gzip because 30 percent of the weights are zeros. FPS is unchanged (31.4 vs 30.7, within
noise): dense kernels multiply the zeros, so unstructured sparsity buys storage and bandwidth,
not compute. A 2-epoch fine-tune that re-applies the sparsity mask after every optimiser step
(on the model and on its EMA copy, so the saved weights stay 30 percent sparse) did *not*
recover accuracy here (0.547 mAP50): two epochs at the automatic AdamW learning rate with the
full mosaic augmentation perturb the network more than they repair it. The recovery recipe that
works in the literature is longer (10 or more epochs) at a tenth of the learning rate with
augmentation off, which `navdet optimize --finetune-epochs 10` plus the lr flags of
`navdet train` can run. Structured (channel) pruning, which removes whole filters and so cuts
FLOPs and latency, is the follow-up that pays off on the Jetson.

**Backbone swap.** On identical frames, YOLOv8n / s / m run at 26.9 / 15.4 / 9.0 FPS on the
CPU and 48.7 / 40.0 / 20.9 FPS on the GPU, with 6.2 / 21.5 / 49.7 MB files; that is 3.5 and
9 times the compute of the n model for s and m. These rows use the COCO weights of each size
(speed only, 80-class NMS, which is also why the COCO n is slightly slower than the fine-tuned
3-class n at 30.7 FPS). Fine-tuning s or m with the same `navdet train --model yolov8s.pt`
command is the obvious accuracy lever if the robot has the budget: on a Jetson AGX Orin all
three sizes exceed real time, so the choice is set by how much of the GPU the rest of the
navigation stack needs. The newer nano models time at 28.0 FPS (YOLO11n) and 26.4 FPS
(YOLO26n) on the CPU and 38.9 / 54.7 FPS on the GPU; YOLO26n is NMS-free, which is why it is
the fastest on the GPU and the natural pick for a TensorRT engine on the Orin.

## What to run on the robot

1. `navdet optimize --device 0` on the Orin fills the CUDA rows and `yolo export format=engine
   quantize=8 data=...` adds TensorRT INT8, the configuration that should ship.
2. Calibrate INT8 on frames from the ZED or RealSense, not on the public datasets.
3. Keep the detector at 640 px and run it every second or third frame with optical flow in
   between (section 4.3 of the README) if the GPU budget is tight.
