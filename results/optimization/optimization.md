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
