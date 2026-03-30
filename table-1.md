Table 1: Per-Module Computational Cost (Traffic Monitoring Pipeline). 

| Module | GMACs | GFLOPs | Total GFLOPs |
|---|---|---|---|
| RT-DETR-R50 (Object Detection) | ~67.0 | ~134.0 | ~134.0 |
| MTCNN (Face Detection) | ~3.79 | ~7.58 | ~10.42 |
| InceptionResnetV1 (Face Recognition) | ~1.42 | ~2.84 | |
| DeepLabV3-ResNet50 (LPR) | ~166.0 | ~332.0 | ~332.0 |
| GIT-base prefill (Image Captioning) | ~27.1 | ~54.2 | ~90.8 |
| GIT-base AR decode (×10 tokens) |  ~18.4 | ~36.8 | |
| Knowledge Retrieval (GPT2Tokenizer + SQLite3) | negligible | negligible | ≈0 |
| Data Upload (UDP) |  negligible | negligible | ≈0 |
