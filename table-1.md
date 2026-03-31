Table R1: Per-Module Computational Cost (Traffic Monitoring Pipeline). 

| Module | GFLOPs |
|---|---|
| RT-DETR-R50 (Object Detection) | ~134.0 | 
| MTCNN (Face Detection) | ~7.58 | 
| InceptionResnetV1 (Face Recognition) | ~2.84 | 
| DeepLabV3-ResNet50 (License Plate Recognition)| ~332.0 | 
| GIT-base prefill (Image Captioning) | ~83 |
| GIT-base AR decode per 10 tokens |  ~0.25 |
| GPT2Tokenizer + SQLite3 (Knowledge Retrieval) | negligible |
| UDP (Data Uploading) |  negligible |
