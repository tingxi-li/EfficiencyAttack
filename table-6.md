**Table R6: Pipeline Performance Under Scheduling, Batching, Buffering, and Defense Configurations.** Test size: 10 images, target vulnerable dataflow: ObjectDetection → LicensePlateRecognition → KnowledgeRetrieval. conf=0.5 means detected objects with confidence below 0.5 will be rejected. buffer=100 means a queue upper bound of 100; items exceeding that limit will be dropped. batch=16 means the pipeline processes inputs in batches of 16, fetched at a time from the queue. #Drops denotes the number of detected bounding boxes dropped due to bounded buffering.

| # | Config | Wall Time | Throughput | Avg E2E Latency | P50 | P95 | P99 | LPR Workload | #Drops | Total FLOPs |
|:--|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1 | Clean | 17.3s | 0.578 img/s | 1.23s | 0.49s | 3.86s | 4.68s | 6 | 0 | 3.27T |
| 2 | Attacked | 1805s | 0.006 img/s | 503.4s | 0.49s | 3.85s | 4.65s | 9,315 | 0 | 215.3T |
| 3 | + conf=0.5 | 1084s | 0.009 img/s | 373.1s | 634.6s | 900.8s | 916.6s | 5,576 | 0 | 129.6T |
| 4 | + conf=0.5 + batch=16 | 550s | 0.018 img/s | 188.1s | 678.3s | 859.2s | 863.4s | 5,576 | 0 | 129.6T |
| 5 | + batch=16 + buffer=100 | 28.3s | 0.354 img/s | 8.0s | 391.1s | 478.6s | 484.4s | 304 | 9,011 | 8.96T |
| 6 | + conf=0.5 + buffer=100 | 34.3s | 0.291 img/s | 15.0s | 8.3s | 26.9s | 27.2s | 215 | 5,361 | 6.87T |
| 7 | + conf=0.5 + batch=16 + buffer=100 | 25.3s | 0.396 img/s | 7.7s | 32.9s | 35.9s | 36.8s | 199 | 5,377 | 6.50T |
| — | Clean (buffer=100) | 18.2s | 0.550 img/s | 1.25s | 12.1s | 22.1s | 22.7s | 6 | 0 | 3.27T |
