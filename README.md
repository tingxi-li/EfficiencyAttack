### A realistic pipeline we built for evaluation

![image](icml-rebuttal-figure.png)

### TFLOPs of each component in the pipeline

| Method | OD (TFLOPs) | FR (TFLOPs) | LPR (TFLOPs) | CAP (TFLOPs) | Total (TFLOPs) | ×Clean |
|---|---|---|---|---|---|---|
| clean | 13.40 | 1.97 | 5.31 | 3.45 | 24.13 | 1.0× |
| overload | 13.40 | 137.54 | 687.24 | 0.82 | 839.00 | 34.8× |
| phantom | 13.40 | 37.10 | 9.63 | 2.18 | 62.30 | 2.6× |
| slowtrack | 13.40 | 107.33 | 687.24 | 1.82 | 809.78 | 33.6× |
| ours | 13.40 | 95.24 | 29.22 | 2.81 | 140.67 | 5.8× |
| ours (0) | 13.40 | 978.44 | 0.33 | 0.36 | 992.53 | 41.1× |
| ours (2) | 13.40 | 0.21 | 30,245.20 | 0.54 | 30,259.35 | **1,254.0×** |
| ours (0,2) | 13.40 | 898.20 | 2,589.60 | 0.36 | 3,501.57 | 145.1× |

