# TABLE C — Input defenses on 10 attacked images (held-out indices 90–99)

Single run per defense, eval_size=10, mix_ratio=1.0. All 10 images are attacked and unseen by the SVM.

| Defense | Wall(s) | Tput(img/s) | AvgE2E | P50    | P95    | P99    | LPR# | Drops | TotalFLOPs |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gauss   | 213.4   | 0.047       | 108.82 | 106.51 | 194.13 | 194.15 | 850  | 0     | 18.2T      |
| smooth  | 512.7   | 0.020       | 299.48 | 260.82 | 465.95 | 477.70 | 1812 | 0     | 6.6T       |
| svm     | 529.2   | 0.019       | 406.42 | 406.42 | 504.76 | 513.51 | 1895 | 8     | 25.4T      |
