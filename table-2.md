
Table R2: TFLOPs Breakdown by Module and Method. ``Clean`` means the tested images are benign, unperturbed images; ``Overload``, ``Phantom`` and ``SlowTrack`` are baselines we compare with [1][2][3]. ``Ours ("person")`` means using our approach and targets the ``Object detection -> Face Recognition -> Knowledge Retrieval -> Data Uploading`` pathway as shown in the figure, similarily, ``Ours ("car")`` targets the the ``Object detection -> License Plate Recognition -> Knowledge Retrieval -> Data Uploading`` and ``Ours ("person", "car")`` targets both.


| Method | OD (TFLOPs) | FR (TFLOPs) | LPR (TFLOPs) | CAP (TFLOPs) | Total (TFLOPs) | ×Clean |
|---|---|---|---|---|---|---|
| clean | 13.40 | 1.97 | 5.31 | 3.45 | 24.13 | 1.0× |
| overload | 13.40 | 137.54 | 687.24 | 0.82 | 839.00 | 34.8× |
| phantom | 13.40 | 37.10 | 9.63 | 2.18 | 62.30 | 2.6× |
| slowtrack | 13.40 | 107.33 | 687.24 | 1.82 | 809.78 | 33.6× |
| ours ("person") | 13.40 | 978.44 | 0.33 | 0.36 | 992.53 | 41.1× |
| ours ("car") | 13.40 | 0.21 | 30,245.20 | 0.54 | 30,259.35 | **1,254.0×** |
| ours ("person", "car") | 13.40 | 898.20 | 2,589.60 | 0.36 | 3,501.57 | 145.1× |


[1] Chen et. al.. Overload: Latency Attacks on Object Detection for Edge Devices.

[2] Shapira et. al.. Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors. 

[3] Ma et. al.. SlowTrack: Increasing the Latency of Camera-based Perception in Autonomous Driving Using Adversarial Examples.
