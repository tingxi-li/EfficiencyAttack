Table 3: Workload Counts by Module and Method. ``Clean`` means the tested images are benign, unperturbed images; ``Overload``, ``Phantom`` and ``SlowTrack`` are baselines we compare with [1][2][3]. ``Ours ("person")`` means using our approach and targets the ``Object detection -> Face Recognition -> Knowledge Retrieval -> Data Uploading`` pathway as shown in the figure, similarily, ``Ours ("car")`` targets the the ``Object detection -> License Plate Recognition -> Knowledge Retrieval -> Data Uploading`` and ``Ours ("person", "car")`` targets both.


| Method | IMG (×10²) | OD (×10²) | FR | LPR | KR | CAP | UDP |
|---|---|---|---|---|---|---|---|
| clean | 1.00 | 1.00 | 1.89×10² | 1.60×10¹ | 2.05×10² | 3.80×10¹ | 2.43×10² |
| ours ("car") | 1.00 | 1.00 | 2.00×10¹ | **9.11×10⁴** | 9.11×10⁴ | 6.00 | 9.11×10⁴ |
| ours ("person") | 1.00 | 1.00 | **9.39×10⁴** | 1.00 | 9.39×10⁴ | 4.00 | 9.39×10⁴ |
| ours ("person", "car") | 1.00 | 1.00 | 8.62×10⁴ | 7.80×10³ | **9.40×10⁴** | 4.00 | **9.40×10⁴** |


[1] Chen et. al.. Overload: Latency Attacks on Object Detection for Edge Devices.

[2] Shapira et. al.. Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors. 

[3] Ma et. al.. SlowTrack: Increasing the Latency of Camera-based Perception in Autonomous Driving Using Adversarial Examples.
