Table R5: Energy Consumption (Joules) by Module and Method. ``Clean`` means the tested images are benign, unperturbed images; ``Overload``, ``Phantom`` and ``SlowTrack`` are baselines we compare with [1][2][3]. ``Ours ("person")`` means using our approach and targets the ``Object detection -> Face Recognition -> Knowledge Retrieval -> Data Uploading`` pathway as shown in the figure, similarily, ``Ours ("car")`` targets the the ``Object detection -> License Plate Recognition -> Knowledge Retrieval -> Data Uploading`` and ``Ours ("person", "car")`` targets both.

| Method | IMG | OD | FR | LPR | KR | CAP | UDP |
|---|---|---|---|---|---|---|---|
| clean | 3.16 | 5.25 | 2.70×10¹ | 5.26 | 3.59×10¹ | 4.96 | 3.69×10¹ |
| ours ("person") | 3.68 | 3.57×10³ | 3.55×10³ | 3.64×10³ | 3.55×10³ | 3.64×10³ | 3.55×10³ |
| ours ("car") | 1.71 | 1.72×10³ | 1.72×10³ | 1.72×10³ | 1.71×10³ | 1.72×10³ | 1.74×10³ |
| ours ("person", "car") | 3.25 | 3.57×10³ | 3.57×10³ | 3.55×10³ | 3.53×10³ | 3.52×10³ | 3.53×10³ |

[1] Chen et. al.. Overload: Latency Attacks on Object Detection for Edge Devices.

[2] Shapira et. al.. Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors. 

[3] Ma et. al.. SlowTrack: Increasing the Latency of Camera-based Perception in Autonomous Driving Using Adversarial Examples.
