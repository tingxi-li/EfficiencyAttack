Table R4: Time Span (seconds) by Module and Method. ``Clean`` means the tested images are benign, unperturbed images; ``Overload``, ``Phantom`` and ``SlowTrack`` are baselines we compare with [1][2][3]. ``Ours ("person")`` means using our approach and targets the ``Object detection -> Face Recognition -> Knowledge Retrieval -> Data Uploading`` pathway as shown in the figure, similarily, ``Ours ("car")`` targets the the ``Object detection -> License Plate Recognition -> Knowledge Retrieval -> Data Uploading`` and ``Ours ("person", "car")`` targets both.


| Method | IMG | OD | FR | LPR | KR | CAP | UDP |
|---|---|---|---|---|---|---|---|
| clean | 4.51×10¹ | 6.78×10¹ | 1.15×10² | 6.87×10¹ | 1.29×10² | 6.91×10¹ | 1.30×10² |
| ours ("car") | 2.66×10¹ | 2.99×10⁴ | 2.99×10⁴ | 3.02×10⁴ | 3.06×10⁴ | 2.99×10⁴ | 3.06×10⁴ |
| ours ("person") | 2.66×10¹ | 3.17×10⁴ | 3.20×10⁴ | 3.17×10⁴ | 3.23×10⁴ | 3.17×10⁴ | 3.23×10⁴ |
| ours ("person", "car") | 2.50×10¹ | 2.93×10⁴ | 2.97×10⁴ | 2.93×10⁴ | 3.00×10⁴ | 2.93×10⁴ | 3.00×10⁴ |


[1] Chen et. al.. Overload: Latency Attacks on Object Detection for Edge Devices.

[2] Shapira et. al.. Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors. 

[3] Ma et. al.. SlowTrack: Increasing the Latency of Camera-based Perception in Autonomous Driving Using Adversarial Examples.
