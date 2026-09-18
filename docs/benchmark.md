# Benchmark method

The launch slide compares one fixed workload: 114 source videos with a combined duration of 11:52:17, represented as 1080p30 input.

| Pipeline | Completion time | Real-time factor |
|---|---:|---:|
| Prior MediaPipe pipeline | 2:57:00 | 4.02× |
| AdvanX capacity model | 0:05:35 | 127.5× |

The resulting end-to-end ratio is 31.7×.

The AdvanX number is a capacity calculation assembled from separately measured fixed-rate components: decoder lanes, TensorRT inference, CPU post-processing, and inter-node transfer. The calculation respects the 30→24 temporal reduction and the measured decoder ceiling. It represents the designed steady-state capacity of the two-node system; filesystem state, source codec mix, GOP structure, thermals, and load balance can change a concrete run.

For a reproducible publication, retain:

1. the input catalog and source metadata;
2. each lane's measured output rate;
3. LAN throughput and staging bytes;
4. the generated `dispatch_plan.json`;
5. `distributed_aggregate.json` and the validation report.

The latest observed local five-lane run on an added 33-video batch processed 365,978 target frames in 949.271 seconds, or 385.536 target fps end to end. This is a workload result and is kept separate from the normalized launch comparison.
