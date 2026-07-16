from cmpnt.c1_img import imgStream
from cmpnt.c2_det import odStream
from cmpnt.c3_fr import frStream
from cmpnt.c4_lpr import lprStream
from cmpnt.c5_cap import capStream
from cmpnt.c6_kr import krStream
from cmpnt.c7_udp import udpStream
from cleanup import cleanup_resources
import os
import time
import torch
import multiprocessing as mp
from multiprocessing import Queue, Process, Event
import logging
import argparse
import json
import glob
from legacy_flops import summary

parser = argparse.ArgumentParser(description="Traffic Monitoring Pipeline")
parser.add_argument("--model_id", type=int, default=0, help="Model ID for object detection")
parser.add_argument('--algorithm', type=str, default=None, choices=["overload", 
                                                                    "slowtrack", 
                                                                    "phantom", 
                                                                    "teaspoon", 
                                                                    "clean"], help="algorithm not found")
parser.add_argument('--target_idx', type=int, nargs='+', default=None, help="List of numbers, unavailable for baseline")
parser.add_argument("--ps_path", type=str, default="./profile", help="Path to save profile data")
parser.add_argument("--eval_size", type=int, default=100, help="num of images to evaluate")
parser.add_argument("--profiling", action="store_true", help="use internal pytorch profiler")
parser.add_argument("--watch", type=float, default=1.0, help="use internal pytorch profiler")
parser.add_argument("--batch_size", type=int, default=1, help="batch size for each module")
parser.add_argument("--conf_filter", type=float, default=None, help="secondary confidence filter for OD output")
parser.add_argument("--buffer_limit", type=int, default=None, help="max queue size before tail-drop")
parser.add_argument("--input_defense", type=str, default="none",
                    choices=["none", "gaussian", "smoothing", "svm"], help="input-level defense at producer")
parser.add_argument("--svm_path", type=str, default="svm_defense.joblib", help="SVM defense joblib path")
parser.add_argument("--mix_mode", action="store_true", help="mix clean and attacked images")
parser.add_argument("--clean_path", type=str, default="../saved/clean_pool", help="clean pool dir for mix mode")
parser.add_argument("--attack_path", type=str, default="../saved/model_0/teaspoon_tgt_2",
                    help="attack pool dir for mix mode")
parser.add_argument("--mix_ratio", type=float, default=0.0, help="fraction of attacked images in the mix (0..1)")
parser.add_argument("--shuffle_seed", type=int, default=0, help="seed for the mix shuffle ordering")
parser.add_argument("--eval_index_start", type=int, default=50,
                    help="skip first N images of each pool to avoid SVM training contamination")
parser.add_argument("--clean_eval_index_start", type=int, default=None,
                    help="override eval_index_start for the clean pool (default: same as eval_index_start)")
parser.add_argument("--attack_eval_index_start", type=int, default=None,
                    help="override eval_index_start for the attack pool (default: same as eval_index_start)")
args = parser.parse_args()

if args.target_idx:
    target_indices = ('_'.join(map(str, args.target_idx)))
else:
    target_indices = "none"
    
base_dir = "../saved"
model_id = args.model_id
algorithm = args.algorithm
eval_size = args.eval_size
profiling = args.profiling
watch_interval = args.watch

if args.mix_mode:
    # mix mode: --ps_path is used directly (driver controls layout)
    ps_path = args.ps_path
    input_dir = None  # not used; producer reads clean_path + attack_path
elif algorithm == "clean":
    input_dir = "../saved/clean"
    ps_path = os.path.join(args.ps_path, "clean")
elif algorithm is None:
    input_dir = "./test_src"
    ps_path = "./test_profile"
else:
    ps_path = os.path.join(args.ps_path, f"model_{args.model_id}", f"{args.algorithm}_tgt_{target_indices}")
    input_dir = os.path.join(base_dir, f"model_{args.model_id}", f"{args.algorithm}_tgt_{target_indices}")
    
# print(f"Input directory: {input_dir}")
# print(f"Profile save path: {ps_path}")

os.makedirs(ps_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

            
def generate_summary(ps_path, start_time, end_time, eval_size):
    """Generate profiling summary from component JSON files."""
    components = ["odStream", "frStream", "lprStream", "capStream", "krStream", "udpStream"]
    component_data = {}
    for comp in components:
        json_path = os.path.join(ps_path, f"{comp}.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                component_data[comp] = json.load(f)

    img_json = os.path.join(ps_path, "imgStream.json")
    img_data = {}
    if os.path.exists(img_json):
        with open(img_json) as f:
            img_data = json.load(f)

    total_wall_time = end_time - start_time
    throughput = eval_size / total_wall_time if total_wall_time > 0 else 0

    # Compute per-image end-to-end latency
    # Collect all per_image logs from all components, find min start and max end per image_id
    image_latencies = {}
    for comp, data in component_data.items():
        per_image = data.get("per_image", {})
        for img_id_str, timing in per_image.items():
            img_id = str(img_id_str)
            if img_id not in image_latencies:
                image_latencies[img_id] = {"start": timing["start"], "end": timing["end"]}
            else:
                image_latencies[img_id]["start"] = min(image_latencies[img_id]["start"], timing["start"])
                image_latencies[img_id]["end"] = max(image_latencies[img_id]["end"], timing["end"])

    per_image_e2e = {}
    for img_id, times in image_latencies.items():
        per_image_e2e[img_id] = times["end"] - times["start"]

    latencies = list(per_image_e2e.values())
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0

    def _pct(values, q):
        if not values:
            return 0.0
        s = sorted(values)
        if len(s) == 1:
            return float(s[0])
        idx = (len(s) - 1) * q
        lo = int(idx)
        hi = min(lo + 1, len(s) - 1)
        frac = idx - lo
        return float(s[lo] * (1 - frac) + s[hi] * frac)

    p50 = _pct(latencies, 0.50)
    p95 = _pct(latencies, 0.95)
    p99 = _pct(latencies, 0.99)

    # FLOPs summary
    flops_components = ["odStream", "frStream", "lprStream", "capStream"]
    flops_summary = {}
    total_flops = 0
    for comp in flops_components:
        if comp in component_data:
            comp_flops = component_data[comp].get("total_flops", 0)
            flops_summary[comp] = {
                "flops_per_call": component_data[comp].get("flops_per_call", 0),
                "total_flops": comp_flops,
                "count": component_data[comp].get("count", 0)
            }
            total_flops += comp_flops

    # Workload counts and drops
    workload_summary = {}
    for comp in components:
        if comp in component_data:
            workload_summary[comp] = {
                "count": component_data[comp].get("count", 0),
                "drop_count": component_data[comp].get("drop_count", 0),
                "time": component_data[comp].get("time", 0)
            }

    total_drops = sum(workload_summary.get(c, {}).get("drop_count", 0) for c in components)
    total_drops += int(img_data.get("drop_count", 0))
    total_drops += int(img_data.get("input_drop_total", 0))

    summary = {
        "pipeline": {
            "total_wall_time": total_wall_time,
            "eval_size": eval_size,
            "throughput_img_per_sec": throughput
        },
        "per_image_latency": {
            "avg": avg_latency,
            "min": min_latency,
            "max": max_latency,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "per_image": per_image_e2e
        },
        "flops": {
            "components": flops_summary,
            "total_flops": total_flops
        },
        "workload": workload_summary,
        "input_defense": {
            "name": img_data.get("input_defense"),
            "input_drop_total": img_data.get("input_drop_total", 0),
            "input_drop_clean": img_data.get("input_drop_clean", 0),
            "input_drop_attack": img_data.get("input_drop_attack", 0),
        },
        "mix": {
            "mode": img_data.get("mix_mode", False),
            "ratio": img_data.get("mix_ratio"),
            "shuffle_seed": img_data.get("shuffle_seed"),
            "source_labels": img_data.get("source_labels", {}),
        },
        "drops_total": total_drops,
    }

    summary_path = os.path.join(ps_path, "SUMMARY.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    # Print summary table
    print("\n" + "=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)
    print(f"  Total wall time:    {total_wall_time:.2f}s")
    print(f"  Images processed:   {eval_size}")
    print(f"  Throughput:         {throughput:.4f} img/s")
    print(f"  Avg E2E latency:    {avg_latency:.4f}s")
    print(f"  Min E2E latency:    {min_latency:.4f}s")
    print(f"  Max E2E latency:    {max_latency:.4f}s")
    print()
    print(f"  {'Component':<15} {'Count':>8} {'Dropped':>8} {'FLOPs/call':>14} {'Total FLOPs':>14}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*14} {'-'*14}")
    for comp in components:
        count = workload_summary.get(comp, {}).get("count", 0)
        drops = workload_summary.get(comp, {}).get("drop_count", 0)
        fpc = flops_summary.get(comp, {}).get("flops_per_call", "-")
        tf = flops_summary.get(comp, {}).get("total_flops", "-")
        fpc_str = f"{fpc:>14.0f}" if isinstance(fpc, (int, float)) and fpc else f"{'n/a':>14}"
        tf_str = f"{tf:>14.0f}" if isinstance(tf, (int, float)) and tf else f"{'n/a':>14}"
        print(f"  {comp:<15} {count:>8.0f} {drops:>8.0f} {fpc_str} {tf_str}")
    print(f"\n  Total FLOPs:        {total_flops:.0f}")
    print("=" * 70)
    logger.info(f"Summary written to {summary_path}")


if __name__ == "__main__":
    date_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    title = "Running: Traffic Monitoring Pipeline"
    subtitle = f"Algorithm: {algorithm}, Model ID: {model_id}"
    subsubtitle = f"Target Indices: {target_indices}"
    
    print("\n\n" + "=" * 80)
    print("|" + " ".center(78) + "|")
    print("|" + title.center(78) + "|")
    print("|" + " ".center(78) + "|")
    print("|" + subtitle.center(78) + "|")
    print("|" + " ".center(78) + "|")
    print("|" + subsubtitle.center(78) + "|")
    print("|" + " ".center(78) + "|")
    print("|" + date_time.center(78) + "|")
    print("|" + " ".center(78) + "|")
    print("=" * 80 + "\n\n")
    
    start_time = time.perf_counter()
    
    cuda_device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {cuda_device_count}")
    cuda_idx = cuda_device_count - 1
    
    # torch.cuda.synchronize()

    mp.set_start_method("spawn")
    
    ms = 1000
    
    img2od_queue = Queue(maxsize=ms)
    od2fr_queue = Queue(maxsize=ms)
    od2lpr_queue = Queue(maxsize=ms)
    od2cap_queue = Queue(maxsize=ms)
    fr2kr_queue = Queue(maxsize=ms)
    lpr2kr_queue = Queue(maxsize=ms)
    cap2udp_queue = Queue(maxsize=ms)
    kr2udp_queue = Queue(maxsize=ms)
    
    # img_stream = imgStream(img2od_queue, "cuda:0")
    # od_stream = odStream(img2od_queue, od2fr_queue, od2lpr_queue, od2cap_queue, "cuda:1")
    # fr_stream = frStream(od2fr_queue, fr2kr_queue, "cuda:2")
    # lpr_stream = lprStream(od2lpr_queue, lpr2kr_queue, "cuda:3")
    # cap_stream = capStream(od2cap_queue, cap2udp_queue, "cuda:4")
    # kr_stream = krStream(fr2kr_queue, lpr2kr_queue, kr2udp_queue, "cuda:5")
    # udp_Stream = udpStream(cap2udp_queue, kr2udp_queue, "cuda:6")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_stream = imgStream(img2od_queue, device)
    od_stream = odStream(img2od_queue, od2fr_queue, od2lpr_queue, od2cap_queue, device)
    fr_stream = frStream(od2fr_queue, fr2kr_queue, device)
    lpr_stream = lprStream(od2lpr_queue, lpr2kr_queue, device)
    cap_stream = capStream(od2cap_queue, cap2udp_queue, device)
    kr_stream = krStream(fr2kr_queue, lpr2kr_queue, kr2udp_queue, device)
    udp_Stream = udpStream(cap2udp_queue, kr2udp_queue, device)
    
    input_defense_kwargs = {}
    if args.input_defense == "svm":
        input_defense_kwargs["model_path"] = args.svm_path
    clean_eval_start = args.clean_eval_index_start if args.clean_eval_index_start is not None else args.eval_index_start
    attack_eval_start = args.attack_eval_index_start if args.attack_eval_index_start is not None else args.eval_index_start
    img_stream.set_config(
        src_folder_path=input_dir, fps=30, profile_save_path=ps_path,
        eval_size=eval_size, buffer_limit=args.buffer_limit,
        input_defense_name=args.input_defense,
        input_defense_kwargs=input_defense_kwargs,
        mix_mode=args.mix_mode,
        clean_path=args.clean_path,
        attack_path=args.attack_path,
        mix_ratio=args.mix_ratio,
        shuffle_seed=args.shuffle_seed,
        clean_eval_index_start=clean_eval_start,
        attack_eval_index_start=attack_eval_start,
    )
    od_stream.set_config(model_id=0, profile_save_path=ps_path, batch_size=args.batch_size, conf_filter=args.conf_filter, buffer_limit=args.buffer_limit)
    fr_stream.set_config(profile_save_path=ps_path, batch_size=args.batch_size, buffer_limit=args.buffer_limit)
    lpr_stream.set_config(profile_save_path=ps_path, batch_size=args.batch_size, buffer_limit=args.buffer_limit)
    cap_stream.set_config(profile_save_path=ps_path, batch_size=args.batch_size, buffer_limit=args.buffer_limit)
    kr_stream.set_config(embedding_path="./face_embeddings", profile_save_path=ps_path, batch_size=args.batch_size, buffer_limit=args.buffer_limit)
    udp_Stream.set_config(profile_save_path=ps_path, batch_size=args.batch_size)

    
    processes = [img_stream, od_stream, fr_stream, lpr_stream, cap_stream, kr_stream, udp_Stream]
    queues = [img2od_queue, od2fr_queue, od2lpr_queue, od2cap_queue, fr2kr_queue, lpr2kr_queue, cap2udp_queue, kr2udp_queue]
    queue_names = ["img2od", "od2fr", "od2lpr", "od2cap", "fr2kr", "lpr2kr", "cap2udp", "kr2udp"]
    
    for sub_p in processes:
        sub_p.enable_profile(profiling)
        
    from queue_watch import QueueWatch
    queue_watcher = QueueWatch(queues, queue_names, processes, ps_path)
    queue_watcher.set_config(sleep_time=watch_interval)
    queue_watcher.start()
    
    for p in processes:
        p.start()
        time.sleep(0.1)  # Optional: small delay to ensure all processes start properly
    
    try:
        for p in processes:
            p.join()
            time.sleep(0.1)  # Optional: small delay to ensure all processes finish properly
        queue_watcher.join()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user in MAIN process")
    except Exception as e:
        logger.error(f"Error in MAIN process: {str(e)}")
    finally:
        logger.info("Reach FINALLY block in MAIN process")
        cleanup_resources(processes, queues)
        logger.info("Cleaned up resources in MAIN process")
        
    end_time = time.perf_counter()

    # Generate profiling summary
    generate_summary(ps_path, start_time, end_time, eval_size)

    time.sleep(5)
    time.sleep(5)
    torch.cuda.empty_cache()