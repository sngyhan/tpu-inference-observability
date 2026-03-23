'''
Reads run.log to parse MoE expert imbalance metrics for both Current Steps and Final Accumulated.
Usage: python analyze_imbalance.py --log_file /path/to/run.log
'''

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def parse_log_file(file_path):
    """
    Parses the log file to extract both Current Step data and Final Accumulated data.
    """
    current_steps_data = {}  # {step_idx: {layer_idx: {category: group_sizes_array}}}
    final_accumulated_data = {}  # {layer_idx: {category: group_sizes_array}}
    
    step_count = 0
    curr_layer = None
    curr_cat = None
    target_type = None  # 'current' or 'accumulated'
    
    reading_array = False
    array_str = ""

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            clean_line = re.sub(r'^.*?(?:EngineCore\s+pid=\d+\)?\s*)', '', line.strip())
            if not clean_line and line.strip():
                clean_line = line.strip()

            # Detect step boundaries
            if "MoE Token Imbalance Profile:" in clean_line and "Accumulated" not in clean_line:
                step_count += 1
                current_steps_data[step_count] = {}
                target_type = 'current'
            elif "Final Accumulated MoE Token Imbalance Profile:" in clean_line:
                target_type = 'final'

            # Parse Context
            elif clean_line.startswith("Layer:"):
                layer_str = clean_line.split("Layer:")[1]
                nums = re.findall(r'\d+', layer_str)
                curr_layer = int(nums[0]) if nums else layer_str.strip()
                
                if target_type != 'final':
                    if curr_layer not in current_steps_data[step_count]:
                        current_steps_data[step_count][curr_layer] = {}
                else:
                    if curr_layer not in final_accumulated_data:
                        final_accumulated_data[curr_layer] = {}
                        
            elif match := re.search(r'\[(PREFILL|DECODE|ALL_VALID|ALL_INCLUDING_PADDING)\]', clean_line, re.IGNORECASE):
                curr_cat = match.group(1).upper()
                
            elif "--- Current Step ---" in clean_line:
                target_type = 'current'
            elif "--- Accumulated ---" in clean_line:
                # If we are inside the 'Final Accumulated' block, keep it as 'final'
                if target_type != 'final':
                    target_type = 'current_accumulated' # Ignored for this specific parsing logic unless requested

            # Parse Group Sizes Array
            elif clean_line.startswith("Group Sizes:"):
                reading_array = True
                array_str = clean_line.split("Group Sizes:")[1].strip()
                if "]" in array_str:
                    reading_array = False
                    _process_array(array_str, target_type, step_count, curr_layer, curr_cat, current_steps_data, final_accumulated_data)
                    array_str = ""
                    
            elif reading_array:
                array_str += " " + clean_line
                if "]" in array_str:
                    reading_array = False
                    _process_array(array_str, target_type, step_count, curr_layer, curr_cat, current_steps_data, final_accumulated_data)
                    array_str = ""

    return current_steps_data, final_accumulated_data, step_count

def _process_array(array_str, target_type, step_count, curr_layer, curr_cat, current_steps_data, final_accumulated_data):
    try:
        start_idx = array_str.find('[')
        end_idx = array_str.find(']')
        if start_idx != -1 and end_idx != -1:
            clean_str = array_str[start_idx:end_idx+1]
            nums = re.findall(r'\d+', clean_str)
            group_sizes = np.array([int(n) for n in nums])
            
            if target_type == 'current':
                current_steps_data[step_count][curr_layer][curr_cat] = group_sizes
            elif target_type == 'final':
                final_accumulated_data[curr_layer][curr_cat] = group_sizes
    except Exception as e:
        print(f"Warning: Failed to parse array for Layer {curr_layer} [{curr_cat}]: {e}")

def plot_heatmaps(data_dict, ep_lines, prefix, out_dir):
    """(1) & (3) Plots a heatmap of Group Sizes across all layers for each category."""
    if not data_dict: return
    categories = list(list(data_dict.values())[0].keys())

    for cat in categories:
        layers = sorted(data_dict.keys())
        matrix = np.array([data_dict[l].get(cat, np.zeros(0)) for l in layers])
        if matrix.size == 0 or matrix.ndim != 2: continue

        plt.figure(figsize=(14, 8))
        ax = sns.heatmap(matrix, cmap="YlOrRd")
        ax.set_yticks(np.arange(len(layers)) + 0.5)
        ax.set_yticklabels(layers, rotation=0)
        
        plt.title(f"{prefix} - Expert Load Heatmap [{cat}]")
        plt.xlabel("Expert Index")
        plt.ylabel("Layer")
        plt.tight_layout()

        # ep_lines에 0이 포함되어 있으면 선이 없는 버전 먼저 저장
        if 0 in ep_lines:
            plt.savefig(os.path.join(out_dir, f"{prefix}_heatmap_{cat.lower()}_no_lines.png"), dpi=300)

        # Draw EP dotted lines
        num_experts = matrix.shape[1]
        for ep in ep_lines:
            if ep == 0: continue
            for v_line in range(ep, num_experts, ep):
                ax.axvline(v_line, color='black', linestyle='--', alpha=0.5)

        plt.savefig(os.path.join(out_dir, f"{prefix}_heatmap_{cat.lower()}.png"), dpi=300)
        plt.close()

def plot_histogram(data_dict, target_layer, ep_lines, prefix, out_dir):
    """(2) & (3) Plots a histogram for a specific layer."""
    if target_layer not in data_dict:
        print(f"    Layer {target_layer} not found for histogram.")
        return

    cat_data = data_dict[target_layer]
    for cat, group_sizes in cat_data.items():
        plt.figure(figsize=(14, 5))
        experts = np.arange(len(group_sizes))
        plt.bar(experts, group_sizes, color='skyblue', edgecolor='black')

        plt.title(f"{prefix} - Histogram Layer {target_layer} [{cat}]")
        plt.xlabel("Expert Index")
        plt.ylabel("Token Count")
        plt.tight_layout()

        # ep_lines에 0이 포함되어 있으면 선이 없는 버전 먼저 저장
        if 0 in ep_lines:
            plt.savefig(os.path.join(out_dir, f"{prefix}_histogram_L{target_layer}_{cat.lower()}_no_lines.png"), dpi=300)

        # Draw EP dotted lines (offset by -0.5 to place between bars)
        for ep in ep_lines:
            if ep == 0: continue
            for v_line in range(ep, len(group_sizes), ep):
                plt.axvline(v_line - 0.5, color='red', linestyle='--', alpha=0.8)

        plt.savefig(os.path.join(out_dir, f"{prefix}_histogram_L{target_layer}_{cat.lower()}.png"), dpi=300)
        plt.close()

def simulate_sharding(group_sizes, num_shards):
    n_experts = len(group_sizes)
    if num_shards > n_experts:
        return None, None

    experts_per_shard = n_experts // num_shards

    def get_intra_shard_metrics(shards):
        rhos = []
        cvs = []
        for chunk in shards:
            if len(chunk) == 0:
                continue
            mean_val = np.mean(chunk)
            if mean_val > 0:
                rhos.append(np.max(chunk) / mean_val)
                cvs.append(np.std(chunk) / mean_val)
            else:
                rhos.append(1.0)
                cvs.append(0.0)
        # 각 샤드에서 구한 max/mean(rho)과 cv의 평균을 반환
        return np.mean(rhos), np.mean(cvs)

    # [1] Baseline (Contiguous): 각 샤드별로 16개씩 자른 배열을 리스트로 생성
    baseline_shards = [
        group_sizes[i*experts_per_shard : (i+1)*experts_per_shard]
        for i in range(num_shards)
    ]

    # [2] Greedy Bin-Packing (LPT): 각 샤드별로 할당된 값들을 리스트로 저장
    lpt_shards = [[] for _ in range(num_shards)]
    lpt_loads = np.zeros(num_shards) # 샤드별 총합(어느 샤드에 넣을지 결정하기 위한 용도)
    sorted_sizes = np.sort(group_sizes)[::-1]
    for size in sorted_sizes:
        idx = int(np.argmin(lpt_loads)) # 현재 총합이 가장 작은 샤드의 인덱스
        lpt_shards[idx].append(size)    # 해당 샤드에 값 추가
        lpt_loads[idx] += size

    return get_intra_shard_metrics(baseline_shards), get_intra_shard_metrics(lpt_shards)

def plot_sharding_simulation(data_dict, ep_list, prefix, out_dir, plot_lpt=False):
    """(4) Calculates CV and Max/Avg ratio simulating different EP degrees."""
    if not data_dict: return
    categories = list(list(data_dict.values())[0].keys())

    for cat in categories:
        avg_rho_b, avg_rho_l, avg_cv_b, avg_cv_l, valid_eps = [], [], [], [], []
        
        for ep in ep_list:
            layer_rhos_b, layer_rhos_l, layer_cvs_b, layer_cvs_l = [], [], [], []
            valid_ep = True
            for layer, cat_data in data_dict.items():
                if cat not in cat_data: continue
                res = simulate_sharding(cat_data[cat], ep)
                if res[0] is None:
                    valid_ep = False; break
                
                (rho_b, cv_b), (rho_l, cv_l) = res
                layer_rhos_b.append(rho_b)
                layer_cvs_b.append(cv_b)
                layer_rhos_l.append(rho_l)
                layer_cvs_l.append(cv_l)
                
            if valid_ep and layer_rhos_b:
                valid_eps.append(ep)
                avg_rho_b.append(np.mean(layer_rhos_b))
                avg_rho_l.append(np.mean(layer_rhos_l))
                avg_cv_b.append(np.mean(layer_cvs_b))
                avg_cv_l.append(np.mean(layer_cvs_l))

        if not valid_eps: continue
        
        if prefix == "Final_Accumulated":
            print(f"  [{cat}] Sharding Metrics (Avg across layers):")
            for i, ep in enumerate(valid_eps):
                lpt_str = f" | LPT Max/Mean: {avg_rho_l[i]:.4f}, LPT CV: {avg_cv_l[i]:.4f}" if plot_lpt else ""
                print(f"    EP={ep:<2d} | Base Max/Mean: {avg_rho_b[i]:.4f}, Base CV: {avg_cv_b[i]:.4f}{lpt_str}")

        # Subplot for both Imbalance Ratio and CV
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].plot(valid_eps, avg_rho_b, marker='o', label='Baseline (Contiguous)')
        for x, y in zip(valid_eps, avg_rho_b):
            axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=9)
            
        if plot_lpt:
            axes[0].plot(valid_eps, avg_rho_l, marker='x', label='Greedy Bin-Packing (LPT)')
            for x, y in zip(valid_eps, avg_rho_l):
                axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=9)
                
        axes[0].set_title(f"Avg Imbalance Ratio vs EP Degree")
        axes[0].set_xlabel("EP Degree")
        axes[0].set_ylabel("Imbalance Ratio (Max/Avg)")
        axes[0].set_xticks(valid_eps)
        axes[0].grid(True)
        axes[0].legend()
        
        axes[1].plot(valid_eps, avg_cv_b, marker='o', label='Baseline (Contiguous)')
        for x, y in zip(valid_eps, avg_cv_b):
            axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=9)
            
        if plot_lpt:
            axes[1].plot(valid_eps, avg_cv_l, marker='x', label='Greedy Bin-Packing (LPT)')
            for x, y in zip(valid_eps, avg_cv_l):
                axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=9)
                
        axes[1].set_title(f"Avg CV vs EP Degree")
        axes[1].set_xlabel("EP Degree")
        axes[1].set_ylabel("Coefficient of Variation (CV)")
        axes[1].set_xticks(valid_eps)
        axes[1].grid(True)
        axes[1].legend()

        plt.suptitle(f"{prefix} Sharding Simulation [{cat}]", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_sharding_sim_{cat.lower()}.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE Expert Imbalance Log Analyzer")
    parser.add_argument("--log_file", type=str, default="run.log", help="Path to the log file")
    parser.add_argument("--layer_idx", type=int, default=16, help="Target layer for the Histogram")
    parser.add_argument("--ep_lines", type=int, nargs='+', default=[0, 16], help="EP degrees to draw boundary lines on plots")
    parser.add_argument("--ep_sim", type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64], help="EP degrees for sharding simulation CV")
    parser.add_argument("--steps", type=int, nargs='+', default=[1, 3, 500, 1500], help="Target Current Steps to plot")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save the generated plots")
    parser.add_argument("--plot_lpt", action="store_true", help="Plot Greedy Bin-Packing (LPT) simulation alongside baseline (default: False)")

    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    print(f"Analyzing log file: {args.log_file}")

    out_dir = args.out_dir
    if not out_dir:
        log_dir = os.path.dirname(os.path.abspath(args.log_file))
        out_dir = os.path.join(log_dir, "analyze_results")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Results will be saved to: {out_dir}")
    
    current_data, final_data, total_steps = parse_log_file(args.log_file)
    print(f"Parsed {total_steps} steps from the log file.")

    # [1] Process Final Accumulated Data
    if final_data:
        print(f"\nProcessing Final Accumulated Data...")
        plot_heatmaps(final_data, args.ep_lines, "Final_Accumulated", out_dir)
        plot_histogram(final_data, args.layer_idx, args.ep_lines, "Final_Accumulated", out_dir)
        plot_sharding_simulation(final_data, args.ep_sim, "Final_Accumulated", out_dir, args.plot_lpt)
    else:
        print("No Final Accumulated data found in log.")

    # [2] Process Specific Steps
    for step in args.steps:
        if step in current_data:
            print(f"\nProcessing Data for Step {step}...")
            prefix = f"Step_{step}"
            plot_heatmaps(current_data[step], args.ep_lines, prefix, out_dir)
            plot_histogram(current_data[step], args.layer_idx, args.ep_lines, prefix, out_dir)
            plot_sharding_simulation(current_data[step], args.ep_sim, prefix, out_dir, args.plot_lpt)
        else:
            print(f"\nStep {step} not found in log (Max Step: {total_steps}).")
