import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

# Delete or comment out this line
# from sklearn.utils.linear_assignment_ import linear_assignment


def read_jsonl(file_path):
    clusters = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                clusters.append(json.loads(line))
    return clusters


def compute_metrics(pred_clusters, true_clusters):
    # Prediction file format: {"incident_id": "cluster_0", "cases": [...]} 
    pred_sets = {
        cluster['incident_id']: set(cluster['cases'])
        for cluster in pred_clusters
    }

    # True file format: {"incident_id": 1, "ids": [...]} 
    true_sets = {
        str(cluster['incident_id']): set(cluster['ids'])  # Convert to string to maintain type consistency
        for cluster in true_clusters
    }

    # Extract all sample IDs and label mappings for standardized metric calculation
    all_samples = set().union(*pred_sets.values(), *true_sets.values())
    pred_labels = {sample: None for sample in all_samples}
    true_labels = {sample: None for sample in all_samples}

    for pred_id, samples in pred_sets.items():
        for sample in samples:
            pred_labels[sample] = pred_id

    for true_id, samples in true_sets.items():
        for sample in samples:
            true_labels[sample] = true_id

    # Ensure all samples have valid labels
    for sample in all_samples:
        if pred_labels[sample] is None:
            pred_labels[sample] = "unclustered"
        if true_labels[sample] is None:
            true_labels[sample] = "unclustered"

    # Convert to integer index arrays
    sample_list = list(all_samples)  # Add this line definition
    unique_pred = sorted(set(pred_labels.values()))
    unique_true = sorted(set(true_labels.values()))
    pred_id_map = {v: i for i, v in enumerate(unique_pred)}
    true_id_map = {v: i for i, v in enumerate(unique_true)}

    y_pred = np.array([pred_id_map[pred_labels[s]] for s in sample_list])
    y_true = np.array([true_id_map[true_labels[s]] for s in sample_list])

    # Map string IDs to integer indices
    pred_ids = list(pred_sets.keys())
    true_ids = list(true_sets.keys())
    pred_id_to_idx = {id: i for i, id in enumerate(pred_ids)}
    true_id_to_idx = {id: i for i, id in enumerate(true_ids)}

    # Build cost matrix (using negative intersection size, as linear_assignment finds minimum)
    D = max(len(pred_ids), len(true_ids))
    cost_matrix = np.zeros((D, D), dtype=int)

    for i, pred_id in enumerate(pred_ids):
        for j, true_id in enumerate(true_ids):
            intersection = len(pred_sets[pred_id] & true_sets[true_id])
            cost_matrix[i, j] = -intersection  # Take negative value to find maximum intersection

    # Use Hungarian algorithm to find optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Calculate Cluster Accuracy
    total_correct = 0
    for i, j in zip(row_ind, col_ind):
        if i < len(pred_ids) and j < len(true_ids):
            pred_id = pred_ids[i]
            true_id = true_ids[j]
            total_correct += len(pred_sets[pred_id] & true_sets[true_id])

    total_samples = len(all_samples)
    cluster_accuracy = total_correct / total_samples if total_samples > 0 else 0

    # Restore macro average accuracy calculation
    true_accuracies = []
    for true_id, true_set in true_sets.items():
        max_intersection = max(
            (len(pred_set & true_set) for pred_set in pred_sets.values()),
            default=0
        )
        true_accuracy = max_intersection / len(true_set) if len(true_set) > 0 else 0
        true_accuracies.append(true_accuracy)

    macro_accuracy = sum(true_accuracies) / len(true_accuracies) if true_accuracies else 0

    # Calculate standardized metrics (ARI and NMI)
    # ari = adjusted_rand_score(y_true, y_pred) if total_samples > 0 else 0
    nmi = normalized_mutual_info_score(y_true, y_pred) if total_samples > 0 else 0

    return {
        'cluster_accuracy': {
            'accuracy': cluster_accuracy,
            'correct': total_correct,
            'total': total_samples
        },
        'macro': {
            'accuracy': macro_accuracy
        },
        'standardized': {
            # 'ari': ari,
            'nmi': nmi
        }
    }


if __name__ == "__main__":
    # File paths
    pred_file = "event_alignment_text\\event_alignment_text.jsonl"  # ← Put corresponding detection result data file
    true_file = "data\\new_incidents.jsonl"  # ← Modify true file path

    # Read clustering results
    pred_clusters = read_jsonl(pred_file)
    true_clusters = read_jsonl(true_file)

    # Calculate evaluation metrics
    metrics = compute_metrics(pred_clusters, true_clusters)

    # Output results
    print("===== Cluster Accuracy =====")
    print(f"Accuracy: {metrics['cluster_accuracy']['accuracy']:.4f} ({metrics['cluster_accuracy']['correct']}/{metrics['cluster_accuracy']['total']})")

    print("\n===== Macro Metrics =====")
    print(f"Accuracy: {metrics['macro']['accuracy']:.4f}")

    print("\n===== Standardized Metrics =====")
    # print(f"Adjusted Rand Index (ARI): {metrics['standardized']['ari']:.4f}")
    print(f"Normalized Mutual Information (NMI): {metrics['standardized']['nmi']:.4f}")