from typing import List, Dict
from collections import defaultdict

def group_by_cluster(utterances: List[str], labels: List[int]) -> Dict[int, List[str]]:
    threads = defaultdict(list)
    for utterance, label in zip(utterances, labels):
        threads[label].append(utterance)
    return dict(threads)

def print_disentangled_threads(labeled_threads: Dict[int, List[str]], thread_labels: Dict[int, str] = None):
    for thread_id, utterances in labeled_threads.items():
        header = f"🧵 Thread {thread_id}"
        if thread_labels:
            header += f": {thread_labels[thread_id]}"
        print(header)
        print("-" * len(header))
        for u in utterances:
            print(f"• {u}")
        print("\n")