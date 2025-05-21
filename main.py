from disentangler.embedder import Embedder
from disentangler.clusterer import Clusterer
from disentangler.labeler import Labeler
from disentangler.utils import group_by_cluster, print_disentangled_threads

import json

def load_chat(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return [turn["message"] for turn in data]

def main():
    utterances = load_chat("sample_data/sample_chat.json")

    embedder = Embedder()
    embeddings = embedder.encode(utterances)

    clusterer = Clusterer(distance_threshold=1.2)
    labels = clusterer.cluster(embeddings)

    clustered_threads = group_by_cluster(utterances, labels)

    labeler = Labeler()
    thread_labels = labeler.label_threads(clustered_threads)

    print_disentangled_threads(clustered_threads, thread_labels)

if __name__ == "__main__":
    main()