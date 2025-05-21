from sklearn.cluster import AgglomerativeClustering
from typing import List
import numpy as np

class Clusterer:
    def __init__(self, n_clusters: int = None, distance_threshold: float = 1.5):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold

    def cluster(self, embeddings: np.ndarray) -> List[int]:
        clustering_model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            distance_threshold=None if self.n_clusters else self.distance_threshold,
            affinity='euclidean',
            linkage='ward'
        )
        labels = clustering_model.fit_predict(embeddings)
        return labels.tolist()