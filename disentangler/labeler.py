from typing import List, Dict
from rake_nltk import Rake

class Labeler:
    def __init__(self):
        self.rake = Rake()

    def label_threads(self, clustered_threads: Dict[int, List[str]]) -> Dict[int, str]:
        labels = {}
        for cluster_id, utterances in clustered_threads.items():
            text = " ".join(utterances)
            self.rake.extract_keywords_from_text(text)
            keywords = self.rake.get_ranked_phrases()
            labels[cluster_id] = keywords[0] if keywords else f"Topic {cluster_id}"
        return labels