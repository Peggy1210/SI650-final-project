from transformers import pipeline
import pandas as pd
from ranker import Ranker
from random import sample

class MajorClassRanker(Ranker):
    def __init__(self, categorized_file_name: str) -> None:
        """
        Initializes an LLM object.

        Args:
            categorized_file_name: 

        """

        data = pd.read_csv(categorized_file_name)

        cat_cnt = data["predicted"].value_counts()
        categories = list(cat_cnt.index)
        
        categories_to_docids = { cat: [] for cat in categories }
        for i in range(len(data)):
            doc = data.iloc[i]
            categories_to_docids[doc["predicted"]].append((doc["job_id"], doc["scores"]))
            
        # for cat in categories:
        #     categories_to_docids[cat] = sorted(categories_to_docids[cat], key=lambda item: item[1], reverse=True)

        self.categories_to_docid = categories_to_docids
        self.major_cat = list(cat_cnt.nlargest(1).index)[0]

    
    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Predicts the category of a give query resume and returns the postings in that class

        Args:
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            
        """
        return self.categories_to_docid[self.major_cat]

class RandomClassRanker(Ranker):
    def __init__(self, categorized_file_name: str) -> None:
        """
        Initializes an LLM object.

        Args:
            categorized_file_name: 

        """

        data = pd.read_csv(categorized_file_name)

        cat_cnt = data["predicted"].value_counts()
        categories = list(cat_cnt.index)
        
        categories_to_docids = { cat: [] for cat in categories }
        for i in range(len(data)):
            doc = data.iloc[i]
            categories_to_docids[doc["predicted"]].append((doc["job_id"], doc["scores"]))
            
        # for cat in categories:
        #     categories_to_docids[cat] = sorted(categories_to_docids[cat], key=lambda item: item[1], reverse=True)

        self.categories_to_docid = categories_to_docids
        self.categories = categories
        # self.return_cat = sample(categories, 1)
        self.used_cat = []

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Predicts the category of a give query resume and returns the postings in that class

        Args:
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            
        """
        cat = sample(self.categories, 1)[0]
        self.used_cat.append(cat)
        return self.categories_to_docid[cat]


class LLMRanker(Ranker):
    def __init__(self, classifier_model_name: str, categorized_file_name: str) -> None:
        """
        Initializes an LLM object.

        Args:
            classifier_model_name: 
            categorized_file_name: 

        """
        self.classifier = pipeline("text-classification", model=classifier_model_name, device='cpu') # Initialize the bi-encoder model here
        
        data = pd.read_csv(categorized_file_name)

        cat_cnt = data["predicted"].value_counts()
        categories = list(cat_cnt.index)
        categories_to_docids = { cat: [] for cat in categories }
        for i in range(len(data)):
            categories_to_docids[data.iloc[i]["predicted"]].append((data.iloc[i]["job_id"], data.iloc[i]["scores"]))

        for cat in categories:
            categories_to_docids[cat] = sorted(categories_to_docids[cat], key=lambda item: item[1], reverse=True)

        self.categories = categories
        self.categories_to_docid = categories_to_docids

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Predicts the category of a give query resume and returns the postings in that class.

        Args:
            query: The query in its original form (no stopword filtering/tokenization).

        Returns:
            Returns the sorted postings in the query's categories.
        """
        if len(self.categories) == 0:
            return []
        
        if query == '' or query == None:
            return []

        query_cat = self.classifier(query[:512])[0]["label"]
        
        return self.categories_to_docid[query_cat]


