import lightgbm

from tqdm import tqdm
from document_preprocessor import Tokenizer
from indexing import InvertedIndex, BasicInvertedIndex
from ranker import *
from collections import Counter
from pandas import DataFrame
from transformers import pipeline
import pandas as pd
import numpy as np
import csv
import os
import json


class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor', docid_to_company_rating = None) -> None:
        """
        Initializes a L2RRanker model.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object ** hw3 modified **
            feature_extractor: The L2RFeatureExtractor object
        """
        # TODO: Save any arguments that are needed as fields of this class
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.ranker = ranker
        self.feature_extractor = feature_extractor
        self.docid_to_company_rating = docid_to_company_rating

        # TODO: Initialize the LambdaMART model (but don't train it yet)
        self.model = LambdaMART() # This should a LambdaMART object

    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]], process_data_path):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores (dict): A dictionary of queries mapped to a list of 
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            tuple: A tuple containing the training data in the form of three lists: x, y, and qgroups
                X (list): A list of feature vectors for each query-document pair
                y (list): A list of relevance scores for each query-document pair
                qgroups (list): A list of the number of documents retrieved for each query
        """
        # NOTE: qgroups is not the same length as X or y.
        # This is for LightGBM to know how many relevance scores we have per query.
        X = []
        y = []
        qgroups = []

        # TODO: for each query and the documents that have been rated for relevance to that query,
        # process these query-document pairs into features
        for i, query in tqdm(enumerate(query_to_document_relevance_scores.keys()), total=len(query_to_document_relevance_scores.keys())):
    
            q_term = self.document_preprocessor.tokenize(query)
            
            # TODO: Accumulate the token counts for each document's title and content here
            doc_word_count = self.accumulate_doc_term_counts(self.document_index, q_term)
            title_word_count = self.accumulate_doc_term_counts(self.title_index, q_term)

            # TODO: For each of the documents, generate its features, then append
            # the features and relevance score to the lists to be returned
            for docid, rel in query_to_document_relevance_scores[query]:
                d_word_count = doc_word_count[docid] if docid in doc_word_count.keys() else {}
                t_word_count = title_word_count[docid] if docid in title_word_count.keys() else {}

                
                features = self.feature_extractor.generate_features(docid, d_word_count, t_word_count, q_term, query)
                X.append(features)
                y.append(rel)

            # TODO: Make sure to keep track of how many scores we have for this query in qrels
            qgroups.append(len(query_to_document_relevance_scores[query]))

            if i % 50 == 0:
                self.save_data(X, y, qgroups, process_data_path)

        self.save_data(X, y, qgroups, process_data_path)
            
        return X, y, qgroups

    def save_data(self, X_train, y_train, qgroups, process_data_filepath):
        os.makedirs(process_data_filepath, exist_ok=True)
        np.save(f'{process_data_filepath}/X_train.npy', X_train)
        np.save(f'{process_data_filepath}/y_train.npy', y_train)
        np.save(f'{process_data_filepath}/qgroups.npy', qgroups)

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        # TODO: Retrieve the set of documents that have each query word (i.e., the postings) and
        # create a dictionary that keeps track of their counts for the query word
        doc_term_counts = {}
        for q_term in query_parts:
            postings = index.get_postings(q_term)
            if postings == []: continue
            for posting in postings:
                docid = posting[0]
                freq = posting[1]
                if docid not in doc_term_counts.keys():
                    doc_term_counts[docid] = { q_term: freq }
                else:
                    doc_term_counts[docid][q_term] = freq

        return doc_term_counts

    def train(self, training_data_filename: str, raw_text_filename: str, process_data_path: str = "l2r_training_data") -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
            raw_text_filename (str)
        """
        if os.path.exists(process_data_path):
            print("load training data from path...")
            X_train = np.load(f'{process_data_path}/X_train.npy')
            y_train = np.load(f'{process_data_path}/y_train.npy')
            qgroups_train = np.load(f'{process_data_path}/qgroups.npy')
        else:
            print("creating training data...")
            raw_text_file = pd.read_csv(raw_text_filename)
            # TODO: Convert the relevance data into the right format for training data preparation
            query_to_document_relevance = {}
            with open(training_data_filename, 'r', newline='', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # query = row['query']
                    docid = int(row['job_id'])
                    query = raw_text_file[raw_text_file["ID"] == int(row['resume_id'])]["Clean_Resume"].item()
                    rel = int(float(row['rel']) * 2)
                    if query in query_to_document_relevance.keys():
                        query_to_document_relevance[query].append((docid, rel))
                    else:
                        query_to_document_relevance[query] = [(docid, rel)]
    
            # TODO: prepare the training data by featurizing the query-doc pairs and
            # getting the necessary datastructures
            X_train, y_train, qgroups_train = self.prepare_training_data(query_to_document_relevance, process_data_path)

        print("training...")
        # TODO: Train the model
        self.model.fit(X_train, y_train, qgroups_train)


    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # TODO: Return a prediction made using the LambdaMART model
        return self.model.predict(X)

    def query(self, query: str, feedback: dict[int, int] = None) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        # TODO: Retrieve potentially-relevant documents
        q_term = self.document_preprocessor.tokenize(query)

        # TODO: Fetch a list of possible documents from the index and create a mapping from
        # a document ID to a dictionary of the counts of the query terms in that document.
        # You will pass the dictionary to the RelevanceScorer as input.
        #
        # NOTE: we collect these here (rather than calling a Ranker instance) because we'll
        # pass these doc-term-counts to functions later, so we need the accumulated representations

        # TODO: Accumulate the documents word frequencies for the title and the main body
        doc_word_counts = self.accumulate_doc_term_counts(self.document_index, q_term)
        title_word_counts = self.accumulate_doc_term_counts(self.title_index, q_term)
        if doc_word_counts == {} and title_word_counts == {}: return []
        
        # TODO: Score and sort the documents by the provided scrorer for just the document's main text (not the title)
        # This ordering determines which documents we will try to *re-rank* using our L2R model
        scores = self.ranker.query(query, feedback)

        # TODO: Filter to just the top 100 documents for the L2R part for re-ranking
        FILTERED_INDEX = 100
        scores_filtered = scores[:FILTERED_INDEX]

        # TODO: Construct the feature vectors for each query-document pair in the top 100
        docid_test = []
        X_test = []
        for (docid, rel) in scores_filtered:
            docid_test.append(docid)
            doc_wc = doc_word_counts[docid] if docid in doc_word_counts.keys() else {}
            title_wc = title_word_counts[docid] if docid in title_word_counts.keys() else {}
            X_test.append(self.feature_extractor.generate_features(docid, doc_wc, title_wc, q_term, query))

        # TODO: Use your L2R model to rank these top 100 documents
        results = self.model.predict(X_test)

        # TODO: Sort posting_lists based on scores
        results = sorted(zip(docid_test, results), key=lambda pair: pair[1], reverse=True) #zip(docids, scores_filtered)

        # TODO: Make sure to add back the other non-top-100 documents that weren't re-ranked
        results += scores[FILTERED_INDEX:]

        if self.docid_to_company_rating is not None:
            # rerank based on company ratings
            results = self.rerank_from_ratings(results)

        # TODO: Return the ranked documents
        return results
    
    def rerank_from_ratings(self, results, top_n=20, alpha=0.1) -> list[tuple[int, float]]:
        """
        Reranks the results based on company ratings.
        """

        top_n = min(top_n, len(results))
        top_n_results = results[:top_n]
        reranked_results = []
        for docid, score in top_n_results:
            if docid in self.docid_to_company_rating and self.docid_to_company_rating[docid] >= 0.7:
                score = score * (1 + alpha * self.docid_to_company_rating[docid])
            reranked_results.append((docid, score))
        
        reranked_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)
        reranked_results += results[top_n:]

        return reranked_results



class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 doc_category_info: dict[int, list[str]],
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 # recognized_categories: set[str], docid_to_network_features: dict[int, dict[str, float]],
                 ce_scorer: CrossEncoderScorer=None,
                 skill_scorer: SkillSimilarityScorer=None,
                 classifier_model_name: str=None, label2id: dict[str, int]=None, label2cat: dict[str, str]=None, fields: list[str]=None) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
            ce_scorer: The CrossEncoderScorer object
        """
        # TODO: Set the initial state using the arguments
        self.document_index = document_index
        self.title_index = title_index
        self.doc_category_info = doc_category_info
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        # self.recognized_categories = recognized_categories
        # self.docid_to_network_features = docid_to_network_features
        self.ce_scorer = ce_scorer
        self.skill_scorer = skill_scorer
        if classifier_model_name:
            self.classifier = pipeline("text-classification", model=classifier_model_name, device='cpu')
            self.label2id = label2id
            self.label2cat = label2cat
            self.fields = fields
        else:
            self.classifier = None

    # TODO: Article Length
    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.document_index.get_doc_metadata(docid)['length']

    # TODO: Title Length
    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.title_index.get_doc_metadata(docid)['length']

    # TODO: TF
    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """

        tf = 0
        for q_term in query_parts:
            if q_term and q_term in index.index:
                if q_term in word_counts.keys():
                    doc_tf = word_counts[q_term]
                else:
                    doc_tf = 0

                if doc_tf > 0:
                    tf += np.log(doc_tf + 1)
        return tf

    # TODO: TF-IDF
    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        tf_idf = TF_IDF(index)
        return tf_idf.score(docid, word_counts, Counter(query_parts))
    

    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        # TODO: Calculate the BM25 score and return it
        bm25 = BM25(self.document_index)
        return bm25.score(docid, doc_word_counts, Counter(query_parts))
    

    # TODO: Pivoted Normalization
    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        # TODO: Calculate the pivoted normalization score and return it
        pivoted_n = PivotedNormalization(self.document_index)
        return pivoted_n.score(docid, doc_word_counts, Counter(query_parts))

    def get_skill_similarity_score(self, docid: int, query: str) -> float:
        return self.skill_scorer.score(docid, query)

    # TODO: Document Categories
    def get_document_categories(self, docid: int) -> list:
        """
        Generates a list of binary features indicating which of the recognized categories that the document has.
        Category features should be deterministically ordered so list[0] should always correspond to the same
        category. For example, if a document has one of the three categories, and that category is mapped to
        index 1, then the binary feature vector would look like [0, 1, 0].

        Args:
            docid: The id of the document

        Returns:
            A list containing binary list of which recognized categories that the given document has.
        """
        return self.doc_category_info[docid]

    # TODO (HW3): Cross-Encoder Score
    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:   
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """        
        return self.ce_scorer.score(docid, query)

    def get_query_category(self, query):
        cls = self.classifier(query[:512])[0]["label"]
        return [1 if i == self.label2id[cls] else 0 for i in range(len(self.label2id))] + \
                 [1 if self.label2cat[cls] == self.fields[i] else 0 for i in range(len(self.fields))]

    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str],  query) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """

        feature_vector = []

        # TODO: Document Length
        feature_vector.append(self.get_article_length(docid))

        # TODO: Title Length
        feature_vector.append(self.get_title_length(docid))

        # TODO Query Length
        feature_vector.append(len(query_parts))

        # TODO: TF (document)
        feature_vector.append(self.get_tf(self.document_index, docid, doc_word_counts, query_parts))

        # TODO: TF-IDF (document)
        feature_vector.append(self.get_tf_idf(self.document_index, docid, doc_word_counts, query_parts))

        # TODO: TF (title)
        feature_vector.append(self.get_tf(self.title_index, docid, title_word_counts, query_parts))

        # TODO: TF-IDF (title)
        feature_vector.append(self.get_tf_idf(self.title_index, docid, title_word_counts, query_parts))

        # TODO: BM25
        feature_vector.append(self.get_BM25_score(docid, doc_word_counts, query_parts))

        # TODO: Pivoted Normalization
        feature_vector.append(self.get_pivoted_normalization_score(docid, doc_word_counts, query_parts))

        # TODO: (HW3) Cross-Encoder Score
        if self.ce_scorer is not None:
            feature_vector.append(self.get_cross_encoder_score(docid, query))

        if self.skill_scorer is not None:
            feature_vector.append(self.get_skill_similarity_score(docid, query))

        if self.classifier is not None:
            feature_vector += self.get_query_category(query)

        # TODO: Calculate the Document Categories features.
        # NOTE: This should be a list of binary values indicating which categories are present.
        feature_vector += self.get_document_categories(docid)

        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 20,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.005,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to
            # the number of CPUs on your machine for faster training
            "n_jobs": 1,
            "force_col_wise": True
            # "verbosity": 1,
        }

        if params:
            default_params.update(params)

        # TODO: initialize the LGBMRanker with the provided parameters and assign as a field of this class
        self.ranker = lightgbm.LGBMRanker(**default_params)


    def fit(self,  X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples.
            y_train (array-like): Target values.
            qgroups_train (array-like): Query group sizes for training data.

        Returns:
            self: Returns the instance itself.
        """

        # TODO: fit the LGBMRanker's parameters using the provided features and labels
        self.ranker.fit(X_train, y_train, group=qgroups_train)
        return self

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length.

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """

        # TODO: Generating the predicted values using the LGBMRanker
        return self.ranker.predict(featurized_docs)

    def load(self, model_name):
        self.ranker = lightgbm.LGBMRanker(model_name=model_name)
