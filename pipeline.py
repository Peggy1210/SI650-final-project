import csv
import gzip
import json
import jsonlines
import os
import pickle
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from pypdf import PdfReader
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download(['stopwords','wordnet'])

from models import BaseSearchEngine, SearchResponse

# project library imports go here
from document_preprocessor import RegexTokenizer
from indexing import *
from ranker import *
from l2r import L2RRanker, L2RFeatureExtractor
from vector_ranker import VectorRanker
from network_features import NetworkFeatures

DATA_PATH = 'data/'
CACHE_PATH = '__cache__/'

DATASET_PATH = DATA_PATH + 'postings_dataset.jsonl'
POSTINGS_CATEGORY_PATH = DATA_PATH + 'predicted_postings.csv.gz'
JOB_CATEGORY_PATH = DATA_PATH + 'job_categories.json'
RELEVANCE_TRAIN_PATH = DATA_PATH + 'train_relevance.csv'
ENCODED_DOCUMENT_EMBEDDINGS_NPY_PATH = DATA_PATH + 'description_embeddings.all-MiniLM-L6-v2.npy'
AUGMENTED_ENCODED_DOCUMENT_EMBEDDINGS_NPY_PATH = DATA_PATH + 'augmented_description_embeddings.all-MiniLM-L6-v2.npy'
DOC_IDS_PATH = DATA_PATH + 'job_posting_ids.txt'
LLM_CLASSIFIER_PATH = 'posting_classifier/models/job_classification_11091504'
SKILLS_EMBEDDINGS_PATH = DATA_PATH + 'skills_embeddings.npy'
RESUME_PATH = DATA_PATH + 'resume_data.csv'
COMPANY_RATINGS_PATH = DATA_PATH + 'company_reviews.csv'

category_info_filename = 'data/category_info.json'
label2id_filename = 'data/label2id.json'

TITLE_INDEXING_DIR = DATA_PATH + 'index-title'
MAIN_INDEXING_DIR = DATA_PATH + 'index-description_mwf10'

DISABLE_TQDM = True

def tqdm_replacement(iterable_object,*args,**kwargs):
    return iterable_object

class SearchEngine(BaseSearchEngine):
    def __init__(self,
                 max_docs: int = -1,
                 ranker: str = 'BM25',
                 l2r: bool = False,
                 aug_docs: bool = True,
                 company_ratings: bool = True,
                 ) -> None:
        # 1. Create a document tokenizer using document_preprocessor Tokenizers
        # 2. Load stopwords, network data, categories, etc
        # 3. Create an index using the Indexer and IndexType (with the Wikipedia JSONL and stopwords)
        # 4. Initialize a Ranker/L2RRanker with the index, stopwords, etc.
        # 5. If using L2RRanker, train it here.

        if DISABLE_TQDM:
            tqdm.tqdm = tqdm_replacement

        self.l2r = False

        print('Initializing Search Engine...')

        ### Initialize stopwords
        self.stopwords = set(stopwords.words("english"))

        print(" - Initializing tokenizer")
        self.preprocessor = RegexTokenizer('\w+')

        ### 
        # self.doc_augment_dict = None
        # if aug_docs:
        #     print('Loading doc augment dict...')
        #     self.doc_augment_dict = defaultdict(lambda: [])
        #     with open(DOC2QUERY_PATH, 'r') as f:
        #         data = csv.reader(f)
        #         for idx, row in tqdm(enumerate(data)):
        #             if idx == 0:
        #                 continue
        #             self.doc_augment_dict[row[0]].append(row[2])

        print(' - Initializing document indexing')
        if os.path.exists(TITLE_INDEXING_DIR):
            self.title_index = BasicInvertedIndex()
            self.title_index.load(TITLE_INDEXING_DIR)
        else:
            self.title_index = Indexer.create_index(
                IndexType.BasicInvertedIndex, DATASET_PATH, self.preprocessor, self.stopwords, 0, text_key='title')
        
        if os.path.exists(MAIN_INDEXING_DIR):
            self.main_index = BasicInvertedIndex()
            self.main_index.load(MAIN_INDEXING_DIR)
        else:
            self.main_index = Indexer.create_index(
                IndexType.BasicInvertedIndex, DATASET_PATH, self.preprocessor, self.stopwords, 10, max_docs=max_docs, text_key="description")

        self.docids = []
        with open(DOC_IDS_PATH, 'r') as f:
            for line in tqdm(f):
                self.docids.append(int(line))

        self.raw_text_dict = {}
        if DATASET_PATH.endswith('.gz'):
            with gzip.open(DATASET_PATH, 'rt', encoding='utf-8') as f:
                for line in tqdm(f):
                    print(line)
                    doc = json.loads(line)
                    self.raw_text_dict[doc['job_id']] = doc['description']
        else:
            with jsonlines.open(DATASET_PATH) as f:
                for doc in tqdm(f):
                    self.raw_text_dict[doc['job_id']] = doc['description']

        self.company_ratings = company_ratings
        self.aug_docs = aug_docs

        print('Loading ranker...')
        self.set_ranker(ranker)
        self.set_l2r(l2r)

        print('Search Engine initialized!')

    def set_ranker(self, ranker: str = 'BM25') -> None:
        if ranker == 'VectorRanker':
            if self.aug_docs:
                with open(AUGMENTED_ENCODED_DOCUMENT_EMBEDDINGS_NPY_PATH, 'rb') as f:
                    self.encoded_docs = np.load(f).astype(np.float32)
            else:
                with open(ENCODED_DOCUMENT_EMBEDDINGS_NPY_PATH, 'rb') as f:
                    self.encoded_docs = np.load(f)
            with open(DOC_IDS_PATH, 'r') as f:
                self.row_to_docid = [int(line.strip()) for line in f]
            self.ranker = VectorRanker(
                'sentence-transformers/msmarco-MiniLM-L12-cos-v5',
                self.encoded_docs, self.row_to_docid
            )
        else:
            if ranker == 'BM25':
                self.scorer = BM25(self.main_index)
            # elif ranker == "PersonalizedBM25":
            #     self.scorer = PersonalizedBM25(self.main_index)
            elif ranker == "WordCountCosineSimilarity":
                self.scorer = WordCountCosineSimilarity(self.main_index)
            elif ranker == "DirichletLM":
                self.scorer = DirichletLM(self.main_index)
            elif ranker == "PivotedNormalization":
                self.scorer = PivotedNormalization(self.main_index)
            elif ranker == "TF_IDF":
                self.scorer = TF_IDF(self.main_index)
            else:
                raise ValueError("Invalid ranker type")
            self.ranker = Ranker(
                self.main_index, self.preprocessor, self.stopwords,
                self.scorer)#, self.raw_text_dict)
        if self.l2r:
            self.pipeline.ranker = self.ranker
        else:
            self.pipeline = self.ranker

    def set_l2r(self, l2r: bool = True) -> None:
        if self.l2r == l2r:
            return
        if not l2r:
            self.pipeline = self.ranker
            self.l2r = False
        else:
            print('Initializing L2R ranker...')
            print(' - loading categories...')
            with open(JOB_CATEGORY_PATH) as f:
                label2cat = json.load(f)
            
            fields = []
            for i in label2cat:
                fields.append(label2cat[i])
            fields = list(set(fields))
            with open(category_info_filename) as f:
                temp = json.load(f)
        
            doc_category = {}
            for key in temp:
                doc_category[int(key)] = temp[key]
            del temp
            with open(label2id_filename) as f:
                label2id = json.load(f)
            
            print(" - initializing skills scorer...")
            skills_embeddings = np.load(SKILLS_EMBEDDINGS_PATH)
            self.skill_scorer = SkillSimilarityScorer(skills_embeddings, self.docids, 
                                                      skill_pattern_path="preprocessing/jz_skill_patterns.jsonl")

            print(" - initializing cross encoder scorer")
            self.cemodel = CrossEncoderScorer(self.raw_text_dict)
            
            print(' - initializing feature extractor')
            self.fe = L2RFeatureExtractor(self.main_index,
                            self.title_index,
                            doc_category,
                            self.preprocessor,
                            self.stopwords,
                            # recognized_categories,
                            # network_features
                            self.cemodel,
                            self.skill_scorer,
                            LLM_CLASSIFIER_PATH, label2id, label2cat, fields
                        )

            if self.company_ratings:
                print(' - initializing company ratings dict')
                company_ratings = {}
                company_reviews = pd.read_csv(COMPANY_RATINGS_PATH, sep=';')
                for idx, row in company_reviews.iterrows():
                    if pd.isna(row['name']) or pd.isna(row['rating']):
                        continue
                    name = row['name'].lower()
                    name = ''.join(char for char in name if char.isalnum())
                    rating = round(float(row['rating']) / 5, 2)
                    company_ratings[name] = rating
    
                self.docid_to_company_rating = {}
                with jsonlines.open(DATASET_PATH) as f:
                    for doc in tqdm(f):
                        if pd.isna(doc['company_name']):
                            continue
                        name = doc['company_name'].lower()
                        name = ''.join(char for char in name if char.isalnum())
                        if name in company_ratings:
                            self.docid_to_company_rating[doc['job_id']] = company_ratings[name]
                print('Ratings for', len(self.docid_to_company_rating), 'companies loaded.')
            else:
                self.docid_to_company_rating = None


            self.pipeline = L2RRanker(
                self.main_index, self.title_index, self.preprocessor,
                self.stopwords, self.ranker, self.fe, self.docid_to_company_rating
            )

            print('Training L2R ranker...')
            self.pipeline.train(RELEVANCE_TRAIN_PATH, RESUME_PATH)
            self.l2r = True

    def search(self, query: str, feedback: dict[int, int] = None) -> list[SearchResponse]:
        # 1. Use the ranker object to query the search pipeline
        # 2. This is example code and may not be correct.
        results = self.pipeline.query(query, feedback)
        if results == []: return []
        return [SearchResponse(id=idx+1, docid=result[0], score=result[1]) for idx, result in enumerate(results)]
    
    def search_pdf(self, filename: str, feedback: dict[int, int] = None) -> list[SearchResponse]:
        # 1. Use the ranker object to query the search pipeline
        # 2. This is example code and may not be correct.
        if not os.path.exists(filename): return []

        query = ""
        reader = PdfReader(filename)
        for page in reader.pages:
            query += page.extract_text() + " "

        results = self.pipeline.query(query, feedback)
        if results == []: return []
        return [SearchResponse(id=idx+1, docid=result[0], score=result[1]) for idx, result in enumerate(results)]


def initialize():
    search_obj = SearchEngine(max_docs=1000, ranker='VectorRanker', l2r=True)
    return search_obj

def main():
    search_obj = SearchEngine(max_docs=10000)
    # search_obj.set_l2r(True)

    testing_resume_data = pd.read_csv(RESUME_PATH)
    query = testing_resume_data.iloc[1]["Clean_Resume"]
    results = search_obj.search(query)
    print(results[:5])

    results = search_obj.search(DATA_PATH + "resume_sample_IT_10089434.pdf")
    print(results[:5])

def postingFormat(raw_data, docid):
    post = raw_data[raw_data['job_id'] == docid].iloc[0]
    print("Job title:", post["title"])
    print("Job description:", post["description"])

def printSearchResponse(raw_data, results: SearchResponse, num_display=5):
    if results == []:
        print("No posting found.")

    print(f"There are a total of {len(results)} postings found.\nThe top {num_display} search results of your resume is: ")

    for i in range(min(num_display, len(results))):
        docid = results[i].docid
        print(f"\n{i+1}\n===")
        postingFormat(raw_data, docid)

def interactive(**args):
    search_obj = SearchEngine(**args)
    raw_data = pd.read_csv(POSTINGS_CATEGORY_PATH)


    print("--------------")
    feedback_dict = {}
    while True:
        mode = input("Choose your mode (pdf/text): ")
        if mode != "pdf" and mode != "text":
            print("Invalid mode. Abort!")
        else:
            if mode == "text":
                query = input("Enter your resume text: ")
                results = search_obj.search(query, feedback_dict)
            elif mode == "pdf":
                query = input("Enter your resume file path: ")
                results = search_obj.search_pdf(query, feedback_dict)
            
            printSearchResponse(raw_data, results)

            ans = input("Provide feedback to get more relevant results? (Y/N): ")
            if ans == "Y":
                feedback = input("Input feedback in order (Relevant = 1, Non-Relevant = 0)")
            
                for i, char in enumerate(feedback):
                    feedback_dict[results[i].docid] = int(char)

            ans = input("End the program? (Y/N): ")
            if ans == "Y": break

if __name__ == '__main__':
    main()