"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (marginally relevant) and 2 (very relevant). When you calculate MAP, treat 1s and 2s are relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant. Thus, we have 
three relevance levels: 0 (non-relevant), 1 (marginally relevant), and 2 (very relevant). 
"""

from tqdm import tqdm
import numpy as np
import time


def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    # TODO: Implement MAP
    cnt = 0
    score = 0
    cut_off = min(cut_off, len(search_result_relevances))
    for i in range(0, cut_off):
        if search_result_relevances[i] == 1:
            cnt += 1
            score += cnt / (i + 1)

    return score / cut_off


def ndcg_score(search_result_relevances: list[float],
               ideal_relevance_score_ordering: list[float], cut_off: int = 10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: Implement NDCG
    dcg = 0
    idcg = 0
    cut_off = min(cut_off, len(search_result_relevances))
    for i in range(0, cut_off):
        if i == 0:
            dcg += search_result_relevances[i]
            idcg += ideal_relevance_score_ordering[i]
        else:
            dcg += search_result_relevances[i] / (np.log(i + 1) / np.log(2))
            idcg += ideal_relevance_score_ordering[i] / (np.log(i + 1) / np.log(2))
    if idcg == 0: return 0
    return dcg / idcg


# TODO:
def f1_score(search_result_relevances: list[float]):
    pass


import pandas as pd
import json

def run_relevance_tests(postings_category_filename: str, resume_category_filename: str, job_category_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.

    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    if postings_category_filename.endswith('.gz'):
        postings_category = pd.read_csv(postings_category_filename, compression='gzip')
    else:
        postings_category = pd.read_csv(postings_category_filename)
    
    if resume_category_filename.endswith('.gz'):
        resume_category = pd.read_csv(resume_category_filename, compression='gzip')
    else:
        resume_category = pd.read_csv(resume_category_filename)

    with open(job_category_filename) as f:
        job_category = json.load(f)

    postings_to_category = {}
    for i in range(len(postings_category)):
        postings_to_category[postings_category.iloc[i]["job_id"]] = postings_category.iloc[i]["predicted"]
    # relevance_data = pd.read_csv(relevance_data_filename, encoding = "ISO-8859-1")
    # queries = relevance_data['query'].value_counts().keys()
    # document_set = { query: {} for query in queries }
    # for i in range(len(relevance_data)):
    #     document_set[relevance_data.iloc[i]['query']][relevance_data.iloc[i]['docid']] = {
    #         'docid': relevance_data.iloc[i]['docid'],
    #         'rel': relevance_data.iloc[i]['rel'],
    #         'rel-map': 1 if relevance_data.iloc[i]['rel'] > 3 else 0,
    #     }

    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance 
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.
    # NOTE: NDCG can use any scoring range, so no conversion is needed.

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out
    map_score_list = []
    ndcg_score_list = []
    map_loose_score_list = []
    ndcg_loose_score_list = []
    time_list = []
    # for query in tqdm(queries):
    
    for i in tqdm(range(len(resume_category))):
        start = time.time()
        res_list = ranker.query(resume_category.iloc[i]["Resume_str"])
        query_cat = resume_category.iloc[i]["Category"]

        # Get map_relevance and ndcg_relevance
        map_relevance = []
        ndcg_relevance = []
        map_loose_relevance = []
        ndcg_loose_relevance = []
        for (docid, score) in res_list:
            posting_cat = postings_to_category[docid]
            if posting_cat == query_cat:
                map_relevance.append(1)
                ndcg_relevance.append(5)
                map_loose_relevance.append(1)
                ndcg_loose_relevance.append(5)
            elif job_category[posting_cat] == job_category[query_cat]:
                map_relevance.append(0)
                ndcg_relevance.append(0)
                map_loose_relevance.append(1)
                ndcg_loose_relevance.append(3)
            else:
                map_relevance.append(0)
                ndcg_relevance.append(0)
                map_loose_relevance.append(0)
                ndcg_loose_relevance.append(0)
        end = time.time()
        time_list.append(end-start)
        map_score_list.append(map_score(map_relevance))
        ndcg_score_list.append(ndcg_score(ndcg_relevance, sorted(ndcg_relevance, reverse=True)))
        map_loose_score_list.append(map_score(map_loose_relevance))
        ndcg_loose_score_list.append(ndcg_score(ndcg_loose_relevance, sorted(ndcg_loose_relevance, reverse=True)))
    
        
    # # TODO: Compute the average MAP and NDCG across all queries and return the scores
    # # NOTE: You should also return the MAP and NDCG scores for each query in a list
    return { 
             'map': sum(map_score_list)/len(map_score_list),
             'ndcg': sum(ndcg_score_list)/len(ndcg_score_list),
             'map_list': map_score_list,
             'ndcg_list': ndcg_score_list,
             'map_loose': sum(map_loose_score_list)/len(map_loose_score_list),
             'ndcg_loose': sum(ndcg_loose_score_list)/len(ndcg_loose_score_list),
             'map_loose_list': map_loose_score_list,
             'ndcg_loose_list': ndcg_loose_score_list,
             'time': time_list
            }
