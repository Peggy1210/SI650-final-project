from sentence_transformers import SentenceTransformer, util
import numpy as np
from ranker import Ranker


class VectorRanker(Ranker):
    def __init__(self, bi_encoder_model_name: str, encoded_docs: np.ndarray,
                 row_to_docid: list[int]) -> None:
        """
        Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoded
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding

        Using zip(encoded_docs, row_to_docid) should give you the mapping between the docid and the embedding.
        """
        # TODO: Instantiate the bi-encoder model here

        # NOTE: we're going to use the cpu for everything here so if you decide to use a GPU, do not 
        # submit that code to the autograder
        self.biencoder_model = SentenceTransformer(bi_encoder_model_name) # Initialize the bi-encoder model here
        
        # TODO: Also include other necessary initialization code
        # self.embedding = zip(encoded_docs, row_to_docid)
        self.encoded_docs = encoded_docs
        self.row_to_docid = row_to_docid
        self.docid_to_row = {docid: idx for idx, docid in enumerate(row_to_docid)}

    def query(self, query: str, feedback: dict[int, int] = None) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.

        Args:
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first or an empty list if a query cannot be encoded
            or no results are return
        """
        # NOTE: Do not forget to handle edge cases on the inputs
        if len(self.row_to_docid) == 0:
            return []
        
        if query == '' or query == None:
            return []

        # TODO: Encode the query using the bi-encoder
        q_embedding = self.biencoder_model.encode(query)

        if feedback:
            q_embedding = self.rocchio(q_embedding, feedback)

        # TODO: Score the similarity of the query vector and document vectors for relevance
        # Calculate the dot products between the query embedding and all document embeddings
        # similarities = self.biencoder_model.similarity(q_embedding, self.encoded_docs).numpy()[0]
        similarities = util.dot_score(q_embedding, self.encoded_docs).numpy().flatten()
        
        # TODO: Generate the ordered list of (document id, score) tuples
        scores = {}
        for i, docid in enumerate(self.row_to_docid):
            scores[docid] = similarities[i]
        # scores = list(zip(self.row_to_docid, similarities))
        # TODO: Sort the list so most relevant are first
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)
        # return sorted(scores, key=lambda item: item[1], reverse=True)



    def rocchio(self, query_embedding: np.ndarray, feedback: dict[int, int]) -> np.ndarray:

        alpha = 1.0
        beta = 0.75
        gamma = 0.15

        relevant_embeddings = []
        non_relevant_embeddings = []

        for docid, rel in feedback.items():
            if rel == 1:
                relevant_embeddings.append(self.encoded_docs[self.docid_to_row[docid]])
            elif rel == 0:
                non_relevant_embeddings.append(self.encoded_docs[self.docid_to_row[docid]])

        if relevant_embeddings:
            mean_relevant = np.mean(relevant_embeddings, axis=0)
        else:
            mean_relevant = np.zeros_like(query_embedding)

        if non_relevant_embeddings:
            mean_non_relevant = np.mean(non_relevant_embeddings, axis=0)
        else:
            mean_non_relevant = np.zeros_like(query_embedding)

        updated_query = alpha * query_embedding + beta * mean_relevant - gamma * mean_non_relevant

        return updated_query


