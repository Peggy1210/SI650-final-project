'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''

from enum import Enum
from document_preprocessor import Tokenizer
from collections import Counter, defaultdict
from tqdm import tqdm
import json
import jsonlines
import gzip
import os



class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex, PositionalIndex
    PositionalIndex = 'PositionalIndex'
    BasicInvertedIndex = 'BasicInvertedIndex'
    SampleIndex = 'SampleIndex'
    InvertedIndex = 'BasicInvertedIndex'


class InvertedIndex:
    """
    This class is the basic implementation of an in-memory inverted index. This class will hold the mapping of terms to their postings.
    The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
    and documents in the index. These metadata will be necessary when computing your relevance functions.
    """

    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        self.statistics = {}   # the central statistics of the index
        self.statistics['vocab'] = Counter()  # token count
        self.vocabulary = set()  # the vocabulary of the collection
        # metadata like length, number of unique tokens of the documents
        self.document_metadata = {}

        self.index = defaultdict(list)  # the index

    # NOTE: The following functions have to be implemented in the two inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        raise NotImplementedError

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        raise NotImplementedError

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError

def find_doc(docid: int, docs):
    for doc in docs:
        if doc[0] == docid: return doc
    return -1

class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        self.statistics['unique_token_count'] = len(self.vocabulary)
        self.statistics['total_token_count'] = 0
        self.statistics['stored_total_token_count'] = 0
        self.statistics['number_of_documents'] = 0
        self.statistics['mean_document_length'] = 0

    def remove_doc(self, docid: int) -> None:
        if docid not in self.document_metadata.keys():
            raise Exception("This is a risky 'delete' operation.")
        
        # Deleted the document id in each token index
        for token in self.index.keys():
            self.index[token].remove(find_doc(docid, self.index[token]))

        # Update statistics
        self.vocabulary = set(self.index.keys())
        self.statistics['vocab'] = Counter(self.vocabulary)
        self.statistics['unique_token_count'] = len(self.vocabulary)
        self.statistics['total_token_count'] -= self.document_metadata[docid]['length']
        # self.statistics['stored_total_token_count'] -= self.document_metadata[docid]['stored_length']
        self.statistics['number_of_documents'] -= 1
        if self.statistics['number_of_documents'] == 0:
            self.statistics['mean_document_length'] = 0
        else:
            self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']
        self.document_metadata.pop(docid)

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        # self.index = {
        #   "token": [(docid, freq), ...]
        # }

        # Adding document to index
        token_cnt = Counter(tokens)
        filtered_tokens = []
        for token in token_cnt.keys():
            if token is not None:
                filtered_tokens.append(token)
                self.index[token].append((docid, token_cnt[token]))

        # Update document metadata
        self.document_metadata[docid] =  {
                # "index_doc_id": docid,
                "unique_tokens": len(filtered_tokens),
                # "stored_length": len(filtered_tokens),
                "length": len(tokens),
            }

        # Update statistics
        self.vocabulary.update(filtered_tokens)
        self.statistics['vocab'].update(token_cnt) # token count
        self.statistics['unique_token_count'] = len(self.vocabulary)
        self.statistics['total_token_count'] += len(tokens)
        # self.statistics['stored_total_token_count'] += len(filtered_tokens)
        self.statistics['number_of_documents'] += 1
        self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']

    def get_postings(self, term: str) -> list:
        if term not in self.index.keys(): return []
        return self.index[term]

    def get_doc_metadata(self, docid: int) -> dict[str, int]:
        if docid not in self.document_metadata.keys(): return {}
        return self.document_metadata[docid]
    
    def get_term_metadata(self, term: str) -> dict[str, int]:
        if term not in self.index.keys(): return {}
        return {
            "term_count":  self.statistics['vocab'][term],
            "doc_frequency": len(self.index[term])
        }
            
    def get_statistics(self) -> dict[str, int]:
        return self.statistics

    def save(self, index_directory_name) -> None:
        os.makedirs(index_directory_name, exist_ok=True)
        index_details = {
            # "volcabulary": self.vocabulary,
            "document_metadata": self.document_metadata,
            "index": self.index,
            "statistics": self.statistics
        } 
        
        # Convert and write JSON object to file
        with open(index_directory_name + "/index.json", "w") as outfile: 
            json.dump(index_details, outfile, indent = 4)

    def load(self, index_directory_name) -> None:
        if not os.path.isdir(index_directory_name): raise FileNotFoundError('No such directory')
        with open(index_directory_name + '/index.json', 'r') as f:
            index_details = json.load(f)
        self.statistics = index_details["statistics"]
        self.statistics["vocab"] = Counter(self.statistics["vocab"])
        self.vocabulary = set(self.statistics["vocab"].keys())
        self.document_metadata = { int(docid): detail for docid, detail in index_details["document_metadata"].items() }
        self.index = index_details["index"]


class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        self.statistics['unique_token_count'] = len(self.vocabulary)
        self.statistics['total_token_count'] = 0
        self.statistics['stored_total_token_count'] = 0
        self.statistics['number_of_documents'] = 0
        self.statistics['mean_document_length'] = 0


    def remove_doc(self, docid: int) -> None:
        if docid not in self.document_metadata.keys():
            raise Exception("This is a risky 'delete' operation.")
        
        # Deleted the document id in each token index
        for token in self.index.keys():
            self.index[token].remove(find_doc(docid, self.index[token]))

        # Update statistics
        self.vocabulary = set(self.index.keys())
        self.statistics['vocab'] = Counter(self.vocabulary)
        self.statistics['unique_token_count'] = len(self.vocabulary)
        self.statistics['total_token_count'] -= self.document_metadata[docid]['length']
        # self.statistics['stored_total_token_count'] -= self.document_metadata[docid]['stored_length']
        self.statistics['number_of_documents'] -= 1
        if self.statistics['number_of_documents'] == 0:
            self.statistics['mean_document_length'] = 0
        else:
            self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']
        self.document_metadata.pop(docid)

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """Tokenize a document and add term ID """
        # self.index = {
        #   "token": [(docid, freq, [pos1, pos2, ...]), ...]
        # }

        # Adding documents to index
        pos_list = {}
        for pos, token in enumerate(tokens):
            if token is not None:
                if token not in pos_list.keys():
                    pos_list[token] = [pos]
                else:
                    pos_list[token].append(pos)
                
        filtered_tokens = pos_list.keys()
        for term in pos_list:
            self.index[term].append((docid, len(pos_list[term]), pos_list[term]))
        
        # Update document metadata
        self.document_metadata[docid] = {
                # "index_doc_id": docid,
                "unique_tokens": len(filtered_tokens),
                # "stored_length": len(filtered_tokens),
                "length": len(tokens) 
            }

        # Update statistics
        self.vocabulary.update(filtered_tokens)
        self.statistics['vocab'].update(filtered_tokens) # token count
        self.statistics['unique_token_count'] = len(self.vocabulary)
        self.statistics['total_token_count'] += len(tokens)
        # self.statistics['stored_total_token_count'] += len(filtered_tokens)
        self.statistics['number_of_documents'] += 1
        self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']

    def get_postings(self, term: str) -> list:
        if term not in self.index.keys(): return []
        return self.index[term]
    
    def get_doc_metadata(self, docid: int) -> dict[str, int]:
        if docid not in self.document_metadata.keys(): return {}
        return self.document_metadata[docid]

    def get_term_metadata(self, term: str) -> dict[str, int]:
        if term not in self.index.keys(): return {}
        return {
            "term_count":  self.statistics['vocab'][term],
            "doc_frequency": len(self.index[term])
        }

    def get_statistics(self) -> dict[str, int]:
        return self.statistics

    def save(self, index_directory_name) -> None:
        os.makedirs(index_directory_name, exist_ok=True)

        index_details = {
            # "volcabulary": self.vocabulary,
            "document_metadata": self.document_metadata,
            "index": self.index,
            "statistics": self.statistics
        } 
        
        # Convert and write JSON object to file
        with open(index_directory_name + "/index.json", "w") as outfile: 
            json.dump(index_details, outfile, indent = 4)

    def load(self, index_directory_name) -> None:
        if not os.path.isdir(index_directory_name): raise FileNotFoundError('No such directory')
        with open(index_directory_name + '/index.json', 'r') as f:
            index_details = json.load(f)
        self.statistics = index_details["statistics"]
        self.statistics["vocab"] = Counter(self.statistics["vocab"])
        self.vocabulary = set(self.statistics["vocab"].keys())
        self.document_metadata = { int(docid): detail for docid, detail in index_details["document_metadata"].items() }

        for idx, docs in index_details["index"].items():
            self.index[idx] = { int(docid): pos for docid, pos in docs.items() }


class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1, doc_augment_dict: dict[int, list[str]] | None = None) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the entire corpus at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
            doc_augment_dict: An optional argument; This is a dict created from the doc2query.csv where the keys are
                the document id and the values are the list of queries for a particular document.

        Returns:
            An inverted index

        '''
         # TODO (HW3): This function now has an optional argument doc_augment_dict; check README.md
       
        # HINT: Think of what to do when doc_augment_dict exists, how can you deal with the extra information?
        #       How can you use that information with the tokens?
        #       If doc_augment_dict doesn't exist, it's the same as before, tokenizing just the document text

        # TODO: Implement this class properly. This is responsible for going through the documents
        #       one by one and inserting them into the index after tokenizing the document

        # TODO: Figure out what type of InvertedIndex to create.
        #       For HW3, only the BasicInvertedIndex is required to be supported

        if index_type == IndexType.BasicInvertedIndex:
            index = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            index = PositionalInvertedIndex()
        else:
            index = SampleIndex()

        # TODO: If minimum word frequencies are specified, process the collection to get the
        #       word frequencies

        # NOTE: Make sure to support both .jsonl.gz and .jsonl as input

        tokens = []
        word_counts = Counter()
        document_cnt = 0

        # print("Tokenizing...")
        if dataset_path.endswith('.gz'):
            with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
                for line in tqdm(f):
                    doc = json.loads(line)
                    document_cnt += 1
                    if max_docs != -1 and document_cnt > max_docs: break

                    docid = doc['job_id']
                    text = doc[text_key]

                    if doc_augment_dict is not None:
                        if docid in doc_augment_dict.keys():
                            for aug in doc_augment_dict[docid]:
                                text = aug + " "+ text

                    token = document_preprocessor.tokenize(text)
                    word_counts.update(token)
                    tokens.append((docid, token))
        else:
            with jsonlines.open(dataset_path) as f:
                for doc in tqdm(f):
                    document_cnt += 1
                    if max_docs != -1 and document_cnt > max_docs: break

                    docid = doc['job_id']
                    text = doc[text_key]

                    # if text is None or text is "": continue

                    if doc_augment_dict is not None:
                        if docid in doc_augment_dict.keys():
                            for aug in doc_augment_dict[docid]:
                                text = aug + " "+ text

                    token = document_preprocessor.tokenize(text)
                    word_counts.update(token)
                    tokens.append((docid, token))
                      
        # TODO: Figure out which set of words to not index because they are stopwords or
        #       have too low of a frequency

        # Process to be deleted
        # print("Filtering stopwords and words less than minimum word frequency...")
        to_be_deleted = set()
        for word in word_counts.keys():
            if word in stopwords:
                to_be_deleted.add(word)
            if word_counts[word] < minimum_word_frequency:
                to_be_deleted.add(word)

        # TODO: Read the collection and process/index each document.
        #       Only index the terms that are not stopwords and have high-enough frequency
        # print("Adding documents...")
        for (docid, token) in tqdm(tokens):
            filtered = [ t if t not in to_be_deleted else None for t in token ]
            index.add_doc(docid, filtered)

        # print("Saving index file...")
        if os.path.exists('index-dir'):
            idx = 2
            while os.path.exists(f'index-dir{idx}'):
                idx += 1
            index.save(f'index-dir{idx}')
        else:
            index.save('index-dir')
        
        # print("Creating index complete!")
        return index


'''
The following class is a stub class with none of the essential methods implemented. It is merely here as an example.
'''


class SampleIndex(InvertedIndex):
    '''
    This class does nothing of value
    '''

    def add_doc(self, docid, tokens):
        """Tokenize a document and add term ID """
        for token in tokens:
            if token not in self.index:
                self.index[token] = {docid: 1}
            else:
                self.index[token][docid] = 1

    def save(self):
        print('Index saved!')
