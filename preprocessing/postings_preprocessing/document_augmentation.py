from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class DocSummaryGenerator:  
    def __init__(self, model_name: str = "facebook/bart-large-cnn") -> None:
        # self.device = torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    def get_summary(self, document: str, prefix_prompt: str = '') -> list[str]:
        if document == "":
            return ""
        
        input_ids = self.tokenizer.encode(prefix_prompt+document, max_length=500, truncation=True, return_tensors='pt')
        outputs = self.model.generate(input_ids=input_ids, max_length=150, do_sample=True)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import jsonlines
    import time
    import os
    import sys
    import kagglehub
    
    from sentence_transformers import SentenceTransformer, util
    
    if len(sys.argv) < 2: FRACTION = int(input("Enter fraction number: "))
    else: FRACTION = int(sys.argv[1])
    EMBEDDING_NPY_PATH = "augmented_description_embeddings.all-MiniLM-L6-v2.npy"
    DOCIDS_PATH = "job_posting_ids.txt"
    AUG_DOCIDS_PATH = "augment_docids.jsonl"

    # print('loading augmentation docids...')
    with jsonlines.open(AUG_DOCIDS_PATH) as f:
        for i, doc in enumerate(f):
            if i == FRACTION:
                aug_docids = doc["docid"]
    print("processing set", FRACTION, "with", len(aug_docids), "documents.\n")
    
    print('loading dataset...')
    path = kagglehub.dataset_download("arshkon/linkedin-job-postings")
    data = pd.read_csv(path + "/postings.csv")
    data = data[data['description'].notna()]
    
    print('loading augmentaion model...')
    aug_model = DocSummaryGenerator("facebook/bart-large-cnn")

    print("loading document ids...")
    docids = []
    with open(DOCIDS_PATH) as f:
        for line in f:
            docids.append(int(line))
    
    print('creating augmented documents...')
    start = time.time()
    for i in tqdm(range(len(data))):
        if data.iloc[i]['job_id'] in aug_docids:
            summary = aug_model.get_summary(data.iloc[i]['description'])
            data.loc[i, 'description'] = summary + " " + data.iloc[i]['description']
    end = time.time()
    print(" - finished after", end - start, "s.")
    
    print("initializing embedding models...")
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_model = SentenceTransformer(model_name)
    
    print("loading embeddings...")
    embeddings = np.load(EMBEDDING_NPY_PATH)
    embeddings = embeddings.tolist()
    
    print("creating document embeddings...")
    aug_embeddings = []
    for i, docid in tqdm(enumerate(docids), total=len(docids)):
        if docid in aug_docids:
            aug_embeddings.append(embedding_model.encode(data.iloc[i]['description']).reshape(1, -1)[0])
        else:
            aug_embeddings.append(embeddings[i])
    
    #aug_embeddings = np.squeeze(np.array(aug_embeddings), axis=1)
    #print(np.array(aug_embeddings))
    np.save(EMBEDDING_NPY_PATH, aug_embeddings)
    print("done!")
