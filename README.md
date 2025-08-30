# Job Searching System

## Introduction

Job seeking is a tedious and time-consuming process. Candidates often spend countless hours browsing through company websites, comparing their skills to job requirements, and checking whether a position aligns with their career goals.

In our project, we want to exploit the advantages of IR system. The users can directly input their resume and find the job postings, so that the system can consider all the information a user wants to include.

## How to Use

### Data

All data used in this repo is stored in `data.zip`. To run the program, unzip `data.zip` and put `\data` in root directory.

### Run with Modules

First, create a SearchEngine object.

```
search_obj = SearchEngine(ranker="VectorRanker", l2r=True)
```

**PDF mode**

You can call `search_pdf()` with resume file directory. The function will return a list of relevant postings.

```
results = search_obj.search_pdf("data/resume_sample_IT_10089434.pdf")
```

**Text mode**

Call `search()` to search with plain text. The function will also return a list of relevant postings.

```
results = search_obj.search(query)
```

**Print results**

You can call `printSearchResponse()` to print out the top 5 relevant postings in a more readable format.

```
raw_data = pd.read_csv(POSTINGS_CATEGORY_PATH)
printSearchResponse(raw_data, results)
```

### Interactive Mode

We allow iteratively searching for multiple queries in an interactive mode. Simply call `interactive()` and the system will initialize and start querying.

Firstly, you can choose what type of resume data you want to use.

**PDF mode**

In PDF mode, you can input the relative directory of your resume file.

```
Choose your mode (pdf/text):  pdf
Enter your resume file path:  data/resume_sample_IT_10089434.pdf
```

**Text mode**

In text mode, you can directly input your whole resume.

```
Choose your mode (pdf/text):  text
Enter your resume text: [resume_text]
```

The program will start retrieving documents right after you tap ENTER. It will print the top 5 relevant postings once it finishes retrieving. Based on this result, you can choose whether or not to give feedback to the postings. If you choose yes, you can type a list of 1's and 0's in order, where 1 indicates relevant and 0 indicates irrelevant. After providing the feedback, the system will generate more relevant results in the next round.

```
Provide feedback to get more relevant results? (Y/N):  Y
Input feedback in order (Relevant = 1, Non-Relevant = 0) 11100
```

You can exit the program after a query is completed.

```
End the program? (Y/N):  N
```

For the complete sample code, see `Interactive.ipynb`.

> [IMPORTANT!] Since our feature space is relatively large, it takes a long time to generate the training data (almost 6 hours!). We saved the training data for L2R model with VectorRanker and document augmentation in a separate file `l2r_training_data.zip`. You can unzip this file and save this in your root directory. This will help you save time~

## Acknowledgement

This project was developed as part of SI650: Information Retrieval at the University of Michigan.

Please note that this project is subject to the University Honor Code Policy. If you use this code or reference this work, please cite the original paper appropriately.
