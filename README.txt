Assignment -1

Team Members’ Names and Details:
Name: Jesly P Johnson
UID: UO1093321
Email: johnson.2123@wright.edu 

Name : Sandhya Bodige
UID : U01100050
Email: bodige.4@wright.edu 


Overview: This Python program utilizes the Cranfield Dataset to construct an information retrieval system. The system is specifically built to catalog and search through documents, offering valuable insights into vectorization methods and query processing algorithms.We have used the Scikit-learn library, a powerful open-source machine learning tool for Python, to create an efficient Indexer and Query Processor.

Application Launch:
1. Download the  Cransfield Dataset Zip folder and unzip it
2. Right click on the unzipped folder and select to open this folder on terminal 
4. Download VSCode and use it for coding
3. To Launch the application run the following commands in the terminal.

Python version -3.11.5 
Mac OS
	python3 -m venv .venv or python -m venv .venv
	source .venv/bin/activate
	pip install scikit-learn
	pip install matplotlib
Run the following to execute the output
	python assignment1.py

Windows OS
	python3 -m venv .venv or python -m venv .venv
	cmd
	source .venv\Scripts\activate
	pip install scikit-learn
	if prompted to upgrade your pip3 , upgarde accordingly
	pip install matplotlib
Run the following to execute the output
	python assignment1.py


External libraries :
import os
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import text as stoptext
import numpy as np
import pprint
import matplotlib.pyplot as plt
import matplotlib.style



Cranfield collection.
1398 abstracts (numbered 1 through 1400).
Aerodynamics.

Smallish collection, with large number of queries (225)

This directory contains:
-rw-rw-r--  1 chrisb   wheel        1446 Feb 14 18:30 README
        This file.
-rw-rw-r--  1 chrisb   wheel      592375 Feb 14 17:47 cran.all.Z
        Compressed version of document text.  Uncompressed version
        is 1644706 bytes
-rw-rw-r--  1 chrisb   wheel        6754 Feb 14 17:47 qrels.text.Z
        Relation giving relevance judgements.  Columns of file are
                query_id  doc_id   0    0
        to indicate doc_id is relevant to query_id.
        Uncompressed version:  21391
-rw-rw-r--  1 chrisb   wheel       12125 Feb 14 17:47 query.text.Z
        Text of queries.  Uncompressed: 28039
-rw-rw-r--  1 chrisb   wheel      559608 Feb 14 18:02 tf_doc.Z
        Indexed documents.  Columns of file are
             doc_id  0  concept_number  tf_weight  stemmed_word
        to indicate stemmed_word occurs in doc_id tf_weight times,
        and has been assigned the designator concept_number.
        Uncompressed: 2003724
-rw-rw-r--  1 chrisb   wheel       17226 Feb 14 17:50 tf_query.Z
        Indexed queries.   Columns of file are
             query_id  0  concept_number  tf_weight  stemmed_word
        to indicate stemmed_word occurs in query_id tf_weight times,
        and has been assigned the designator concept_number.
        Uncompressed: 28039

