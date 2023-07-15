import os
import nltk
import contextlib
from sentence_transformers import SentenceTransformer, util
import numpy as np
from LexRank import degree_centrality_scores

# You must also bring the latest version of LexRank into your local directory, I had issues.

document = open("document-name.txt", "r").read()
print(document)

model = SentenceTransformer('all-mpnet-base-v2')

sentences = nltk.sent_tokenize(document)
print("Num sentences:", len(sentences))

embeddings = model.encode(sentences, convert_to_tensor=True)

cos_scores = util.cos_sim(embeddings, embeddings).numpy()

centrality_scores = degree_centrality_scores(cos_scores, threshold=None)

most_central_sentence_indices = np.argsort(-centrality_scores)

file_path = '/content/summaries/summary.txt'
with open(file_path, "w") as o:
    with contextlib.redirect_stdout(o):
      for idx in most_central_sentence_indices[0:20]: # set summary sentnece qty here
        print(sentences[idx])