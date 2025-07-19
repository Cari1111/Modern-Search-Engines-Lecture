from datasets import load_dataset
import numpy as np
from model import BM25, ColSentenceModel
import os
import torch
import json
import math
import itertools

class Ranking_Benchmark:
    def __init__(self, dataset_name, dir_name, prefix="", result_path="datasets/preprocessed/", max_samples=10):
        dataset = load_dataset(dataset_name, dir_name)
        dataset = dataset["test"][:max_samples]
        self.bm25 = BM25()
        self.queries = dict(zip(dataset["query"], dataset["query_id"]))

        self.documents, self.relevance_map = self.preprocess(dataset, prefix, result_path)
        self.bm25.preprocess(self.documents) # computes bow components ahead of time

        self.per_query_rankings, self.avg_rel_rank = self.get_per_query_rankings()

    def preprocess(self, dataset, prefix="", result_path="datasets/preprocessed/"):
        self.prefix = prefix # assign object var here, so other methods can access it without blowing up the passed parameters. This is not clean. Fix later.
        self.result_path = result_path # assign object var here, so other methods can access it without blowing up the passed parameters. This is not clean. Fix later.
        result_dir_contents = os.listdir(result_path)
        
        if not any(map(lambda x: f"{self.prefix}passages.json"==x, result_dir_contents)): # passages json is not already present. compute from scratch.
            passages, relevance_tensor = self.extract_passages_and_relevance(dataset)

            torch.save(relevance_tensor, f"{self.result_path}{self.prefix}relevance_assignments.pt")
        
            # save list of passages (NOTE: we treat passage ids implicitly by referencing the index in this list. DO NOT SHUFFLE.)
            with open(f"{self.result_path}{self.prefix}passages.json", "w") as f:
                json.dump(passages, f)
        else:
            with open(f"{self.result_path}{self.prefix}passages.json", "r") as f:
                passages = json.load(f)
            relevance_tensor = torch.load(f"{self.result_path}{self.prefix}relevance_assignments.pt")

        return passages, relevance_tensor

    def extract_passages_and_relevance(self, dataset): # slow for loop implementation. But not worth the effort to optimize rn. fix if nessecary
        passages = list(list(itertools.chain.from_iterable([ex["passage_text"] for ex in dataset["passages"]]))) # extracts list of all text at once. Probably very big
        relevance_assignments = torch.Tensor(list(itertools.chain.from_iterable([ex["is_selected"] for ex in dataset["passages"]])))
        query_ids = torch.Tensor(list(itertools.chain.from_iterable([[dataset["query_id"][idx]]*len(dataset["passages"][idx]["passage_text"]) for idx in range(len(dataset["query_id"]))])))

        passage_ids = torch.arange(0, len(passages))
        relevance_assignments = torch.stack((query_ids, passage_ids, relevance_assignments), dim=1)

        return passages, relevance_assignments
    
    def get_per_query_rankings(self):
        relevance_rankings = {}
        #relevant_ranks = []
        for query in self.queries.keys(): # slow
            relevance_assignments = self.bm25.calculate_rels(query)
            sorted_relevance_idx = np.flip(np.argsort(relevance_assignments))
            #sorted_relevance_assignments = relevance_assignments[sorted_relevance_idx]
            relevance_rankings[query] = (relevance_assignments, sorted_relevance_idx)
            #for relevant, q_idx, doc_idx in self.relevance_map:  # slow
            #    if relevant == 1 and q_idx == self.queries[query]:
            #        relevant_ranks.append(np.argwhere(sorted_relevance_assignments[:,1]==doc_idx)[0]) # find first occurence of doc idx of document marked as relevant. Sanity check of BM25 ranking

        avg_rel_rank = 0 # sum(relevant_ranks)/len(relevant_ranks) # NO RELEVANCE LABELS IN THE TEST DATASET ANYWAYS?
        return relevance_rankings, avg_rel_rank
    
    def dcg(self, query, ranking):
        discount_factors = np.vectorize(lambda x: 1/math.log2(x+1))(np.arange(1, len(ranking)+1))
        return np.sum(self.per_query_rankings[query][0][ranking]*discount_factors)

    def ndcg(self, query, ranking):
        rank_dcg = self.dcg(query, ranking)
        ideal_dcg = self.dcg(query, self.per_query_rankings[query][1][:len(ranking)])
        return rank_dcg/ideal_dcg
    
    def benchmark(self, model): # Needs some kind of batching. cant hold a significant amount of embeddings in memory.
        doc_embeddings = model.embed(self.documents, query=False) # assumes no shuffelling happens here
        q_embedding = model.embed(list(self.queries.keys()))
        ndcg_scores = []
        for query_idx in range(len(self.queries.keys())):
            doc_relevancies = model.resolve(q_embedding[query_idx], doc_embeddings).flatten().detach().cpu().numpy() # we know flattening is ok, because this is of shape (queries, docs) where queries is 1
            doc_sort_idx = np.argsort(doc_relevancies)
            local_ndcg = self.ndcg(query=list(self.queries.keys())[query_idx], ranking=doc_sort_idx)
            ndcg_scores.append(local_ndcg)
        avg_ndcg = sum(ndcg_scores)/len(ndcg_scores)
        return avg_ndcg


model = ColSentenceModel()
model_path = "./clip/ColSent/bert-mini/b64_lr1E-06_microsoft/ms_marcov2.1/"
model_name = "model.safetensors"
model.load(model_path+model_name)
model.data_path=model_path+"embed_data/"
benchmark = Ranking_Benchmark("microsoft/ms_marco", "v2.1", "[rank]", model_path)
print(benchmark.benchmark(model))