from datasets import load_dataset
import numpy as np
from model import BM25, ColSentenceModel, SiglipStyleModel, MentorModel
import os
import torch
import json
import math
import itertools
import torch
# from project import HTML_FILE
from tqdm import tqdm
from bs4 import BeautifulSoup
import random

HTML_FILE = "indexed_html.jsonl"
def generate_queries():
    base_queries = [
        # (shortened for brevity, but in your code, use the full list of 500 queries)
        "best restaurants in tübingen",
        "tübingen university ranking",
        "things to do in tübingen",
        "tübingen tourist attractions",
        "tübingen old town walking tour",
        "how to get to tübingen from stuttgart",
        "tübingen castle opening hours",
        "tübingen christmas market dates",
        "public transport in tübingen",
        "tübingen university admission requirements",
        "tübingen nightlife guide",
        "tübingen bike rental",
        "tübingen weather forecast",
        "tübingen student housing options",
        "tübingen museums",
        "tübingen river cruise",
        "tübingen local events",
        "tübingen public library opening hours",
        "tübingen vegan restaurants",
        "tübingen best coffee shops",
        "tübingen art galleries",
        "tübingen hiking trails",
        "tübingen city map",
        "tübingen parking information",
        "tübingen shopping streets",
        "tübingen local markets",
        "tübingen swimming pools",
        "tübingen gyms",
        "tübingen yoga studios",
        "tübingen language schools",
        "tübingen job opportunities",
        "tübingen internships",
        "tübingen volunteer work",
        "tübingen expat community",
        "tübingen international schools",
        "tübingen kindergartens",
        "tübingen pet-friendly hotels",
        "tübingen taxi services",
        "tübingen car sharing",
        "tübingen electric car charging stations",
        "tübingen recycling centers",
        "tübingen waste collection schedule",
        "tübingen emergency numbers",
        "tübingen hospitals",
        "tübingen pharmacies",
        "tübingen dentists",
        "tübingen doctors",
        "tübingen covid testing centers",
        "tübingen vaccination centers",
        "tübingen city hall address",
        "tübingen post offices",
        "tübingen banks",
        "tübingen atm locations",
        "tübingen currency exchange",
        "tübingen mobile phone shops",
        "tübingen internet providers",
        "tübingen movie theaters",
        "tübingen concerts",
        "tübingen theater performances",
        "tübingen festivals",
        "tübingen flea markets",
        "tübingen second hand shops",
        "tübingen bookstores",
        "tübingen souvenir shops",
        "tübingen bakeries",
        "tübingen butchers",
        "tübingen supermarkets",
        "tübingen organic food stores",
        "tübingen wine bars",
        "tübingen breweries",
        "tübingen cocktail bars",
        "tübingen live music venues",
        "tübingen escape rooms",
        "tübingen bowling alleys",
        "tübingen playgrounds",
        "tübingen dog parks",
        "tübingen picnic spots",
        "tübingen scenic viewpoints",
        "tübingen photography spots",
        "tübingen walking tours",
        "tübingen guided tours",
        "tübingen boat tours",
        "tübingen canoe rental",
        "tübingen stand up paddling",
        "tübingen fishing spots",
        "tübingen bird watching",
        "tübingen botanical gardens",
        "tübingen parks",
        "tübingen skate parks",
        "tübingen tennis courts",
        "tübingen football fields",
        "tübingen basketball courts",
        "tübingen climbing gyms",
        "tübingen martial arts schools",
        "tübingen dance studios",
        "tübingen music schools",
        "tübingen art classes",
        "tübingen cooking classes",
        "tübingen pottery workshops",
        "tübingen photography courses",
        "tübingen language exchange",
        "tübingen meetup groups",
        "tübingen dating spots",
        "tübingen romantic restaurants",
        "tübingen wedding venues",
        "tübingen conference centers",
        "tübingen coworking spaces",
        "tübingen startup scene",
        "tübingen business networking",
        "tübingen technology events",
        "tübingen science events",
        "tübingen university open days",
        "tübingen student discounts",
        "tübingen alumni events",
        "tübingen graduation ceremonies",
        "tübingen research institutes",
        "tübingen scholarships",
        "tübingen student jobs",
        "tübingen student clubs",
        "tübingen student unions",
        "tübingen student health insurance",
        "tübingen student visa",
        "tübingen erasmus program",
        "tübingen international office",
        "tübingen buddy program",
        "tübingen student orientation",
        "tübingen campus map",
        "tübingen dormitories",
        "tübingen private apartments",
        "tübingen flat share",
        "tübingen housing market",
        "tübingen rent prices",
        "tübingen moving services",
        "tübingen furniture stores",
        "tübingen home improvement",
        "tübingen cleaning services",
        "tübingen laundry services",
        "tübingen tailoring",
        "tübingen shoe repair",
        "tübingen key cutting",
        "tübingen locksmiths",
        "tübingen electricians",
        "tübingen plumbers",
        "tübingen handymen",
        "tübingen gardening services",
        "tübingen car repair",
        "tübingen bike repair",
        "tübingen gas stations",
        "tübingen car wash",
        "tübingen driving schools",
        "tübingen traffic rules",
        "tübingen parking garages",
        "tübingen park and ride",
        "tübingen bus schedule",
        "tübingen train station",
        "tübingen train timetable",
        "tübingen regional trains",
        "tübingen long distance buses",
        "tübingen airport shuttle",
        "tübingen nearest airport",
        "tübingen taxi fare",
        "tübingen car rental",
        "tübingen bike paths",
        "tübingen walking routes",
        "tübingen city tours",
        "tübingen sightseeing",
        "tübingen historical sites",
        "tübingen monuments",
        "tübingen churches",
        "tübingen synagogues",
        "tübingen mosques",
        "tübingen cemeteries",
        "tübingen war memorials",
        "tübingen city history",
        "tübingen famous people",
        "tübingen local legends",
        "tübingen traditions",
        "tübingen dialect",
        "tübingen cuisine",
        "tübingen local dishes",
        "tübingen food delivery",
        "tübingen takeout",
        "tübingen catering",
        "tübingen food trucks",
        "tübingen street food",
        "tübingen ice cream shops",
        "tübingen chocolate shops",
        "tübingen tea houses",
        "tübingen breakfast places",
        "tübingen brunch spots",
        "tübingen lunch deals",
        "tübingen dinner specials",
        "tübingen happy hour",
        "tübingen late night food",
        "tübingen gluten free restaurants",
        "tübingen lactose free options",
        "tübingen allergy friendly restaurants",
        "tübingen kids activities",
        "tübingen family attractions",
        "tübingen senior activities",
        "tübingen accessible places",
        "tübingen wheelchair friendly",
        "tübingen dog friendly cafes",
        "tübingen pet stores",
        "tübingen veterinarians",
        "tübingen animal shelters",
        "tübingen adoption centers",
        "tübingen lost and found",
        "tübingen police station",
        "tübingen fire department",
        "tübingen city council",
        "tübingen local news",
        "tübingen weather warnings",
        "tübingen flood risk",
        "tübingen air quality",
        "tübingen noise pollution",
        "tübingen green spaces",
        "tübingen sustainability projects",
        "tübingen renewable energy",
        "tübingen climate initiatives",
        "tübingen recycling tips",
        "tübingen composting",
        "tübingen community gardens",
        "tübingen farmers markets",
        "tübingen local produce",
        "tübingen organic farms",
        "tübingen wine tasting",
        "tübingen brewery tours",
        "tübingen culinary tours",
        "tübingen food festivals",
        "tübingen art festivals",
        "tübingen music festivals",
        "tübingen film festivals",
        "tübingen literature events",
        "tübingen science fairs",
        "tübingen technology fairs",
        "tübingen job fairs",
        "tübingen university fairs",
        "tübingen open air events",
        "tübingen christmas market",
        "tübingen easter market",
        "tübingen spring festival",
        "tübingen summer festival",
        "tübingen autumn festival",
        "tübingen winter festival"
    ]
    queries = []
    for q in base_queries:
        # Decide if this query should reference the city name (50% of the time)
        if random.random() < 0.5:
            # Remove "tübingen" or "tuebingen" from the query
            q_no_city = q.replace("tübingen ", "").replace("tuebingen ", "")
            q_no_city = q_no_city.replace(" in tübingen", "").replace(" in tuebingen", "")
            q_no_city = q_no_city.replace(" near tübingen", "").replace(" near tuebingen", "")
            queries.append(q_no_city.strip())
        else:
            # 40% of all queries (i.e., 80% of those with city name) should use "tuebingen"
            if random.random() < 0.4:
                q_city = q.replace("tübingen", "tuebingen")
                queries.append(q_city)
            else:
                queries.append(q)
    return queries

QUERIES = generate_queries()

class Ranking_Benchmark:
    def __init__(self, dataset_name, dir_name, prefix="", result_path="datasets/preprocessed/", max_samples=100, dataset = None, benchmark_model="BM25"):
        if dataset is None:
            dataset = load_dataset(dataset_name, dir_name)
            dataset = dataset["test"][:max_samples]
            self.queries = dict(zip(dataset["query"], dataset["query_id"]))
            self.documents, _ = self.preprocess(dataset, prefix, result_path)
        else:
            self.queries = dataset["queries"]
            self.documents = dataset["documents"]

        if benchmark_model=="BM25":
            self.model = BM25()
            self.model.preprocess(self.documents) # computes bow components ahead of time
        else:
            self.model = MentorModel()

        self.per_query_rankings, self.avg_rel_rank = self.get_per_query_rankings()

    def preprocess(self, dataset, prefix="", result_path="datasets/preprocessed/"):
        # assign object var here, so other methods can access it without blowing up the passed parameters. This is not clean. Fix later.
        self.prefix = prefix
        # assign object var here, so other methods can access it without blowing up the passed parameters. This is not clean. Fix later.
        self.result_path = result_path
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
            relevance_assignments = self.model.calculate_rels(query)
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
        if ideal_dcg == 0:
            return 0
        else:
            return rank_dcg/ideal_dcg
    
    def benchmark(self, model): # Needs some kind of batching. cant hold a significant amount of embeddings in memory.
        doc_embeddings = model.embed(self.documents) # assumes no shuffelling happens here
        q_embedding = model.embed(list(self.queries.keys()))
        ndcg_scores = []
        for query_idx in range(len(self.queries.keys())):
            doc_relevancies = model.resolve(q_embedding[query_idx], doc_embeddings).flatten().detach().cpu().numpy() # we know flattening is ok, because this is of shape (queries, docs) where queries is 1
            doc_sort_idx = np.flip(np.argsort(doc_relevancies))[:100] # compare top 100 results
            local_ndcg = self.ndcg(query=list(self.queries.keys())[query_idx], ranking=doc_sort_idx)
            ndcg_scores.append(local_ndcg)
        avg_ndcg = sum(ndcg_scores)/len(ndcg_scores)
        return avg_ndcg

def load_data(path: str = "data"):
    html_path = os.path.join(path, HTML_FILE)
    if not os.path.exists(html_path):
        raise FileNotFoundError(f"HTML file not found at {html_path}")

    docs = {}
    with open(html_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, "Line"): docs.update(json.loads(line.strip()))
    return docs

def preprocess_html(html: str, seperator: str = ' ') -> str:
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator=seperator, strip=True)
    return text.strip()

def generate_dset(data_dict: dict, samples=10000) -> list:
    documents = []
    items = list(data_dict.items())
    for i in tqdm(range(min(len(data_dict),samples)), "Embedding"):
        html = preprocess_html(items[i][1], ". ")[:10000]
        documents.append(html)
    return documents

docs = load_data(path="data")

queries = dict(zip(QUERIES, list(range(len(QUERIES)))))
documents = generate_dset(docs)

dataset = {"queries": queries, "documents":documents}

model = ColSentenceModel()
# model_path = "./clip/ColSent/bert-mini/b64_lr1E-06_microsoft/ms_marcov2.1/"
# model_name = "model.safetensors"
model_path = "clip/ColSent/bert-mini/b64_lr1E-06_microsoft/ms_marcov2.1/"
model_name = "model.safetensors"
model.load(model_path+model_name)
model.data_path=model_path+"embed_data/"
# model.use_max_sim = False
benchmark = Ranking_Benchmark("microsoft/ms_marco", "v2.1", "[rank]", model_path, dataset=dataset, benchmark_model="mentor")
print(benchmark.benchmark(model))
model.use_max_sim = False
print(benchmark.benchmark(model))