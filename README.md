# Modern-Search-Engines-Projekt
## Basic Documentation

Install requirements with:
```
pip install -r requirements.txt
```

The data for the search engine has to be in a `data` folder. Copy the data from e.g. [`final/cutoff`](https://github.com/Cari1111/Modern-Search-Engines-Lecture/tree/extra/final_data/cutoff) into a `data` folder which should look like:

```
Modern-Search-Engines-Lecture
└── data
    ├── bm25_state.json
    ├── clustering_embeddings.pkl
    ├── embeddings.pkl
    └── indexed_docs.jsonl
```

Start web app with (arguments are optional):

```
python -m project [--host 127.0.0.1] [--port 8080] [--no_logging]
```

## Important files and folders

- [`Group Project Rules.ipynb`](https://github.com/Cari1111/Modern-Search-Engines-Lecture/blob/main/Group%20Project%20Rules.ipynb): Notebook with cells for crawler, indexing, calculating embeddings, creating evaluation txt
- [`Crawler.ipynb`](https://github.com/Cari1111/Modern-Search-Engines-Lecture/blob/main/Crawler.ipynb): Notebook for the crawler and crawler debugging
- [`Clustering.ipynb`](https://github.com/Cari1111/Modern-Search-Engines-Lecture/blob/main/Clustering.ipynb): Notebook for testing diffrent clustering methodes to identify the topics
- [`project/search.py`](https://github.com/Cari1111/Modern-Search-Engines-Lecture/blob/main/project/search.py): Main `SearchEngine` class that loads all embeddings and can execure searches and clustering
- [`project/crawler`](https://github.com/Cari1111/Modern-Search-Engines-Lecture/tree/main/project/crawler): Folder for all crawler files. Implements the crawler  in `crawler.py`
- [`project/retriever`](https://github.com/Cari1111/Modern-Search-Engines-Lecture/tree/main/project/retriever): Folder for all retriever files. Implements the models `SiglipStyleModel` and `ColSentenceModel` in `model.py`
- [`project/frontend`](https://github.com/Cari1111/Modern-Search-Engines-Lecture/tree/main/project/frontend): Folder for all fronted files. The frontend uses the `SearchEngine` class to connect to the backend in `page.py`

## Organisatorisches
- Branch Benamung: Erster Buchstabe des Eigenen namens + branch namen
- Requirements: nutzung einer zentralen requirements.txt, die auf Anaconda ausgeleg ist

## Themen
- Crawler
  - Kilian, Simon
- Ranker
  - Simon, Jan
- Presentation
  - Kilian, Carina, Martin
- Creative
  - Carina, Jan, Martin

