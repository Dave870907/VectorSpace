# WSM Project 1: Ranking by Vector Space Models

## How to use:
    $ python main.py --query {query}
## example:
    $ python main.py --query="Trump Biden Taiwan China"

    After the query,there will be five outputs:
* Term Frequency (TF) Weighting + Cosine Similarity
* Term Frequency (TF) Weighting + Euclidean Distance
* TF-IDF Weighting + Cosine Similarity
* TF-IDF Weighting + Euclidean Distance
* Relevance Feedback + TF-IDF Weighting + Cosine Similarity
