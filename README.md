# *WSM Project 1: Ranking by Vector Space Models*

## How to use:
    $ python main.py --query {query}
## example:
    $ python main.py --query="Trump Biden Taiwan China"

  **After the query,there will be five outputs:**
* Term Frequency (TF) Weighting + Cosine Similarity
* Term Frequency (TF) Weighting + Euclidean Distance
* TF-IDF Weighting + Cosine Similarity
* TF-IDF Weighting + Euclidean Distance
* Relevance Feedback + TF-IDF Weighting + Cosine Similarity

## Files:
* **EnglishNews:** 

    collection of english news
    
* **English.stop:** 

    collection of english stop words
    
* **main.py:**
 
    main execution file

* **Parser.py:** 

    clean,remove stop words,tokenise the documents

* **PorterStemmer.py:**

    the Porter stemming algorithm, ported to Python from the version coded up in ANSI C by the author.

* **util.py:**

    utilities like tf, idf weighting, and cosine similarity and distance function
    
* **VectorSpace.py:**

    class of vectorSpaceModel
    
## Output:
![截圖 2021-04-05 下午9 53 18](https://user-images.githubusercontent.com/44217401/113581165-5a873c00-9659-11eb-9bb7-aafc72c71813.png)
