# News-popularity-analysis

## Usage

- ```python3 main.py``` Enter urls
- ```python3 main.py < urls.txt``` txt file with urls on each line end with empty

You can also provide pre-downloaded htmls to the preprocess file class ```Article``` directly.


## Method

1. The Article class gets the url content and searches for article ```Title, Keywords, Description, Author, Published date and content```. Everything except the content is really easy to get.

2. The content is then processed to extract features listed in this [paper](http://cs229.stanford.edu/proj2015/328_report.pdf)

3. Readability scores are obtained from content.

4. LDA based on the latest popular topics from the [News API](https://newsapi.org/). 
    - The LDA model can be updated by running ```python3 update_headlines.py```.
    - You will need a news api key, just store it in a file named ```newsapi.key```

5. All this is then fed into the XGBClassifier for predictions. Can also add regression models but it won't port as easily because of the outdated dataset.
 
 ## TODO
 
 - Using [UMAP](https://umap-learn.readthedocs.io/en/latest/) reduced [InferSent](https://github.com/facebookresearch/InferSent) vectors for titles improved performance about 3-4%. Adding it to the test pipeline was slow, so postponed for now.
 - Use GDELT for trending topics, will improve complexity but with weekly update frequency its feasible
 
 ## References
 > Will update later
