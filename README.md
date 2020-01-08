## Classification-Based User Modeling Using Search Queries

### Dataset
This dataset is users' search query history within 1 month. It is collected by Sougou, a popular search engine in China.  
It has 5 columns and 100k+ rows. The categorical features has already been transformed into numerical data.
* ID: Encoded User ID
* Age: 0 (unknown) 1 (0-18) 2 (19-23) 3 (24-30) 4 (31-40) 5 (41-50) 
* Gender: 0 (unknown) 1 (Male) 2 (Female)
* Education: 0 (unknown) 1 (Phd) 2 (Master) 3 (Bachelor) 4 (High School) 5 (Junior High School and under)
* Search Query: A list of tuples of search queries

### Embedding / Chinese-Word-Vectors
To test the behaviors of different models in Chinese words, we tried JIEBA, NLPIR, THULC to predict a simple task on the dataset. JIEBA has the best performance in prediction. To generate our effective dictionary, we used Bigrams model and filtered those words of frequency less than 5. 

### Machine Learning Models
![Model Snapshot](https://github.com/caoyr03/search-query-user-portrait/blob/master/model%20snapshot.png)
We created a stacking ensemble model to take full advantage of predictions from base models. For the base layer of our model, we generated three classifiers, Logistic Regression using TF-IDF, Neural Network using DM, and Neural Network using DBOW. These combinations of word vectors and ML models perform relatively the highest accuracy on prediction task. After feeding the classfiers into stacking ensemble, we then run a XGBoost model as the final output. 
![Result Curve](https://github.com/caoyr03/search-query-user-portrait/blob/master/result%20curve.png)
The result curve shows that ensemble model performs the best compared with each of the three base models along.



### How to run
Training and testing dataset could be downloaded using the link 'https://drive.google.com/drive/folders/1Hhvy6d8OgTFonaF4orrzWVXAlO8NjHlF?usp=sharing'. Put these two files in './data' directory.  
Clone the repository into local. Run 'run.sh', this will generate 'tfidf_dm_dbow_20W.csv'

### Dependency
* Anaconda 4.2.0(Python 3.5 version)
* jieba 0.38
* keras 1.1.0
* xgboost 0.6
* gensim 0.13.2


