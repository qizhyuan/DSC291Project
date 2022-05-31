# DSC291Project

## Dataset Generation
### Source Domain - MovieLens
The data is available at https://grouplens.org/datasets/movielens/. We use ml-latest.zip to construct our cross-domain source data, which can be download at https://files.grouplens.org/datasets/movielens/ml-latest.zip. It contains 27,000,000 ratings and 1,100,000 tag applications applied to 58,000 movies by 280,000 users.

### Target Domain - Netflix Prize data
The data is available at https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data, which contains 17770 movies and 480189 users.

The generated data is provided in dataset.rar file. The notebook dataset_generator_ml_nf.ipynb contains the code to generate the data for cross-domain recommendation. 

The generated dataset has three csv files, and each file has the following format:
user, item, rating

Since our task is to predict the interactions, we can simply ignore the rating data (treat them as positive instances).

## Dataset Statistics
### Target Domain:  
Train:
User Num: 10660  
Item Num: 3589  
Rating Num: 19334  
Average User Degree: 1.81  
Average Item Degree: 5.38  

Test:
Num: 5019  
  
### Source Domain:  
User Num: 18305  
Item Num: 3589  
Rating Num: 767582  
Average User Degree: 41.93  
Average Item Degree: 213.87 

## Experimental Result
1. MF model on target domain:  
----------Hits@K----------  
Hits@1: 0.03207810320781032  
Hits@5: 0.13668061366806136  
Hits@10: 0.2430763100219167  
Hits@15: 0.3486750348675035  
----------NDCG@K----------  
NDCG@1: 0.03207810320781032  
NDCG@5: 0.08320132097460295  
NDCG@10: 0.1170858422246534  
NDCG@15: 0.1449333807804888     

2. EMCDR model on target domain:  
----------Hits@K----------  
Hits@1: 0.09065550906555091  
Hits@5: 0.2612074118350269  
Hits@10: 0.3703925084678223  
Hits@15: 0.4504881450488145  
----------NDCG@K----------  
NDCG@1: 0.09065550906555091  
NDCG@5: 0.17741692709534743  
NDCG@10: 0.2125777696080721  
NDCG@15: 0.23377164806322093  

3. Jaccard model:     
hit@1 : 0.755130504084479  
hit@5 : 0.9996015142458657  
hit@10 : 1.0  
hit@15 : 1.0  

4. Popularity model: 
hit@1 : 0.02171747360031879  
hit@5 : 0.10958358238692967  
hit@10 : 0.21797170751145648  
hit@15 : 0.3189878461844989  

5. Cosine-similarity model: 
hit@1 : 0.07352062163777645  
hit@5 : 0.1528192867105001  
hit@10 : 0.19884439131301057  
hit@15 : 0.2442717672843196  


## New Data
1. MF model:  
----------Hits@K----------  
Hits@1: 0.11582629355860613  
Hits@5: 0.4593453009503696  
Hits@10: 0.6632787750791974  
Hits@15: 0.778247096092925  
----------NDCG@K----------  
NDCG@1: 0.11582629355860613  
NDCG@5: 0.2888443365470047  
NDCG@10: 0.35485398007136854  
NDCG@15: 0.3853008484016402  

2. EMCDR model:  
----------Hits@K----------  
Hits@1: 0.1463833157338965  
Hits@5: 0.44324181626187964  
Hits@10: 0.5984688489968321  
Hits@15: 0.6793822597676874  
----------NDCG@K----------  
NDCG@1: 0.1463833157338965  
NDCG@5: 0.29809340544491375  
NDCG@10: 0.34844307231052357  
NDCG@15: 0.36990617303729073  

3. Jaccard model:    
hit@1 : 0.04197465681098205  
hit@5 : 0.3944693769799366  
hit@10 : 0.6954197465681098  
hit@15 : 0.833025343189018  

4. Popularity model: 
hit@1 : 0.048706441393875394  
hit@5 : 0.28662882787750793  
hit@10 : 0.5560982048574445  
hit@15 : 0.7292106652587117  

5. Cosine-similarity model: 
hit@1 : 0.0
hit@5 : 0.0
hit@10 : 0.0
hit@15 : 6.599788806758183e-05


