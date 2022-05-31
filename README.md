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


