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
Hits@1: 0.26340454858718126  
Hits@5: 0.6096485182632667  
Hits@10: 0.7508614748449345  
Hits@15: 0.8152308752584424  
----------NDCG@K----------  
NDCG@1: 0.26340454858718126  
NDCG@5: 0.4433683759379118  
NDCG@10: 0.4891383027559529  
NDCG@15: 0.5061984949125046  

2. EMCDR model on target domain:  
----------Hits@K----------  
Hits@1: 0.28035837353549276  
Hits@5: 0.6167470709855272  
Hits@10: 0.7483115093039283  
Hits@15: 0.8099241902136458  
----------NDCG@K----------  
NDCG@1: 0.28035837353549276  
NDCG@5: 0.45665275794534727  
NDCG@10: 0.49940808474809945  
NDCG@15: 0.5157231433783096  


