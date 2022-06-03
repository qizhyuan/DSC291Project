# DSC291Project

## Dataset Generation
### Source Domain - MovieLens
The data is available at https://grouplens.org/datasets/movielens/. We use ml-latest.zip to construct our cross-domain source data, which can be download at https://files.grouplens.org/datasets/movielens/ml-latest.zip. It contains 27,000,000 ratings and 1,100,000 tag applications applied to 58,000 movies by 280,000 users.

### Target Domain - Netflix Prize Data
The data is available at https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data, which contains 17770 movies and 480189 users.

The generated data is provided in dataset.rar file. The notebook dataset_generator_ml_nf.ipynb contains the code to generate the data for cross-domain recommendation. 

The generated dataset has three csv files, and each file has the following format:
user, item, rating

Since our task is to predict the interactions, we can simply ignore the rating data (treat them as positive instances).

## Dataset Statistics
### Target Domain:  
Train:  
User Num: 29,818    
Item Num: 4,799   
Rating Num: 227,355    
Average User Degree: 7.62    
Average Item Degree: 47.37    

Test:  
Num: 15152    
  
### Source Domain:  
User Num: 1,2481  
Item Num: 4,799  
Rating Num: 737,822  
Average User Degree: 59.11  
Average Item Degree: 153.74  

## Setup
### Dependencies
- Numpy
- PyTorch
- scikit-learn

### Start
Unzip the file dataset.rar at first. Then, use the following command to perform MF and EMCDR models.

    python main.py

The experimental results will be saved in the file result.json in the current dictionary.

## Experimental Result
### MovieLens (S) -> Netflix Prize Data (T)
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
hit@1 : 0.018083421330517425  
hit@5 : 0.20142555438225976  
hit@10 : 0.42245248152059134  
hit@15 : 0.5837513199577613  


