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
|        Model       | Hits@1 | Hits@5 | Hits@10 | Hits@15 |
|:------------------:|:------:|:------:|:-------:|:-------:|
|     Popularity     | 0.0487 | 0.2866 |  0.5560 |  0.7292 |
| Jaccard Similarity | 0.0419 | 0.3944 |  0.6954 |  0.8330 |
|  Cosine Similarity | 0.0180 | 0.2014 |  0.4224 |  0.5837 |
|         MF         | 0.1197 | 0.4634 |  0.6713 |  0.7748 |
|      MF-Mixed      | 0.1209 | 0.4362 |  0.6398 |  0.7503 |
|        EMCDR       | 0.1570 | 0.4854 |  0.6463 |  0.7259 |




