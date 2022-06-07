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
Note that this branch does not contain the SOTA methods. For SOTA method, please swith to the sota branch.


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


|   Model  | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@15 |
|:--------:|:------:|:------:|:-------:|:-------:|
|    MF    | 0.1197 | 0.2932 |  0.3607 |  0.3881 |
| MF-Mixed | 0.1209 | 0.2793 |  0.3453 |  0.3746 |
|   EMCDR  | 0.1570 | 0.3253 |  0.3775 |  0.3986 |

### Netflix Prize Data (T) -> MovieLens (S) [Switch the Domain]
|        Model       | Hits@1 | Hits@5 | Hits@10 | Hits@15 |
|:------------------:|:------:|:------:|:-------:|:-------:|
|    MF     | 0.3762 | 0.6911 |  0.8154 |  0.8745 |
| MF-Mixed  | 0.3378 | 0.6689 |  0.8051 |  0.8676 |
|   EMCDR   | 0.2781 | 0.5822 |  0.7194 |  0.7909 |


|   Model  | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@15 |
|:--------:|:------:|:------:|:-------:|:-------:|
|    MF    | 0.3762 | 0.5424 |  0.5828 |  0.5985 |
| MF-Mixed | 0.3378 | 0.5125 |  0.5567 |  0.5733 |
|   EMCDR  | 0.2781 | 0.4364 |  0.4809 |  0.4999 |

## Visualization
For convenience purpose, we stored and provided the detailed results of our experiments in result.json file. You may also run the main.py script to reduce the results. The visualization relies on the result.json file.  
