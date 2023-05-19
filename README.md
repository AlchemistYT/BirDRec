# BirDRec

## Requirements
- `numpy==1.24.2`
- `scipy==1.10.1`
- `torch==2.0.0`
- `python=3.8.10`
- `CUDA==11.4`
- `seaborn==0.12.2`
- `apex==0.9.10dev`
- `matplotlib==3.7.1`

## Usage
1. Install required packages.
2. run <code>python3 ml1m-main.py</code> to train BirDRec with ML-1M dataset. Similar scripts are prepared for: 
   - Beauty (<code>python3 beauty-main.py</code>)
   - Yelp (<code>python3 yelp-main.py</code>) 
   - and QK-Video (<code>python3 qkvideo-main.py</code>)

## Datasets
- All the datasets used in our paper are organized in [datasets/](datasets/), where each data file contains a list of [userId itemId] pairs which are chronologically ranked for each user.
- ML-1M is from https://grouplens.org/datasets/movielens/1m/,
- Beauty is from http://jmcauley.ucsd.edu/data/amazon/.
- Yelp is from https://www.yelp.com/dataset
- QK-Video is from https://github.com/yuangh-x/2022-NIPS-Tenrec

## Results
The detailed training logs for each dataset are availble in [log/](log/).

## Codes for baselines
- GRU4Rec: https://github.com/hidasib/GRU4Rec
- Caser: https://github.com/graytowne/caser_pytorch
- SASRec: https://github.com/kang205/SASRec
- BERT4Rec: https://github.com/FeiSun/BERT4Rec
- BERD: https://bitbucket.org/SunYatong/berd-ijcai-2021/src/master/
- FMLP-Rec: https://github.com/Woeee/FMLP-Rec
- STEAM: https://github.com/tempsdu/steam
