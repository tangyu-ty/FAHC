# FAHC: Frequency Adaptive Hypergraph Constraint for Collaborative Filtering

## introduction
Frequency Adaptive Hypergraph Constraint for Collaborative Filtering(FAHC) devises a Frequency Adaptive graph convolution neural network and hypergraph constraint loss. 
To solve the issue of difficulty in capturing signals with different frequency for conventional graph neural networks, and a new loss is proposed to improve recommendation performance in collaborative filtering.




## Environment
The codes of FAHC are implemented and tested under the following development environment:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install numpy=1.23.4
conda install scipy==1.9.3
```

## Datasets
We utilized three datasets to evaluate HCCF: Yelp, MovieLens, and Amazon. Following the common settings of implicit feedback, if user has rated item , then the element is set as 1, otherwise 0. We filtered out users and items with too few interactions. The datasets are divided into training set, validation set and testing set by 7:1:2.

For more information, please refer to https://github.com/akaxlh/HCCF .
## How to Run the Code
Please unzip the datasets first.Here is an example of a yelp dataset
```bash
python main.py --data=yelp
```
Detailed hyperparameter in `Params.py`


## others

The log file is generated in `/Log`and the model saved file name is in `Saved` .

## Acknowledgements
xxxx
