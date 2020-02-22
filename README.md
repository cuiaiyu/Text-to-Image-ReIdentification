# Language-to-vision Re-Identification: A Re-Implementation of Multi-granularity Image-text Alignments 
- A algorithm to lan
- This is an implementation of ICCV'19 paper [Improving Description-based Person Re-identification by Multi-granularity Image-text Alignments](https://arxiv.org/abs/1906.09610). Refer to the original paper for details.
- This is a beta version implementation. Bug could exist.
- If you find this is useful in your research work, please cite the original paper (and probably star this repo ;-))

## Dataset
Please download the train and val1 setfrom [WIDER Person Search by Language](https://competitions.codalab.org/competitions/20142) dataset and save it in proper folder.

## Performance
With bi-GRU as caption encoder and ResNet-50 as image encoder, we got the following results:
| Model  | R@1  | R@5   | R@10  | 
|---|---|---|---|
|MIA (global-global)  |  47.56 | 71.34  | 79.34  | 
|MIA (global-global + global-part)   | 50.78 | 73.03  | 82.11   |
| MIA (global-global + global-part + part-part) __(reported)__  | 53.10 | 75.00   | 82.90  | 

for computational reason, we haven't implemented the __global-global + global-part + part-part__ versiono of MIA, which might be released in next updates.

## Train
run ```sh src/train.sh```

## Inference

## Visualization
1. Check notebook [src/inference.ipynb](src/inference.ipynb) for interactive retrieval with config set properly

2. 



