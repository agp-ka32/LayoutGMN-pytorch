# LayoutGMN: Neural Graph Matching for Structural Layout Similarity
This repo provides the source code for our **[CVPR 2021 paper](https://arxiv.org/pdf/2012.06547.pdf)**.

Overall, the repo consists of two parts:

1) Preparing Layout Graphs
2) Leveraging Graph Matching Networks for structural similarity

# Preparing Layout Graphs
We used two kinds of layout data in our work: Floorplans and UI designs.
Our code for layout graph data preparation is, in parts, borrowed from the work of **[Dipu et al](https://github.com/dips4717/gcn-cnn)**. We will upload the training data here soon. 

# Graph Matching Network-pytorch
The main machinery in our work is a Graph Matching Network that operates on the obtained layout graphs. 
We provide our own pytorch implementation of Graph Matching Networks built on top of the Tensorflow **[Colab implementation](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/graph_matching_networks/graph_matching_networks.ipynb)** by DeepMind Research.

## Requirements
Pytorch >=1.6

CUDA >= 9.1 

networkx >= 2.3

torch-sparse==0.6.7 (pip install torch-sparse)

torch-cluster==1.4.5 (pip install torch-cluster)

torch-geometric==1.3.2 (pip install torch-geometric)


## Usage
The code is split into sub-modules for a cleaner understanding.

The naming of each of the files is self-explanatory.

## Citation
If you find our work useful in your research, please consider citing:

@inproceedings{patil2020layoutgmn,
  title={LayoutGMN: Neural Graph Matching for Structural Layout Similarity},
  author={Patil, Akshay Gadi and Li, Manyi and Fisher, Matthew and Savva, Manolis and Zhang, Hao},
  booktitle ={proc. of Computer Vision and Pattern Recoginition},
  year={2021}
}
