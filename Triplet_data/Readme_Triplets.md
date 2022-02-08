## Generating triplets for training GCN-CNN-TRI model

* To construct positive and negative pairs, we compute an average IoU value between a pair of UIs. For each UI in training set, we compute IoU with remaining UIs. This is computationally expensive. We thus share our generated triplets.

* Triplets used for all the experiments can be obtained from [here](https://drive.google.com/drive/folders/1Qp94A2NQLdBcgaIEuJDJIffk5NHxIVxH?usp=sharing)

* If you want to obtain your own training triplets, then follow steps below.
	* Since it takes several hours to compute pairwise ious, we divide the training set into several segments, and later combine them. Run `python compute_pairwise_IoU.py` for several segment (batch of each 1000) of training UIs. 
	* Once iou values for all the segments (all training UIs) are computed, combine the output pickle files into one. Run `python combine_segments.py` .
	* Next, run `python get_APN_triplet_dict.py` to get the final dictionary of the triplets. 