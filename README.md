# Pairs-Relation-Machine-Learning-Neural-Network
5 Neural Networks architecture, 3 types of datasets, 3 pre-processing pipelines to use.

About:

This Project is about taking pairs of sentences or any other combination of pairs (X1…. Xn, Y2…. Ym) and learn the relation between the pairs.
The main object is to return a prediction Y for a new input X Using Neural Networks.
It might be related to a lot of subjects in our lives, you can understand the idea by seeing the datasets we chose to work with.  
The System can work and generate pairs from 3 types of datasets:
1. Only Positive Pairs (Dynamic Generating Negative samples)
2. Mixed Positive and Negative (Organize Positive and Negative samples).
3. Already Organized Pairs with class. (No change needed)
The System have 3 pre-processing pipelines to use, from soft preprocessing to hard preprocessing.
The System 5 Neural Networks Models, some work better some less, it more depends on the data itself.
You can save all the preprocessing files and load to train on any Neural Model you want.
The Neural Model also saved after the Learning process.

Models: 
Machine-Learning , Neural-Network:

1)	Deep Convolutional Neural Network with language Embeddings Representation.
2)	Siamese Neural networks with language Embeddings Representation.

*embedding_model
*embedding_model2
*embedding_lstm_model_manhattan_dist (Siamese)
*embedding_lstm_model (Siamese)

Evaluation:

K-Fold.

