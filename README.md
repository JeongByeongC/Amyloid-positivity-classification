# Amyloid positivity classification

## Overview

This repository is designed to provide tools and models for the classification of amyloid positivity. Amyloid positivity is a critical marker for diagnosing neurodegenerative diseases, such as Alzheimer's disease. This project aims to leverage machine learning techniques to process imaging and clinical data, ensuring robust predictions and aiding in the early detection and management of these conditions.

if you want to see details see [here](https://petalite-linseed-228.notion.site/Amyloid-positivity-classificati-175d3452270680a89eb3d504bce39ba6?pvs=4)

## code description

+ **Louvain**: Louvain and K-nearest neighborhod (KNN) based amyloid positivity classifier

  + KFold.m: K-fold validation for Louvain method
 
  + KNNpredictor.m: Amyloid positivity prediction using KNN
 
  + Louvain_classifier.m: Louvain method based amyloid positivity classification
 
  + community_louvain.m: Louvain method for commuinity detection on the graph
 
  + louvain_KNN.m: Training the Louvain-KNN classifier

+ GCN_classifier.py: Amyloid positivity classification using graph convolutional neural network (GCN) with or without updating the similarity matrix

+ GCN_draw.m: Bar plot for prediction results

+ analysis_weight.m: Draw the learned weights

+ gcn_conv.py: GCN module

+ predict.py: Predict the amyloid positivity using trained GCN

## Usage

for Luvain method:

```
matlab -nodisplay -nosplash -r    "run('louvain_KNN.m'); exit"
```

for GCN methd:

```
python GCN_classifier.py {number_of_channels_for_GCN} {update_flag}
```
