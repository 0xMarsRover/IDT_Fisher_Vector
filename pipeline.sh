#!/bin/bash

#UCF_DIR="/Users/Bryan/CS/CS_Research/data/UCF101/"
UCF_DIR="/media/psf/Kellan/video_data_30/data/UCF101"
#TRAIN_LIST="/Users/Bryan/CS/CS_Research/data/class_attributes_UCF101/ucfTrainTestlist/trainlist01.txt"
TRAIN_LIST="/media/psf/Kellan/video_data_30/data/ucfTrainTestlist/trainlist01.txt"
#GMM_OUT="/Users/Bryan/CS/CS_Research/code/CS221/UCF101_Fishers/gmm_list"
GMM_OUT="/media/psf/Kellan/video_data_30/data/UCF101_Fishers/gmm_list"


# Run once
python gmm.py 120 $UCF_DIR $TRAIN_LIST $GMM_OUT --pca


#trainlist01="/Users/Bryan/CS/CS_Research/data/class_attributes_UCF101/ucfTrainTestlist/trainlist01.txt"
trainlist01="/media/psf/Kellan/video_data_30/data/ucfTrainTestlist/trainlist01.txt"

#testlist01="/Users/Bryan/CS/CS_Research/data/class_attributes_UCF101/ucfTrainTestlist/testlist01.txt"
testlist01="/media/psf/Kellan/video_data_30/data/ucfTrainTestlist/testlist01.txt"

#training_output="/Users/Bryan/CS/CS_Research/code/CS221/UCF101_Fishers/train"
training_output="/media/psf/Kellan/video_data_30/data/UCF101_Fishers/train"

#testing_output="/Users/Bryan/CS/CS_Research/code/CS221/UCF101_Fishers/test"
testing_output="/media/psf/Kellan/video_data_30/data/UCF101_Fishers/test"


python computeFVs.py $UCF_DIR $trainlist01 $training_output $GMM_OUT
python computeFVs.py $UCF_DIR $testlist01 $testing_output $GMM_OUT



#CLASS_INDEX="/Users/Bryan/CS/CS_Research/data/class_attributes_UCF101/Class_Index.txt"
CLASS_INDEX="/media/psf/Kellan/video_data_30/data/ucfTrainTestlist/classInd.txt"

#class_index.npz in local path
CLASS_INDEX_OUT="./class_index"

python compute_UCF101_class_index.py $CLASS_INDEX $CLASS_INDEX_OUT

# Perform SVM
python classify_experiment.py