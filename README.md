# Description
LSTM architecture to evaluate sequences of images based off the Oxford LSTM Tutorial [practical 1 repository](https://github.com/oxford-cs-ml-2015/practical1).

# Code Base
There are two basic Lua functions, one for training `train_lstm.lua` and one for testing the performance of the model `test_lstm.lua`.    

## Preprocessing 
Both train and test operate on multidimensional [Torch Tensors](https://github.com/torch/torch7/blob/master/doc/tensor.md), so we must convert our CSVs into this format with the `CreateTensors.lua` script.  The `CreateTensors.lua` script has a function, generateTensors which takes as argument the base-name for the CSV file and the relative reference paths to the directory for the CSV files and the outputted tensors.  It is only necesary to run this when the input CSVs have not yet been converted to Torch Tensors.  

## Training the LSTM Architecture
From the appropriate environment, one can call them from the command line

```
th train.lua
```

# Questions
1.  Representation is sparse at the hidden layer FC7, concerning?
2.  Fix the max-sequence for each image (training and testing) and just generally devise a better strategy here.
