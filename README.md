# Description
LSTM architecture to evaluate sequences of images based off the Oxford LSTM Tutorial [practical 1 repository](https://github.com/oxford-cs-ml-2015/practical1).

# Work Flow
There are two basic Lua functions, one for training `train_lstm.lua` and one for testing the performance of the model `test_lstm.lua`.    

## Input and Preprocessing 
In the `raw/` directory, store the train and test CSVs separately.  For training, the CSVs should be of the format 

| PropertyID    | ImageID     | Class Label |  Representation Columns (input) |
| ------------- |-------------| ------------| :------------------------------:|
| propertyid_1  | imageid_1   |  class_int  |              Numbers            |
| propertyid_2  | imageid_2   |  class_int  |              Numbers            |

With this format, we may excecute the R-script `ReplicateImages`, which creates fixed length sequences of images for the purpose of training.  Specifically, it groups all properties together and repeats the images associated with the property until it reaches the fixed sequence length.  These CSVs may be now stored in the `data/` directory and can now be converted to the necessary format.

## Conversion to Torch 
Both train and test operate on multidimensional [Torch Tensors](https://github.com/torch/torch7/blob/master/doc/tensor.md), so we must convert our CSVs into this format with the `CreateTensors.lua` script.  The `CreateTensors.lua` script has a function, generateTensors which takes as argument the base-name for the CSV file and the relative reference paths to the directory for the CSV files and the outputted tensors.  It is only necesary to run this when the input CSVs have not yet been converted to Torch Tensors.  

## Training the LSTM Architecture
From main directory, one can now call from the command line

```
th train.lua
```

which will execute over the inputted files specified in the `opt` Lua table.  Additional specifications are provided here, however, one must match the following criteria in the table:

1.  Sequence length `opt.seq_length` must match the sequence length declared in R
2.  The input size must `opt.input_size` must match the dimension of the Representation Chosen
3.  The output size `opt.output_size` must match the number of classes provided

# Questions
1.  Representation is sparse at the hidden layer FC7, concerning?
2.  Fix the max-sequence for each image (training and testing) and just generally devise a better strategy here.
