# Neural_Network_Charity_Analysis

## Overview

The purpose of this project is to use deep-learning neural networks with the TensorFlow platform in Python, to analyze and classify the success of charitable donations.
We use the following methods for the analysis:

1. Preprocessing the data for the neural network model

2. Compile, train and evaluate the model

3. Optimize the model

## Resources

The link to the Data is: [Charity_data](https://github.com/manasidek/Neural_Network_Charity_Analysis/blob/main/charity_data.csv)

## Results

The Script for the analysis is: [AlphabetSoupCharity](https://github.com/manasidek/Neural_Network_Charity_Analysis/blob/main/Deliverable1-2/AlphabetSoupCharity.ipynb)

### Preprocessing the data

- The columns **EIN** and **NAME** are identification information and are removed from the input data.

- The column **IS_SUCCESSFUL** is considered as the target for the deep learning model.

- The following columns **APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT** are the features for the model.

- Encoding of the categorical variables, spliting into training and testing datasets and standardization is applied to the features.


### Compiling, Training, and Evaluating the Model

- This deep-learning neural network model is made of two hidden layers with 80 and 30 neurons respectively.

- The input data has 44 features and 34,298 samples.

- The output layer is made of a unique neuron as it is a binary classification.

- To speed up the training process and to get higher accuracy **ReLU** and **TanH** activation functions are used for the hidden layers. As the output is a binary classification, Sigmoid is used on the output layer.

- For the compilation, the optimizer is **adam** and the loss function is **binary_crossentropy**.

- The model accuracy is under 75%. This is not a satisfying performance to help predict the outcome of the charity donations.

- To increase the performance of the model, bucketing is applied to the feature ASK_AMT and organized the different values by intervals and increased the number of neurons on one of the hidden layers. The accuracy is 72.8%

- Then ran the model with three hidden layers. The accuracy is 72.7%

- Lastly, reduced the number of epochs in the model. The accuracy is 73% 

- None of these steps helped improve the model's performance to 75%

The Script for optimization is: [AlphabetSoupCharity_optimization](https://github.com/manasidek/Neural_Network_Charity_Analysis/blob/main/Deliverable3/AlphabetSoupCharity-optimization.ipynb) 

## Summary

- The deep learning neural network model did not reach the target of 75% accuracy. Considering that this target level is pretty average we could say that the model is not outperforming.

- Since it is a binary classification, a supervised machine learning model such as the **Random Forest Classifier** can be applied, to combine a multitude of decision trees to generate a classified output and evaluate its performance against the deep learning model.