# Deep Learning Binary Classification of Charity Grant Funding Success
## Project Overview
This project aims to design and optimize a deep learning binary classification model using TensorFlow 2 capable of predicting whether applicants will be successful if funded. The data used in developing this model is historical data of more than 34,000 organizations who applied for funding and the result. These data contain the following feature classes:
- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested.

And the target class IS_SUCCESSFUL — Was the money used effectively.

A first-round attempt at training the model was made that used a preprocessed version of the data in which "rare" categorical variables were binned together as a catchall "Other" category. These data were encoded using one-hot encoding, and used in training a validation of the initial model. Subsequent attempts focused on optimizing the model either by further reducing categorical variables in the dataset, allowing the model to train on the entire dataset without and significant preprocessing, or addition of hidden layers and nodes. The results from these attempts are summarized in the results section below.

## Results
Currently working on this section. Will be adding a summary report of 5 iterations (trained_application_new_v${num}.h5) of optimization efforts soon. 
