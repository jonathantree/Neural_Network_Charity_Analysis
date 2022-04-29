# Deep Learning Binary Classification of Charity Grant Funding Success
## Project Overview
This project aims to design and optimize a deep learning binary classification model using TensorFlow 2 capable of predicting whether applicants will be successful if funded. The data used in developing this model is historical data of more than 34,000 orginizations who applied for funding and the result. These data contain the following feature classes:
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

A first round attempt at training the model was made that used a preproccessed version of the data in which "rare" categorical variables were binned together as a catchall "Other" category. These data were encoded using one-hot encoding, and used in training an validation of the initial model.

