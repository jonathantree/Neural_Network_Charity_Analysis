# Deep Learning Binary Classification of Charity Grant Funding Success
## Project Overview
This project aims to design and optimize a deep learning binary classification model using TensorFlow 2 capable of predicting whether applicants will be successful if funded. The data used in developing this model is historical data of more than 34,000 organizations who applied for funding and the result (successful use/unsuccessful use). 

## Data Preprocessing

The target class: IS_SUCCESSFUL — Was the money used effectively.

These data contain the following feature classes:
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested.

The following data are neither considered a target or feature and are removed from the data prior to subsequent preprocessing steps:

- EIN and NAME—Identification columns
- STATUS—Active status

To streamline preprocessing, a python class with callable cleaning methods was developed. This is the preprocess_class.py file which can be imported and used within the notebooks. Additionally, there is a [preprocessing function](Funcitons/preproc_funcs.py) which can run a full intensive cleaning process on the data if imported and used. An example of how to use the preprocess_class is shown in the preprocess_test.ipynb notebook. Initial preprocessing steps that led to developing these algorithms are shown in the AlphabetSoupCharity.ipynb jupyter notebook.

### Example using the `preprocess_class`

```python
from preprocess_class import preprocess_charity_data as pcd
import pandas as pd
```


```python
csv = 'charity_data.csv'
df = pd.read_csv(csv)
df
```
<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EIN</th>
      <th>NAME</th>
      <th>APPLICATION_TYPE</th>
      <th>AFFILIATION</th>
      <th>CLASSIFICATION</th>
      <th>USE_CASE</th>
      <th>ORGANIZATION</th>
      <th>STATUS</th>
      <th>INCOME_AMT</th>
      <th>SPECIAL_CONSIDERATIONS</th>
      <th>ASK_AMT</th>
      <th>IS_SUCCESSFUL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10520599</td>
      <td>BLUE KNIGHTS MOTORCYCLE CLUB</td>
      <td>T10</td>
      <td>Independent</td>
      <td>C1000</td>
      <td>ProductDev</td>
      <td>Association</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>5000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10531628</td>
      <td>AMERICAN CHESAPEAKE CLUB CHARITABLE TR</td>
      <td>T3</td>
      <td>Independent</td>
      <td>C2000</td>
      <td>Preservation</td>
      <td>Co-operative</td>
      <td>1</td>
      <td>1-9999</td>
      <td>N</td>
      <td>108590</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10547893</td>
      <td>ST CLOUD PROFESSIONAL FIREFIGHTERS</td>
      <td>T5</td>
      <td>CompanySponsored</td>
      <td>C3000</td>
      <td>ProductDev</td>
      <td>Association</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10553066</td>
      <td>SOUTHSIDE ATHLETIC ASSOCIATION</td>
      <td>T3</td>
      <td>CompanySponsored</td>
      <td>C2000</td>
      <td>Preservation</td>
      <td>Trust</td>
      <td>1</td>
      <td>10000-24999</td>
      <td>N</td>
      <td>6692</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10556103</td>
      <td>GENETIC RESEARCH INSTITUTE OF THE DESERT</td>
      <td>T3</td>
      <td>Independent</td>
      <td>C1000</td>
      <td>Heathcare</td>
      <td>Trust</td>
      <td>1</td>
      <td>100000-499999</td>
      <td>N</td>
      <td>142590</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>34294</th>
      <td>996009318</td>
      <td>THE LIONS CLUB OF HONOLULU KAMEHAMEHA</td>
      <td>T4</td>
      <td>Independent</td>
      <td>C1000</td>
      <td>ProductDev</td>
      <td>Association</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34295</th>
      <td>996010315</td>
      <td>INTERNATIONAL ASSOCIATION OF LIONS CLUBS</td>
      <td>T4</td>
      <td>CompanySponsored</td>
      <td>C3000</td>
      <td>ProductDev</td>
      <td>Association</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34296</th>
      <td>996012607</td>
      <td>PTA HAWAII CONGRESS</td>
      <td>T3</td>
      <td>CompanySponsored</td>
      <td>C2000</td>
      <td>Preservation</td>
      <td>Association</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34297</th>
      <td>996015768</td>
      <td>AMERICAN FEDERATION OF GOVERNMENT EMPLOYEES LO...</td>
      <td>T5</td>
      <td>Independent</td>
      <td>C3000</td>
      <td>ProductDev</td>
      <td>Association</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>5000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34298</th>
      <td>996086871</td>
      <td>WATERHOUSE CHARITABLE TR</td>
      <td>T3</td>
      <td>Independent</td>
      <td>C1000</td>
      <td>Preservation</td>
      <td>Co-operative</td>
      <td>1</td>
      <td>1M-5M</td>
      <td>N</td>
      <td>36500179</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>34299 rows × 12 columns</p>
</div>

```python
data_2_clean = pcd(df)
```
```python
cleaning_df = data_2_clean.drop_columns()
```

    Choose the columns from the following:
    Index(['EIN', 'NAME', 'APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION',
           'USE_CASE', 'ORGANIZATION', 'STATUS', 'INCOME_AMT',
           'SPECIAL_CONSIDERATIONS', 'ASK_AMT', 'IS_SUCCESSFUL'],
          dtype='object')
    Enter the list items as listed separated by spaces : EIN NAME SPECIAL_CONSIDERATIONS
    The list of columns to be dropped is: ['EIN', 'NAME', 'SPECIAL_CONSIDERATIONS']
  
```python
cleaning_df.columns
```
    Index(['APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE',
           'ORGANIZATION', 'STATUS', 'INCOME_AMT', 'ASK_AMT', 'IS_SUCCESSFUL'],
          dtype='object')
```python
data_2_clean = pcd(cleaning_df)
clean2_df = data_2_clean.reduce_classification(cutoff=1000)
```

    Reducing the number of Classification Types in df.CLASSIFICATION
    Cutoff value = 1000
  
```python
clean2_df.CLASSIFICATION.value_counts()
```

    C1000    17326
    C2000     6074
    C1200     4837
    Other     2261
    C3000     1918
    C2100     1883
    Name: CLASSIFICATION, dtype: int64
```python
data_2_clean = pcd(clean2_df)
clean3_df = data_2_clean.reduce_affiliation(cutoff=1000)
clean3_df.AFFILIATION.value_counts()
```
    Reducing the number of Classification Types in df.AFFILIATION
    Cutoff value = 1000
    Independent         18480
    CompanySponsored    15705
    Other                 114
    Name: AFFILIATION, dtype: int64

```python
# see what else is available in the pcd class
print (dir(pcd))
```

    ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'ask_amt_binning', 'drop_columns', 'encode_df', 'reduce_affiliation', 'reduce_app_types', 'reduce_classification']
    


```python

```

## Model Results
### Initial Model:
- Hyperparameters:
  - Hidden Layers: 2
    - Layer one: 80 nodes, relu activation function
    - Layer two: 30 nodes, relu activation function
  - Output layer: sigmoid activation function
- Performance: Loss (binary cross-entropy): 0.56 | Accuracy: 72.8%

### All subsequent optimization attempts used a training period of 150 epochs to ensure optimal accuracy had been achieved

### Optimization Attempt #1:
#### Use the same hyperparameters from the previous model to test if intensive feature reduction improves model performance
##### *See AlphabetSoupCharity_Optimzation.ipynb for dataset preparation steps
- Hyperparameters:
  - Hidden Layers: 2
    - Layer one: 80 nodes, relu activation function
    - Layer two: 30 nodes, relu activation function
  - Output layer: sigmoid activation function
- Performance: Loss (binary cross-entropy): 0.56 | Accuracy: 72.6%

This result indicates that model performance was not increased from extensive feature reduction and consolidation of feature variables.

### Optimization Attempt #2:
#### Use the same extensively preprocessed data from the previous model but tune the hyperparameters to investigate if model performance can be improved
#### Add another hidden layer
- Hyperparameters:
  - Hidden Layers: 3
    - Layer one: 80 nodes, relu activation function
    - Layer two: 30 nodes, relu activation function
    - Layer three: 30 nodes, relu activation function
  - Output layer: sigmoid activation function
- Performance: Loss (binary cross-entropy): 0.57 | Accuracy: 72.7%

This result indicates that model performance was not increased from addition of a third hidden layer.

### Optimization Attempt #3:
#### Use the same extensively preprocessed data from the previous model but tune the hyperparameters to investigate if model performance can be improved
#### Switch back to two hidden layers since the previous addition of a third layer did not produce better accuracy but increase the nodes of the layers to 3X the features for the first layer and half of the first layer nodes in the second
- Hyperparameters:
  - Hidden Layers: 2
    - Layer one: 117 nodes, relu activation function
    - Layer two: 58 nodes, relu activation function
  - Output layer: sigmoid activation function
- Performance: Loss (binary cross-entropy): 0.57 | Accuracy: 72.8%

This result indicates that model performance was not increased from hidden layer node addition. 

### Optimization Attempt #4:
#### Use the same extensively preprocessed data from the previous model but tune the hyperparameters to investigate if model performance can be improved
#### Keep the number of layers and node count the same but change the activation function of the two hidden layers 
- Hyperparameters:
  - Hidden Layers: 2
    - Layer one: 117 nodes, tanh activation function
    - Layer two: 58 nodes, tanh activation function
  - Output layer: sigmoid activation function
- Performance: Loss (binary cross-entropy): 0.56 | Accuracy: 72.2%

This result indicates that model performance was not increased from modifying the activation function of the hidden layers. 

### Hyperparameter Optimization using `keras-tuner`
### Attempt 5: Use automate hyperparameter tuning using the extensively processed dataset to gain insight in to unexplored possibilities in parameterization
#### Hyperparameter tuning set up:
```python
# Create a method that creates a new Sequential model with hyperparameter options
def create_model(hp):
    number_input_features = len(X_train[0])
    
    nn_model = tf.keras.models.Sequential()

    # Allow kerastuner to decide which activation function to use in hidden layers
    activation = hp.Choice('activation',['relu','tanh'])
    
    # Allow kerastuner to decide number of neurons in first layer
    nn_model.add(tf.keras.layers.Dense(units=hp.Int('first_units',
        min_value=1,
        max_value=300,
        step=5), activation=activation, input_dim=number_input_features))

    # Allow kerastuner to decide number of hidden layers and neurons in hidden layers
    for i in range(hp.Int('num_layers', 1, 5)):
        nn_model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
            min_value=1,
            max_value=300,
            step=5),
            activation=activation))
    
    nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
    
    # Allow kerastuner to decide which optimization function to use
    optimizers = hp.Choice('optimizer',['Adadelta','Adagrad', 'Adam', 'Adamax', 'FTRL', 'NAdam', 'RMSprop', 'SGD'])
    
    # Compile the model
    nn_model.compile(loss="binary_crossentropy", optimizer=optimizers, metrics=["accuracy"])
    
    return nn_model
    
  tuner = kt.Hyperband(
    create_model,
    objective="val_accuracy",
    max_epochs=50,
    hyperband_iterations=5,
    overwrite=True)
```
### Results from hyperparameter tuning revealed no significant improvements. The project folder checkpoints contains the results from this experiment.

## Summary
Given that extensive feature reduction and preprocessing yeilded no improvements on the model accuracy after hyperparameter tuning, future optimization attempts will have to focus on different preprocessing approaches to attempt to improve model accuracy.
