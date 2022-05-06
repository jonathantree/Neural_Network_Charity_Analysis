import pandas as pd 
from sklearn.preprocessing import StandardScaler,OneHotEncoder

def intensive_preprocessing():

    application_df = pd.read_csv("charity_data.csv")
    # Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
    application_df=application_df.drop(columns=['EIN', 'NAME'],axis=1)

    #Bin ASK_AMT
    ASK_AMT_bins = [0, 5000, 500000, 9000000000]
    group_names = ["A", "B", "C"]
    # Categorize spending based on the bins.
    application_df["Amt_Binned"] = pd.cut(application_df.ASK_AMT, ASK_AMT_bins, labels=group_names)

    #Drop the ASK_AMT column and make sure we have what we want going forward
    application_df = application_df.drop('ASK_AMT', axis=1)

    # Look at APPLICATION_TYPE value counts for binning
    app_type_count = application_df.value_counts('APPLICATION_TYPE')

    # Determine which values to replace if counts are less than ...?
    replace_application = list(app_type_count[app_type_count < 500].index)

    # Replace in dataframe
    for app in replace_application:
        application_df.APPLICATION_TYPE = application_df.APPLICATION_TYPE.replace(app,"Other")

    # Look at CLASSIFICATION value counts for binning
    classification_counts = application_df.value_counts('CLASSIFICATION')

    # Determine which values to replace if counts are less than 
    replace_class = list(classification_counts[classification_counts < 500].index)
    # Replace in dataframe
    for cls in replace_class:
        application_df.CLASSIFICATION = application_df.CLASSIFICATION.replace(cls,"Other")

    ## Consolidate the AFFILIATION column    
    aff_counts = application_df.value_counts('AFFILIATION')
    # Determine which values to replace if counts are less than 1000
    replace_aff = list(aff_counts[aff_counts < 1000].index)
    # Replace in dataframe
    for cls in replace_aff:
        application_df.AFFILIATION = application_df.AFFILIATION.replace(cls,"Other")

    final_df = application_df.drop(columns=['SPECIAL_CONSIDERATIONS', 'INCOME_AMT', 'STATUS'])
    # Generate our categorical variable lists
    final_cat = list(final_df.select_dtypes(include=['object', 'category']))
    final_cat
    # Create a OneHotEncoder instance
    enc = OneHotEncoder(sparse=False)

    # Fit and transform the OneHotEncoder using the categorical variable list
    encode_df = pd.DataFrame(enc.fit_transform(final_df[final_cat]))

    # Add the encoded variable names to the dataframe
    encode_df.columns = enc.get_feature_names_out(final_cat)

    # Merge one-hot encoded features and drop the originals
    model_data_df = final_df.merge(encode_df,left_index=True, right_index=True)
    model_data_df = model_data_df.drop(final_cat,1)

    return model_data_df

def get_full_data():
    application_df = pd.read_csv("charity_data.csv")
    application_df=application_df.drop(columns=['EIN', 'NAME'],axis=1)
    # Generate our categorical variable lists
    final_cat = list(application_df.select_dtypes(include=['object', 'category']))
    # Create a OneHotEncoder instance
    enc = OneHotEncoder(sparse=False)

    # Fit and transform the OneHotEncoder using the categorical variable list
    encode_df = pd.DataFrame(enc.fit_transform(application_df[final_cat]))

    # Add the encoded variable names to the dataframe
    encode_df.columns = enc.get_feature_names_out(final_cat)
    
    # Merge one-hot encoded features and drop the originals
    model_data_df = application_df.merge(encode_df,left_index=True, right_index=True)
    model_data_df = model_data_df.drop(final_cat,1)

    return model_data_df 



