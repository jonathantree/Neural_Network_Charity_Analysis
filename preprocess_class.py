import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder


class preprocess_charity_data:
    
    def __init__(self, dataframe_input):
        self.df = dataframe_input

    def ask_amt_binning(self):
        #Define binning function to call later
        def aa_bin(df):
            print('Binning the AM_ASK column')
            #Bin ASK_AMT
            ASK_AMT_bins = [0, 5000, 500000, 9000000000]
            group_names = ["A", "B", "C"]
            # Categorize spending based on the bins.
            df["Amt_Binned"] = pd.cut(df.ASK_AMT, ASK_AMT_bins, labels=group_names)

            #Drop the ASK_AMT column and make sure we have what we want going forward
            df = df.drop('ASK_AMT', axis=1)
            
            return df
        
        if 'EIN' and 'NAME' in self.df: 
            print('Deleting columns: EIN and NAME')
            self.df = self.df.drop(columns=['EIN', 'NAME'],axis=1)
            
            df = aa_bin(self.df)
            
        else:
            
            df = aa_bin(self.df)
            
        return df
    
    def reduce_app_types(self, cutoff=500):
        
        def reduce_apptype_func(df, cutoff=cutoff):
            
            print ('Reducing the number of Applitcation Types in df.APPLICATION_TYPE')
            print (f'Cutoff value = {cutoff}')
            
            # Look at APPLICATION_TYPE value counts for binning
            app_type_count = df.value_counts('APPLICATION_TYPE')

            # Determine which values to replace if counts are less than ...?
            replace_application = list(app_type_count[app_type_count < cutoff].index)

            # Replace in dataframe
            for app in replace_application:
                df.APPLICATION_TYPE = df.APPLICATION_TYPE.replace(app,"Other")
                
            return df
        
        if 'EIN' and 'NAME' in self.df: 
            print('Deleting columns: EIN and NAME')
            self.df = self.df.drop(columns=['EIN', 'NAME'],axis=1)
            
            df = reduce_apptype_func(self.df)
            
        else:
            
            df = reduce_apptype_func(self.df)
            
        return df
    
    def reduce_classification(self, cutoff=500):
        
        def class_reduce_func(df, cutoff=cutoff):
            
            print ('Reducing the number of Classification Types in df.CLASSIFICATION')
            print (f'Cutoff value = {cutoff}')
            
            # Look at CLASSIFICATION value counts for binning
            classification_counts = df.value_counts('CLASSIFICATION')

            # Determine which values to replace if counts are less than 
            replace_class = list(classification_counts[classification_counts < cutoff].index)
            # Replace in dataframe
            for cls in replace_class:
                df.CLASSIFICATION = df.CLASSIFICATION.replace(cls,"Other")

            return df
            
        if 'EIN' and 'NAME' in self.df: 
            print('Deleting columns: EIN and NAME')
            self.df = self.df.drop(columns=['EIN', 'NAME'],axis=1)
            
            df = class_reduce_func(self.df, cutoff)
            
        else:
            
            df = class_reduce_func(self.df, cutoff)
            
        return df
    
    def reduce_affiliation(self, cutoff=500):
        
        def affiliation_reduce_func(df, cutoff):
        
            print ('Reducing the number of Classification Types in df.AFFILIATION')
            print (f'Cutoff value = {cutoff}')
            
            ## Consolidate the AFFILIATION column    
            aff_counts = df.value_counts('AFFILIATION')
            # Determine which values to replace if counts are less than 1000
            replace_aff = list(aff_counts[aff_counts < cutoff].index)
            # Replace in dataframe
            for cls in replace_aff:
                df.AFFILIATION = df.AFFILIATION.replace(cls,"Other")
                
            return df
        
        if 'EIN' and 'NAME' in self.df: 
            print('Deleting columns: EIN and NAME')
            self.df = self.df.drop(columns=['EIN', 'NAME'],axis=1)
            
            df = affiliation_reduce_func(self.df, cutoff)
            
        else:
            
            df = affiliation_reduce_func(self.df, cutoff)
            
        return df
    
    def drop_columns(self,column_list=[]):
        print ('Choose the columns from the following:')
        print (self.df.columns)
        column_list = [item for item in input("Enter the list items as listed separated by spaces : ").split()]
        print (f'The list of columns to be dropped is: {column_list}')
        
        df = self.df.drop(columns=column_list)
        
        return df
    
    def encode_df(self):
        # Generate our categorical variable lists
        final_cat = list(self.df.select_dtypes(include=['object', 'category']))
        
        # Create a OneHotEncoder instance
        enc = OneHotEncoder(sparse=False)
        # Fit and transform the OneHotEncoder using the categorical variable list
        encode_df = pd.DataFrame(enc.fit_transform(self.df[final_cat]))
        # Add the encoded variable names to the dataframe
        encode_df.columns = enc.get_feature_names_out(final_cat)

        # Merge one-hot encoded features and drop the originals
        model_data_df = seld_df.merge(encode_df,left_index=True, right_index=True)
        model_data_df = model_data_df.drop(final_cat,1)

        return model_data_df