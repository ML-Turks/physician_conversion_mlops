######import packages and dependencies#########

#Data manimulation libraries
import pandas as pd
import numpy as np
import boto3
from io import BytesIO

#ML packages
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.model_selection import train_test_split

#System and config packages
import sys
import os
import importlib.util #load utils and other scripts
import yaml #read config

#feature store using hopsworks
import hopsworks

#warning ignore package
import warnings
warnings.filterwarnings('ignore')

#custom module
from utils import utils


class DataPrep:

    # Load the YAML configuration from the file
    with open('./conf/feature_pipeline.yml', 'r') as config_file:
        conf = yaml.safe_load(config_file)

    def preprocess_data(self):
        """
        Perform data preprocessing steps as specified in the configuration.

        This method loads data from an S3 bucket, cleans the data, performs one-hot encoding,
        applies feature selection, and saves the preprocessed data back to S3.
        """

        #Load data from s3
        bucket_name = self.conf['s3']['bucket_name']
        file_path = self.conf['s3']['file_path']
        aws_region = self.conf['s3']['aws_region']

        df_input = utils.load_data_from_s3(self, bucket_name,aws_region, file_path)

        #Clean column name and convert to lower case
        df_input.columns = df_input.columns.str.strip()
        df_input.columns = df_input.columns.str.replace(' ', '_')
        df_input.columns = df_input.columns.str.lower()
        
        #Drop unwanted column: "HCO Affiliation" - "Affiliation Type" is more valid column for us
        df_input.drop(self.conf['feature_transformation']['drop_column_list'], axis=1, inplace=True)

        # #One hot encode categorical features
        # df_input = pd.get_dummies(df_input, columns=self.encode_col_list, drop_first=True)
        # df_input.columns = df_input.columns.str.replace(" ", "").str.replace("-", "")
        
        # #Select variables for feature selection
        # id_target_col_list = self.conf['feature_transformation']['id_target_col_list']
        # col_for_feature_selection = df_input.columns.difference(id_target_col_list)
        
        # #Variance threshold feature selection method
        # var_thr = VarianceThreshold(threshold=self.threshold)
        # var_thr.fit(df_input[col_for_feature_selection])

        # df_input_subset = df_input[col_for_feature_selection]
        # remove_col_list = [col for col in df_input_subset.columns if col not in df_input_subset.columns[var_thr.get_support()]]
        
        # #remove above list column from master dataframe
        # df_input.drop(remove_col_list, axis=1, inplace=True, errors='ignore')
        # df_feature_store = df_input.copy()
        
        # # create Xtrain and Ytrain
        # X = df_input.drop("TARGET", axis=1)
        # y = df_input["TARGET"]
        # X_train_set, X_inference, y_train_set, y_inference = train_test_split(X, y,
        #                                                                test_size=self.inference_size, 
        #                                                                random_state=42,
        #                                                                  stratify= y)
        # train_df = pd.concat([X_train_set, y_train_set], axis=1)
        # inference_df = pd.concat([X_inference, y_inference], axis=1)
        
        # # push data to s3 bucket
        # push_status_train = utils_func.push_df_to_s3(train_df, self.bucket_name, self.aws_region, self.train_path)
        # push_status_inference = utils_func.push_df_to_s3(inference_df, self.bucket_name, self.aws_region, self.inference_path)
        # push_status_df = utils_func.push_df_to_s3(df_feature_store, self.bucket_name, self.aws_region, self.s3_object_key)
        
        # #Feature Selection Using Select K Best
        # df = df_input.drop(self.id_col_list, axis=1)
        # target_col_var = df_input[self.target_col]
        # top_n_col_list = utils_func.select_kbest_features(df, target_col_var, self.n)
        
        # #Convert to list
        # top_n_col_list = top_n_col_list.tolist()

        # # Dump top_n_col_list to s3 bucket
        # utils_func.pickle_dump_list_to_s3(top_n_col_list, self.folder_path, self.file_name, self.bucket_name, self.aws_region)

        # #column list for dataframe
        # cols_for_model_df_list = self.id_col_list + top_n_col_list
        # df_feature_eng_output = df_input[cols_for_model_df_list]
        # df_model_input = df_feature_eng_output.copy()

        
        
        
        # #uploading data to Feature Store
        # project = hopsworks.login(
        #     api_key_value=self.api_key,
        #     project=self.project_name,
        # )
        # fs = project.get_feature_store()
        # physician_fs = fs.get_or_create_feature_group(
        #     name=self.table_name,
        #     version=1,
        #     description=self.description,
        #     primary_key=self.lookup_key
        # )
        # physician_fs.insert(train_df)



if __name__ == '__main__':
    # with open('./conf/feature_pipepline.yml', 'r') as config_file:
    #     configuration = yaml.safe_load(config_file)
    task = DataPrep()
    task.preprocess_data()

