import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
    new_df: pandas dataframe, output dataframe with joined generic drug name
    '''
    new_df = df.copy()
    df_join = pd.merge(df['ndc_code'], ndc_df[['NDC_Code', 'Non-proprietary Name']], how="left", left_on='ndc_code', right_on='NDC_Code')
    new_df['generic_drug_name'] = df_join['Non-proprietary Name']
    new_df.drop(columns ='ndc_code')
    return new_df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    first_encounter_df = df.copy()
    first_encounters = first_encounter_df.sort_values('encounter_id').groupby('patient_nbr')['encounter_id'].head(1)
    first_encounter_df = first_encounter_df[first_encounter_df['encounter_id'].isin(first_encounters)]
    return first_encounter_df

#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    shuffled_df = df.iloc[np.random.permutation(len(df))]
    unique_keys = shuffled_df[patient_key].unique()
    train_lim = round(len(unique_keys) * .6)
    val_lim = round(len(unique_keys) * .8)
    train = df[df[patient_key].isin(unique_keys[:train_lim])].reset_index(drop=True)
    validation = df[df[patient_key].isin(unique_keys[train_lim:val_lim])].reset_index(drop=True)
    test = df[df[patient_key].isin(unique_keys[val_lim:])].reset_index(drop=True)
    
    assert len(set(train[patient_key].unique()).intersection(set(validation[patient_key].unique()),set(test[patient_key].unique())))==0, \
    "At least one patient's data is in more than one partition!"
    assert train[patient_key].nunique() + validation[patient_key].nunique() + test[patient_key].nunique() == df[patient_key].nunique(), \
    'Number of unique patients is not the same after split!'
    assert len(train) + len(validation) + len(test) == len(df), 'Total number of rows is not the same after split!'
    return train, validation, test

#Question 7
def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        column_vocab = tf.feature_column.categorical_column_with_vocabulary_file(key=c,
                                                                                 vocabulary_file = vocab_file_path,
                                                                                 num_oov_buckets=1)
        tf_categorical_feature_column = tf.feature_column.indicator_column(column_vocab)
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std


def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key=col,
                                                          default_value = default_value,
                                                          normalizer_fn=normalizer,
                                                          dtype=tf.float64)
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = df[col].apply(lambda x: 1 if x >= 5 else 0).values
    return student_binary_prediction
