import os
import re
import pandas as pd
import numpy as np
import pickle

import statsmodels.formula.api as smf
from sklearn import metrics
from sklearn import tree
from sklearn import ensemble

import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
import statsmodels.formula.api as smf

def get_classification_model(model_file):
    """
    Load a classification model from a file using pickle.

    Parameters:
        model_file (str): The filename or full path to the saved model file.

    Returns:
        model: The loaded classification model.
    """
    try:
        # Load the saved model from the file
        with open(model_file, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    except FileNotFoundError:
        print(f"Model file '{model_file}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

def start_pooling(experiment_dataset):

    # will hold resulting dataset come from pooling
    pooling_dataset = pd.DataFrame()

    # Turn 'clean_tweet' into string type
    experiment_dataset['clean_tweet'] = experiment_dataset['clean_tweet'].apply(str) 

    # load the five models
    dir_path = os.path.abspath("pooling_trained_models")
    logistic_regression_model = get_classification_model(os.path.join(dir_path,"logistic_regression.pkl"))
    naive_bayes_model = get_classification_model(os.path.join(dir_path,"naive_bayes.pkl"))
    gradient_boosting_model = get_classification_model(os.path.join(dir_path,"gradient_boosting.pkl"))
    svc_model = get_classification_model(os.path.join(dir_path,"svc.pkl"))
    linear_svc_model = get_classification_model(os.path.join(dir_path,"linear_svc.pkl"))
    
    # load tfidf vectorizer
    tfidf_vectorizer = get_classification_model(os.path.join(dir_path,"trained_tfidf_vectorizer.pkl"))

    models = [logistic_regression_model,
              naive_bayes_model,
              gradient_boosting_model,
              svc_model,
              linear_svc_model]
    
    model_names = ["LogisticRegression", "NaiveBayes", 
                   "GradientBoostingClassifier", "SVC", "LinearSVC"]
    
    i = 0
    
    for model in models:
        i += 1
        model_name = model_names[i-1]
        print("=========================================")
        print(f"Running {model_name} model : ")
        df = experiment_dataset

        # convert 'clean_tweet' into matrix
        data_features = tfidf_vectorizer.transform(df['clean_tweet'])

        # get the predictions of the model
        predictions = model.predict(data_features)

        # get the probabilities of those predictions
        prediction_probabilities = model.predict_proba(data_features)

        df['y_hat'] = predictions
        df['y_hat_probability'] = [max(prob) for prob in prediction_probabilities]

        fpr, tpr, thresholds=metrics.roc_curve(df['target'],df['y_hat_probability'])
        auc_rl=metrics.auc(fpr,tpr)

        print(f"AUC: {auc_rl}")
        # print(f"TPR: {list(tpr)}")
        # print(f"FPR: {list(fpr)}")
        # print(f"Thresholds: {list(thresholds)}")

        # Create a DataFrame to store thresholds and their Youden's statistic
        df_thresholds = pd.DataFrame({'tpr':tpr,'fpr':fpr,'thresholds':thresholds,'Youden':tpr+(1-fpr)-1})

        # Find the threshold that maximizes Youden's statistic
        optimal_threshold_row = df_thresholds[df_thresholds['Youden'] == df_thresholds['Youden'].max()]

        optimal_threshold = optimal_threshold_row['thresholds'].values[0]
        print(f"Optimal threshold: {optimal_threshold}")

        # Filter rows based on the optimal threshold
        filtered_df = df[df['y_hat_probability'] >= optimal_threshold]

        pooling_dataset = pd.concat([pooling_dataset,filtered_df], ignore_index=True)

        print(f"{model_name} extracted {len(filtered_df)} / {len(df)}")

    
    # delete any duplicates
    print("=========================================")
    print(f"Preliminary pooling dataset with duplicates length: {len(pooling_dataset)}")

    # Check for duplicates in the 'ID' column and count them
    duplicate_count = pooling_dataset['ID'].duplicated().sum()
    print("Number of duplicates:", duplicate_count)
    print("Removing duplicates....")
    # Remove duplicates from the 'ID' column and update the DataFrame
    pooling_dataset = pooling_dataset.drop_duplicates(subset='ID', keep='first')
    print(f"Pooling dataset length: {len(pooling_dataset)}")

    return pooling_dataset


def main():
    exp_df = pd.read_csv(os.path.join("dataset","experiment_datasetV2.csv"))

    pooling_data = start_pooling(exp_df)
    csv_path = os.path.join("dataset","pooling_dataset.csv")
    pooling_data.to_csv(csv_path, index=False)

    print(f"Successfully exported {csv_path}")



if __name__ == "__main__":
    main()
    