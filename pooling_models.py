import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle

# Program will train and save Pooling models

def train_and_save_model(x, y, model_name, save_path):
    seed = 220
    # recommended learning rate for Adam 5e-5, 3e-5, 2e-5
    learning_rate = 2e-5
    number_of_epochs = 10
    # Step 2: Split the Data (Optional) - we can just use the entire X values
    # since we won't be testing
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = None

    if model_name == "naive_bayes":
        model = MultinomialNB()
    elif model_name == "logistic_regression":
        model = LogisticRegression()
    elif model_name == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=seed)
    elif model_name == 'svc':
        model = SVC(probability= True, C=0.01, kernel="rbf")
    elif model_name == "linear_svc":
        svc = LinearSVC(C=0.01)
        model = CalibratedClassifierCV(svc)

    # Step 3: Train the Model
    model.fit(X_train, y_train)

    # Step 3.1 Save the trained model
    with open(save_path, 'wb') as model_file:
            pickle.dump(model, model_file)
    print(f"Exported model to: {save_path}")
    #   return model, X_test, y_test

def main():

    data = pd.read_csv(os.path.join("dataset","train_test_datasetV2.csv"))
    # Step 1: Vectorize the Text Data using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                binary=True,
                                smooth_idf=False)
    X = tfidf_vectorizer.fit_transform(data['clean_tweet'])
    y = data['target']

    save_path = os.path.abspath("pooling_trained_models")

    train_and_save_model(X, y, "logistic_regression", os.path.join(save_path,"logistic_regression.pkl"))
    train_and_save_model(X, y, "naive_bayes",  os.path.join(save_path,"naive_bayes.pkl"))
    train_and_save_model(X, y, "gradient_boosting",  os.path.join(save_path,"gradient_boosting.pkl"))
    train_and_save_model(X, y, "svc", os.path.join(save_path,"svc.pkl"))
    train_and_save_model(X, y, "linear_svc", os.path.join(save_path,"linear_svc.pkl"))

    

if __name__ == "__main__":
    main()