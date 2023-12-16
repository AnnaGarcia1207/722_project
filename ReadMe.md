Group: Anna Garcia & Brian Llinas

This is a term project for Machine Learning class (CS722) - The goal of the project is re-create the [NeurIPS research paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/e00da03b685a0dd18fb6a08af0923de0-Paper-round2.pdf).


Our group re-created the proposed methods in the research paper using different environment. We primarily used  `Python3`, `pandas DataFrame`, `sklearn library`, `transformers`, and others. Please see the different steps on how to execute the code.

## Table of Contents


- [Local Setup](https://github.com/AnnaGarcia1207/722_project?tab=readme-ov-file#local-setup)
- [Running 1st algorithm: Pooling](https://github.com/AnnaGarcia1207/722_project?tab=readme-ov-file#running-1st-algorithm-pooling)
- [Running 2nd Algorithm: Active Learning](https://github.com/AnnaGarcia1207/722_project?tab=readme-ov-file#running-2nd-algorithm-active-learning)
- [Running Extra credit proposal algorithm](https://github.com/AnnaGarcia1207/722_project?tab=readme-ov-file#running-extra-credit-proposal-algorithm)

```
File Structure:

├── al_trained_models <- Trained Logistic Regression For Active Learning method
├── dataset
│   ├── bert_CAL_experiment_results_v3.csv <- Resulting dataset from algorithm 2 with BERT & CAL protocol
│   ├── bert_SAL_experiment_results_v3.csv <- Resulting dataset from algorithm 2 with BERT & SAL protocol
│   ├── tfidf_SAL_experiment_results_v2.csv <- Resulting dataset from algorithm 2 with TFIDF & SAL protocol
│   ├── tfidf_CAL_experiment_results_v2.csv <- Resulting dataset from algorithm 2 with TFIDF & CAL protocol
│   ├── train_test_datasetV2.csv <- Training dataset used to train our models
│   ├── experiment_datasetV2.csv <- Our "Experiment-set" to test the models to build the "Dataset"
├── notebooks
│   ├── 722_PoolingAlgorithmNotebook.ipynb <- Same as pooling.py and pooling_models.py in notebook 
│   ├── 722_Project_CreateDataset.ipynb <- Codes used to build training-set and experiment-set
│   ├── AL-bert-cal.ipynb <- Code for algorithm 2 with BERT & CAL
│   ├── AL-bert-sal.ipynb <- Code for algorithm 2 with BERT & SAL
│   ├── AL-tfidf-cal.ipynb <- Code for algorithm 2 with TFIDF & CAL
│   ├── AL-tfidf-sal.ipynb <- Code for algorithm 2 with TFIDF & SAL
├── pooling_trained_models <- Various trained models for Pooling method
├── pooling.py
├── pooling_models.py
├── ReadMe.md
├── requirements.txt
```

## Local Setup:

1. Make your machine has python3 or you can download [python page](https://www.python.org/downloads/)

```bash
python --version
```

2. Clone the repository:

```bash
git clone https://github.com/AnnaGarcia1207/722_project.git
```

3. `cd` into the directory and install required libraries.
> We suggest to create a vitual environment in this case. On your terminal run line by line:

```bash
python -m venv myenv

myenv/Scripts/activate

pip install -r requirements.txt
```

------
## Running 1st algorithm: Pooling

** The datasets and the models are already trained. They can be found under `/dataset` and `/pooling_trained_models`. But you can also re-run the models and create the poooling dataset yourself.

4. Run the Pooling Models. The porgram will train and save the models
```bash
python pooling_models.py
```

5. Run the Pooling algorithm.
```bash
python pooling.py
```
Sample output:

```
=========================================
Running LogisticRegression model :
AUC: 0.385757532998961
Optimal threshold: 0.9126040168712445
LogisticRegression extracted 62 / 1000
=========================================
Running NaiveBayes model :
AUC: 0.22649452048351443
Optimal threshold: 0.9888543820117094
NaiveBayes extracted 7 / 1000
=========================================
Running GradientBoostingClassifier model :
AUC: 0.4905845530228202
Optimal threshold: 0.9035055984947704
GradientBoostingClassifier extracted 85 / 1000
=========================================
Running SVC model :
AUC: 0.6956505937033663
Optimal threshold: 0.9999812549974917
SVC extracted 228 / 1000
=========================================
Running LinearSVC model :
AUC: 0.5658525426410286
Optimal threshold: 0.9986208248027768
LinearSVC extracted 164 / 1000
=========================================
Preliminary pooling dataset with duplicates length: 546
Number of duplicates: 269
Removing duplicates....
Pooling dataset length: 277
```
----

## Running 2nd Algorithm: Active Learning

Our group opted to use [ODU's HPC Wahab Cluster](https://ondemand.wahab.hpc.odu.edu/pun/sys/dashboard/) because it uses BERT from `transformers library from HuggingFace` and it is very resource intensive, so it must be ran using the HPC environment.

1. You must have an account with `ODU's HPC Wahab Cluster`.

2. Create Jupyter Server with the following parameters:
    > **Python Suite:** tensorflow 2.12 + pytorch 1.13GPU

    > **Number of Cores:** 8

3. Once Jupyter instance has started, click on `Connect to Jupyter`.

4. Create a Jupyter notebook

5. Under `\notebooks`, please upload the following notebooks:

    a.  AL-bert-sal.ipynb

    b. AL-bert-cal.ipynb

    c. AL-tfidf-sal.ipynb

    d. AL-tfidf-cal.ipynb

6. There are four different models being ran for Algorithm 2. To save time, we separated code in four different notebooks to run concurrently.

7. To run the code, click on every cell then click the `RUN` button. You can repeat step 7 for every notebook.

8. Each notebook may produce the following:

    1. Experiment results in CSV format
    2. logistic regression model saved via `pickle library`



----

## Running Extra credit proposal algorithm

1. In order to develop this framework proposal, we used `HPC ODU cluster` and Jupyter notebook and the environment was set up with the following parameters:
    + **Python Suite:** tensorflow 2.12 + pytorch 1.13 GPU
    **Number of Cores:** 8
2. Follow steps in jupyter notebook [Extra credit proposal](notebooks/722_ActiveLearning_bert.ipynb)

