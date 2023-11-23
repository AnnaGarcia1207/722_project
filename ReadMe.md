Group: Anna Garcia & Brian Llinas

This is a term project for Machine Learning class (CS722) - The goal of the project is re-create the [NeurIPS research paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/e00da03b685a0dd18fb6a08af0923de0-Paper-round2.pdf)


### Setup

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
### Running 1st algorithm: Pooling

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



