## Machine Learning - Text Classification with Facebook's FastText and call it from Python


### Development tools

+ Linux ubuntu 18.04
+ Anaconda (https://www.anaconda.com/distribution/) with python 3.7
+ Junyper Notebook

### Python packages required

+ Cython
+ fasttext

### Training Data

In this example we will use a dataset with more than twenty thousand comments user reviews to build a user review model. Luckily, Yelp provides a research dataset of 4.7 million user reviews.

We have a json file called `reviews.json` on [data](https://gitlab.com/reya/ml_supervised_classification_algorithms/tree/master/NLP_Model_for_Beat/data) folder. Each line in the file is a json object with data like this:

```
{
  "review_id": "abc123",
  "user_id": "xyy123",
  "business_id": "1234",
  "stars": 5,
  "date":" 2015-01-01",
  "text": "This restaurant is great!",
  "useful":0,
  "funny":0,
  "cool":0
}
```

### Steps for run example

#### Step 1: Install FastText on the Jupyter Notebook environment

+ Run `prerequisite_install_Cython_FastText.ipynb` file [Code =>](https://gitlab.com/reya/ml_supervised_classification_algorithms/blob/master/NLP_Model_for_Beat/prerequisite_install_Cython_FastText.ipynb)

#### Step 2: Format and Pre-process Training Data

+ Run `preprocess_data_to_fasttext_format.ipynb` file [Code =>](https://gitlab.com/reya/ml_supervised_classification_algorithms/blob/master/NLP_Model_for_Beat/preprocess_data_to_fasttext_format.ipynb)

That will read the reviews.json file and write out a text file in fastText format like this:

```
__label__5 This restaurant is great!
__label__1 This restaurant is terrible :'(
```

Running this creates a new file called `fasttext_dataset.txt` that we can feed into fastText for training.

#### Step 3: Split the data into a Training set and a Test set

+ Run `create_test_train_file_fastText_model.ipynb` file [Code =>](https://gitlab.com/reya/ml_supervised_classification_algorithms/blob/master/NLP_Model_for_Beat/create_test_train_file_fastText_model.ipynb)

Run that and we will have two files, `fasttext_dataset_training.txt` and `fasttext_dataset_test.txt`. Now we are ready to train!

#### Step 4: Train the Model

+ Run `train_fasttext_model.ipynb` file [Code =>](https://gitlab.com/reya/ml_supervised_classification_algorithms/blob/master/NLP_Model_for_Beat/train_fasttext_model.ipynb)

Running this creates a new file called `reviews_model_ngrams.bin` and test the Model

> Important: we created the train model using wordNgrams parameter, That will make it track groups of words instead of just individual words. Read more about this in 
  https://pypi.org/project/fasttext/#description
  
#### Step 5: Run example using the Model

+ Run `test_4_run_example_to_fasttext_text_classification.ipynb` file [Code =>](https://gitlab.com/reya/ml_supervised_classification_algorithms/blob/master/NLP_Model_for_Beat/test_4_run_example_to_fasttext_text_classification.ipynb)

Running to load the model and it uses it to automatically score user reviews.

And here’s what it looks like when it runs:

```
☆ (1 Start)
I don't know. It was ok, I guess. Not really sure what to say.

☆☆ (2 Start)
I hate this place so much. They were mean to me.

☆☆☆ (3 Start)
This restaurant literally changed my life

```

**And that's all, ML is fun but not easy :)**

### References

+ https://pypi.org/project/fasttext/#description
+ https://www.tutorialkart.com/fasttext/make-model-learn-word-representations-using-fasttext-python/
+ https://idevji.com/tutorial-text-classification-with-python-using-fasttext/
+ https://github.com/ahegel/yelp-dataset/blob/master/data/review_sample_cleveland.json
+ https://medium.com/@ageitgey/natural-language-processing-is-fun-9a0bff37854e


### Example without Fasttext for Supervised Algorithms of classification

The `test_3_iris_classification_algorithms.ipynb` file has a code example about supervised algorithms of classification:

1) Logistic Regression, 
2) Support Vector Machines, 
3) Nearest Neighbors and
4) Decision Trees Classification. 

[With this code it created a classifier for the Iris flower.](https://gitlab.com/reya/ml_supervised_classification_algorithms/blob/master/NLP_Model_for_Beat/test_3_iris_classification_algorithms.ipynb)

This dataset it was obtained from [Kaggle](https://www.kaggle.com/uciml/iris). 
