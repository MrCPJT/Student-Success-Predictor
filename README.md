# Student Status Classification
 
## Contents

1. [Overview](#overview)

2. [Data](#data)

3. [Notebook](#notebook)
    1. [EDA](#eda)
    2. [Model Development](#model-development)

4. [Using the Model](#using-the-model)
    1. [Final Model](#final-model)
    2. [Serving Locally](#serving-locally)
    3. [Containerisation](#containerisation)
    4. [Cloud Deployment](#cloud-deployment)

<a name="overview"></a>

## Overview

The quality of higher education institutions is largely determined by the success of their students. Therefore, institutions should aim to minimise the rate at which their students drop out (do not finish the degree course). With the large, feature-rich data that institutions collect from students, there is great potential for analysing current trends, training models and hence making predictions as to whether current students are likely to drop out or not. Students who are deemed likely to drop out can then be targeted and provided with extra social/economic support, hopefully decreasing the likelihood of them not finishing their degree course. The dataset was put together by Portuguese researchers and uses data from Portuguese higher education institutions [1]. Additional information can be found on the Kaggle dataset page (see below) or the featured publication (open access).

 Dataset (Kaggle): https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data

**References:**

[1] Realinho V, Machado J, Baptista L, Martins MV. Predicting Student Dropout and Academic Success. Data. 2022; 7(11), 146-163: https://doi.org/10.3390/data7110146 https://www.mdpi.com/2306-5729/7/11/146

<a name="data"></a>

## Data 
From https://zenodo.org/records/5777340#.Y7FJotJBwUE, an official description of the dataset :

> A dataset created from a higher education institution (acquired from several disjoint databases) related to students enrolled in different undergraduate degrees, such as agronomy, design, education, nursing, journalism, management, social service, and technologies.

> The dataset includes information known at the time of student enrolment (academic path, demographics, and social-economic factors) and the students' academic performance at the end of the first and second semesters.

> The data is used to build classification models to predict students' dropout and academic success. The problem is formulated as a three category classification task (dropout, enrolled, and graduate) at the end of the normal duration of the course.

The dataset can be found in `./Data/dataset.csv` folder along with `train.csv`, `val.csv`, `train_full.csv` and `test.csv` for reproducibility.

<a name="notebook"></a>

## Notebook 
For this project, a Jupyter Notebook (`notebook.ipynb`) was used to understand the data as well as to determine a suitable model for deployment.

Below, I outline some of the key sections of the notebook, but for a complete understanding of what was done, I would encourage one to read through the `notebook.ipynb` file which I have done my best to annotate where appropriate.

<a name="eda"></a>

### EDA 

Having a similar amount of categorical and numerical features, analysis was performed on both types using visualisation packages `matplotlib` and `seaborn`. Univariate analysis was performed for a few categorical features, but halted due to time constraints. The intention is to potentially complete analysis regarding individual features at a later date. Categorical features were modelled as discrete data and visualised using histograms and stacked bar plots, whereas numerical data was modelled as continuous data and visualised using histograms and box plots to highlight any outliers.

In addition to making visual comparisons, correlation and mutual information scores were calculated and used to identify features which were the least important/ had the least relevance to the target vector. The 15 lowest scoring features were then cross examined and used to create a list of features to potentially exclude during model development.

<a name="model-development"></a>

### Model Development 

In order to determine which model we should use for our 'final model', we want to evaluate the performance of several classification models and choose the best performing one. Additionally, we want to compare whether the 'weakest' features (determined during the EDA) affect model performance and hence decided on them included/excluded in the final train.

Using a 60/20/20 (train/validation/test) split - models were trained and cross-validated on the training set then evaluated on the validation set. The best performing model was then trained on an extended-training set (80% of the dataset; train + validation splits) and evaluated on the remaining test data.

Feature matrices were prepared using scikit-learn's `DictVectorizer` and appropriate scaling using scikit-learn's
`MinMaxScaler` to accommodate use of Naïve Bayes (NB) models, which do not accept negative values in the feature matrices.

The following classification models were considered (`random_state = 42` where applicable for reproducibility):

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbours
- Stochastic Gradient Descent
- Support Vector Classification (SVC) / Linear SVC
- Multinomial NB / Bernoulli NB / Gaussian NB

Models were evaluated using the following metrics (applying macro averaging):

- Accuracy
- Cross-validation Accuracy
- Precision
- Recall
- ROC AUC
- F1

Scores were calculated for each model including/excluding the 'weakest features' and the difference between scores was stored in a matrix and represented using a heatmap. It was found that the best performing model without any feature tuning was the Random Forest Classifier with all of the features included.

Having settled on using Random Forest, we used `GridSearchCV` from scikit-learn to tune the model's hyperparameters and optimise performance. The 'best' parameters were recorded and were used to train the model on the extended-training data. When validating our final model on the test split, we found consistent scores to previously which gives us some confidence that our model is not overfitting.

<a name="using-the-model"></a>

## Using the Model 

This section includes some instructions and measures that were taken in order to use the final model.

<a name="final-model"></a>

### Final Model

Having determined our final model (Random Forest) and some optimal model parameter values, we want to prepare a `train.py` file containing only the code necessary to train and export the final model. 

`pickle` is used to write the trained model to `model.bin` and later on used again to read `model.bin` when appropriate.

<a name="serving-locally"></a>

### Serving Locally

With the final model trained and saved, we now want to serve the model using some web service so that we can make calls from our local machine to the service and receive responses.

`predict.py` uses `pickle` to load and read `model.bin` and `Flask` to initialise the webservice (using the POST method). Being a Windows user, `waitress`, a production-quality pure-Python WSGI server, is then used to serve the model using:

`waitress -serve --listen=0.0.0.0:2222 predict:app`

From here, `predict-test.py` can be run (utilising the `requests` package), to test the webservice (see serve_proof.png at the end of this file for a demonstration).

<a name="containerisation"></a>

### Containerisation

Having trained and served our final model, we now want to contain the entire project in an environment and isolate that environment in a container. By doing this, we are able to manage dependencies manually and avoid version conflicts.

This project uses `pipenv` to create and manage a virtual environment. Specific dependencies as well as versions of python can then be installed to the virtual environment using `pipenv install`, and are documented in `Pipfile` / `Pipfile.lock`.

`Pipfile` contains the dependencies and their respective versions necessary for running the service.

Now that the project is contained in a virtual environment, we want to isolate the environment using `Docker`. Isolation is important to avoid conflict with other running environments/services.

`Dockerfile` provides specific instructions on building a Docker container image which can then be run.

Using the `Dockerfile` in this repo, we can build the Docker image by doing:

`docker build -t student-success-serving`

And run the image by using:

`docker run -it --rm -p 2222:2222 student-success-serving`

The following images are demonstrations of building and running Docker on my local machine (NOTE: I found that Docker Desktop needed to be running in order to use Docker successfully).

![Building Docker Image](Proof\docker_build_proof.png "Demonstration of building Docker container image")

![Running Docker Image](Proof\docker_run_proof.png "Demonstration of running Docker container image")  

<a name="cloud-deployment"></a>

### Cloud Deployment

The final step is to deploy our Docker image to a cloud service and for this particular project we will be using `AWS Elastic Beanstalk (EB)`. In order to do this, one must first do the following:

1. Set up an AWS account
2. Create an IAM user 
3. Generate an access key for the IAM user

From here, we can use the EB CLI to initialise a cloud service from the command line. In our virtual environment we first want to install a dev dependency of the EB CLI using:

`pipenv install awsebcli --dev`

Now, we can initialise the EB application, specifying the platform, the region of our service and the service name (NOTE: Your region is most likely different to my own):

`eb init -p docker -r eu-west-2 student-success-serving`

**NOTE: At this stage you will be instructed to input your `access_key_ID` and `secret_access_key` (if you haven't previously done so).**

This generates a `config.yml` file which we can inspect the application's details before testing locally using:

`eb local run --port 2222`

**NOTE: In `config.yml` I had to change `default_platform: Docker` to `default_platform: Docker running on 64bit Amazon Linux 2023`, in order to get the service to run locally. Finally, we can host the EB service on the cloud by running (This make take some time: ~5 minutes):**

`eb create student-success-serving`

And we're done! The service is now deployed to the cloud and accessible from the AWS EB dashboard and the URL provided in the command line (for me this was: http://student-success-serving.eba-yzdssnnc.eu-west-2.elasticbeanstalk.com/predict, but note that I have since terminated the service to save money).

The following two images demonstrate me calling the cloud service from my local command line and my AWS EB dashboard recording that request, respectively.

![Calling Served Model](Proof\serve_proof.png "Demonstration of served model using waitress")

![Deployment Dashboard](Proof\AWS_deploy_proof.png "AWS deployment dashboard")

