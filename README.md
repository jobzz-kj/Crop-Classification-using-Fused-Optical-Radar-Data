# Crop Classification Using Fused Optical and Radar Data

## Project Overview

This project applies machine learning techniques to classify cropland types using fused optical and radar data. The primary objective is to identify seven different crop types—corn, peas, canola, soybeans, oats, wheat, and broadleaf—using a dataset collected near Winnipeg, Canada. The dataset combines data from RapidEye satellites (optical) and UAVSAR (radar) collected on July 5 and July 14, 2012.

## Dataset Details
- **Source:** UCI Machine Learning Repository - "Crop Mapping Using Fused Optical-Radar Data Set"
- **Instances:** ~325,834
- **Features:** 175 total
  - f1 to f49: Polarimetric radar features (July 5, 2012)
  - f50 to f98: Polarimetric radar features (July 14, 2012)
  - f99 to f136: Optical features (July 5, 2012)
  - f137 to f174: Optical features (July 14, 2012)
  - f175: Crop type label (target variable)
- **Target Classes:**
  - 1: Corn
  - 2: Pea
  - 3: Canola
  - 4: Soybean
  - 5: Oat
  - 6: Wheat
  - 7: Broadleaf

## Machine Learning Pipeline

### 1. Data Loading & Verification
- Loaded dataset using `pandas`.
- Verified the class distribution to ensure all seven crop types were present.

### 2. Train-Test Split
- Split into 80% training and 20% testing.
- Used stratified sampling to maintain class distribution.
- Split performed before preprocessing to avoid data leakage.

### 3. Data Preprocessing
- **Data Cleaning:** Replaced missing values with column medians (from training data).
- **Multicollinearity Analysis:** Removed features with a correlation >0.9.
- **Class Balancing:** Applied SMOTE to balance minority classes.
- **Feature Standardization:** Scaled features using `StandardScaler` to have mean 0 and variance 1.
- **Dimensionality Reduction:** Applied PCA (n=50 components) to reduce dimensionality and improve computational efficiency.

### 4. Model Training
We implemented and trained the following models:
- **Gradient Boosting:**
  - Best parameters: `n_estimators=150`, `max_depth=5`, `learning_rate=0.1`
- **Logistic Regression:**
  - Balanced class weights, max iterations=500
- **Random Forest:**
  - 100 estimators, default depth
- **Artificial Neural Network (ANN):**
  - Architecture: 3 hidden layers with (100, 50, 25) neurons
  - Activation: ReLU
  - Optimizer: Adam

### 5. Model Evaluation
- **Cross-validation:** 3-fold stratified
- **Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, MCC, Cohen's Kappa
- **Results:**
  - Gradient Boosting: 98% test accuracy
  - Logistic Regression: 98% test accuracy
  - Random Forest: 99% test accuracy
  - ANN: 100% test accuracy

### 6. Analysis Insights
- High performance attributed to:
  - SMOTE for class balance
  - PCA for feature reduction
  - Robust models like Random Forest and ANN
- Minimal signs of overfitting or data leakage detected.

## Code Execution
To run the model pipeline, follow these steps:

```bash
# Clone the repository
git clone https://github.com/your_username/crop-classification.git
cd crop-classification

# Create a virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
```

### Directory Structure
```
.
├── data
│   └── WinnipegDataset.txt
├── src
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── models
│   └── saved_models
├── results
│   └── evaluation_metrics.txt
├── main.py
├── requirements.txt
└── README.md
```

### Sample Output
```bash
Training Completed.
Gradient Boosting - Accuracy: 98%
Logistic Regression - Accuracy: 98%
Random Forest - Accuracy: 99%
ANN - Accuracy: 100%
```

## Requirements
The following libraries are required to run the project:

```
absl-py==2.1.0
accelerate==1.1.1
ace_tools==0.0
anyascii==0.3.2
appdirs==1.4.4
astunparse==1.6.3
atomicwrites==1.4.0
blis==0.7.11
catalogue==2.0.10
cloudpathlib==0.19.0
confection==0.1.5
contractions==0.1.73
cymem==2.0.8
flatbuffers==24.3.25
gast==0.6.0
google-pasta==0.2.0
grpcio==1.66.2
huggingface-hub==0.26.2
jsonpointer==2.1
keras==3.5.0
langcodes==3.4.1
language_data==1.2.0
libclang==18.1.1
marisa-trie==1.2.0
ml-dtypes==0.4.1
murmurhash==1.0.10
namex==0.0.8
oauthlib==3.2.2
opt_einsum==3.4.0
optree==0.13.0
preshed==3.0.9
protobuf==3.20.3
pyahocorasick==2.1.0
pyasn1-modules==0.2.8
pydantic==1.10.12
pyls-spyder==0.4.0
PyQt5==5.15.10
PyQtWebEngine==5.15.6
safetensors==0.4.5
sentence-transformers==3.3.0
setuptools==69.5.1
shellingham==1.5.4
snscrape==0.7.0.20230622
spacy==3.7.6
spacy-legacy==3.0.12
spacy-loggers==1.0.5
srsly==2.4.8
sympy==1.13.1
tensorboard==2.17.1
tensorboard-data-server==0.7.2
tensorflow==2.17.0
tensorflow-hub==0.16.1
termcolor==2.4.0
textblob==0.18.0.post0
textsearch==0.0.24
tf_keras==2.17.0
thinc==8.2.5
tokenizers==0.20.3
torch==2.5.1
transformers==4.46.3
tweepy==4.14.0
typer==0.12.5
vaderSentiment==3.3.2
wasabi==1.1.3
weasel==0.4.1
wheel==0.43.0
wordcloud==1.9.4
xgboost==2.1.2
```

## Acknowledgements
- UCI Machine Learning Repository for dataset

## License
This project is licensed under the MIT License.

