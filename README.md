# Bug Report Priority Classification with BERT

This repository contains code for automated bug report priority classification using BERT (Bidirectional Encoder Representations from Transformers). The project includes data extraction, labeling, and both traditional machine learning (XGBoost) and deep learning (BERT) approaches for multi-class classification of bug report priorities (P1-P5).

## 📋 Project Overview

The project automates bug report priority classification through:
1. **Data Extraction**: Converting JSON bug reports to CSV format
2. **Sentiment Analysis**: Multiple approaches for labeling (VADER, SentiWordNet)
3. **Traditional ML**: XGBoost with TF-IDF features
4. **Deep Learning**: Fine-tuned BERT model for sequence classification
5. **Evaluation**: Comprehensive metrics including micro/macro averages and per-class performance

## 📁 Repository Structur


## 🚀 Features

### Data Processing Pipeline
- **JSON to CSV Conversion**: Extract title, body, ID, labels from bug reports
- **Text Preprocessing**: Combine title and body for comprehensive text analysis
- **Sentiment Labeling**: 
  - VADER-based 5-scale sentiment classification
  - SentiWordNet-based scoring and normalization
  - Keyword-based priority labeling (Critical/High/Medium/Low/Negligible → P1-P5)

### Machine Learning Models

#### 1. XGBoost Classifier
- TF-IDF vectorization (5000 features)
- Hyperparameter tuning for optimal performance
- Multi-class classification with 5 priority levels

#### 2. BERT-based Classifier
- Pre-trained `bert-base-uncased` model
- Fine-tuned for 5-class classification
- Configurable hyperparameters:
  - Learning rate: 2e-5
  - Batch size: 16
  - Epochs: 16
  - Max sequence length: 256
  - Hidden size: 768
  - Attention heads: 8

### Evaluation Metrics
- **Per-class metrics**: Precision, Recall, F1-score, Support
- **Aggregate metrics**: Micro and Macro averages
- **Visualization**: Confusion matrices, training curves, distribution plots

## 📊 Dataset

The dataset consists of bug reports with the following fields:
- `title`: Bug report title
- `body`: Detailed description
- `id`: Unique identifier
- `labels`: Original labels
- `sentiment`: VADER-based sentiment (1-5)
- `swn_label`: SentiWordNet-based label (1-5)
- `Priority_Label`: Priority classification (1-5 mapping to Critical → Negligible)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bug-priority-bert.git
cd bug-priority-bert

# Install dependencies
transformers
torch
scikit-learn
xgboost
pandas
numpy
matplotlib
seaborn
nltk
tensorflow

# Run the data extraction and labeling notebook
jupyter notebook Aprio_BERT_data_extraction_and_labeling_Micro_Macro_P1_P5.ipynb
