# Bot/spam comment detection through Transformer Fine-Tuning

This repository contains an implementation of a fine-tuned BERT model to classify whether or not textual content was produced by a bot or not. The project demonstrates effective strategies for adapting pre-trained transformer to a domain-specific dataset.

For this project, I fine-tuned the BERT model on two seperate datasets with a similar objective:

**1 Youtube comment dataset**: [Kaggle repository]([https://github.com/HaohanWang/ImageNet-Sketch](https://www.kaggle.com/datasets/ahsenwaheed/youtube-comments-spam-dataset))\
Dataset is composed of 1936 distinct Youtube comments

**2 Spam data dataset**: [Hugginface dataset]([[https://github.com/HaohanWang/ImageNet-Sketch](https://www.kaggle.com/datasets/ahsenwaheed/youtube-comments-spam-dataset](https://huggingface.co/datasets/Deysi/spam-detection-dataset)))\
Training set: 8175 written spam content
Test set: 2725 written spam content

---

## Project Files: Notebooks only

### **1. bert_spam_detect.ipynb**
- Implement the 'BertForSequenceClassification' model, which:
  - Cleans and Tokenizes the data
  - Fine-tunes the BERT model on 2 epochs
  - Prints the accuracy on the test set
  - Tests predictions on the youtube dataset
---

### **2. bert_spam_detect.ipynb**
- Implement the 'BertForSequenceClassification' model, which:
  - Cleans and Tokenizes the data
  - Fine-tunes 2 BERT models on 3 epochs simultaneously:
    - Model1: Uses youtube comment and Author Name as seperate tensors for training
    - Model2: Only trained on the comment
  - Prints the accuracy on the test set for each model
  - Tests predictions of Model2 on the spam content dataset

### **3. youtube_dataset_creation** (optional, code can be included in main notebook for easier implementation)
- Separates data in training and test sets
