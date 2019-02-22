from models.keras import ensemble
import trainer
from data_process.data_processor import DataLoader
from configs.config import *
from clean_texts import clean_data

import os
import re
import csv
import codecs
import numpy as np
import pandas as pnd
import sys
import logging

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

logging.basicConfig(filename="pipeline.log", level=logging.INFO)

path = 'Dataset/'
EMBEDDING_FILE='features/crawl-300d-2M.vec' 
TRAIN_DATA_FILE=path + 'train.csv'
TEST_DATA_FILE=path + 'test.csv'

MAX_SEQUENCE_LENGTH = 400
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300

logging.info("Start loading data")
train_df = pnd.read_csv(TRAIN_DATA_FILE)
test_df = pnd.read_csv(TEST_DATA_FILE)
data_loader = DataLoader()
embeddings_index = data_loader.load_embedding(EMBEDDING_FILE)

if NEED_CLEANING:
    logging.info("Doing proprocessing cleaning")
    train_df, test_df = clean_data(train_df, test_df)

logging.info('Processing text dataset')
list_sentences_train = train_df["comment_text"].fillna("no comment").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train_df[list_classes].values
list_sentences_test = test_df["comment_text"].fillna("no comment").values

comments = []
for text in list_sentences_train:
    comments.append(text)
    
test_comments=[]
for text in list_sentences_test:
    test_comments.append(text)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(comments + test_comments)

sequences = tokenizer.texts_to_sequences(comments)
test_sequences = tokenizer.texts_to_sequences(test_comments)

word_index = tokenizer.word_index
logging.info('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
logging.info('Shape of data tensor:', data.shape)
logging.info('Shape of label tensor:', y.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
logging.info('Shape of test_data tensor:', test_data.shape)



logging.info('Preparing embedding matrix')
nb_words = min(MAX_NB_WORDS, len(word_index))
#We can initialize embedding with random distribution and train
#embedding_matrix = np.random.normal(loc=matrix_mean, scale=matrix_std, size=(nb_words, EMBEDDING_DIM))
#Or load pre-trained embedding
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
null_count = 0
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        null_count += 1
logging.info('Null word embeddings: %d' % null_count)

def get_model():
    return ensemble.get_av_rnn(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, out_size=6)

keras_model_trainer = trainer.KerasModelTrainer(model_stamp='kmax_text_cnn', epoch_num=50, learning_rate=1e-3)
models, val_loss, total_auc, fold_predictions = keras_model_trainer.train_folds(data, y, fold_count=10, batch_size=256, get_model_func=get_model)
logging.info("Overall val-loss: %f\n" % val_loss)
logging.info("AUC-ROC: %f" % total_auc)

models[0].save(MODEL_CHECKPOINT_FOLDER + 'done_model.h5')
#DO SOME PREDICTIONS


