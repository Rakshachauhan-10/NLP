
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords,words
import re
from contractions import contractions
import pandas as pd

lst_stopwords = nltk.corpus.stopwords.words("english")

def utils_preprocess_text(txt, punkt=True, lower=True, slang=True, lst_stopwords=None, stemm=False, lemm=True):
    ### separate sentences with '. '
    txt = re.sub(r'\.(?=[^ \W\d])', '. ', str(txt))
    ### remove punctuations and characters
    txt = re.sub(r'[^\w\s]', '', txt) if punkt is True else txt
    ### strip
    txt = " ".join([word.strip() for word in txt.split()])
    ### lowercase
    txt = txt.lower() if lower is True else txt
    ### slang
    txt = contractions.fix(txt) if slang is True else txt   
    ### tokenize
    lst_txt = txt.split()
    ### stemming
    if stemm is True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_txt = [ps.stem(word) for word in lst_txt]
    ### lemmatization
    if lemm is True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_txt = [lem.lemmatize(word) for word in lst_txt]
    ### remove Stopwords
    if lst_stopwords is not None:
        lst_txt = [word for word in lst_txt if word not in 
                   lst_stopwords]
        
    txt = " ".join(lst_txt)
    return txt



utils_preprocess_text(txt)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(txt)
sequences = tokenizer.texts_to_sequences(txt)


# Pad sequences
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')




def summarize(text, num_sentences=2):

    sentences = sent_tokenize(text)
    
    words = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    

    freq_dist = FreqDist(words)
    

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_words = word_tokenize(sentence)
        sentence_words = [word.lower() for word in sentence_words if word.isalnum() and word.lower() not in stop_words]
        score = sum([freq_dist[word] for word in sentence_words])
        sentence_scores[i] = score
    

    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

    top_sentences = sorted(top_sentences)
    
    summary = ' '.join([sentences[i] for i in top_sentences])
    
    return summary