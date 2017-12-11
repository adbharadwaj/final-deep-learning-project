import numpy as np
import pandas as pd
import re
import itertools
from collections import Counter
from tensorflow.contrib import learn
import pickle



def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    x_text = sentence_support_df.tokenizedSentenceFromPaper.as_matrix()
    y = sentence_support_df.label.as_matrix()
    y = [[0, 1] if x == 1 else [1, 0] for x in y  ]
    return [x_text, np.array(y)]

def compute_pathway_name_terms(pathway):
    pathway = pathway.replace('signaling', '').replace('pathway', '').replace('-', ' ')
    return [t for t in pathway.lower().strip().split() if len(t)>1]

def tokenize_pathway_names(sentence, pathwayA, pathwayB):
    genesA = [gene.lower() for gene in pathway_to_genes_dict[pathwayA]] + compute_pathway_name_terms(pathwayA)
    genesB = [gene.lower() for gene in pathway_to_genes_dict[pathwayB]] + compute_pathway_name_terms(pathwayB)
    tokenized_sentence = []
    for word in sentence.lower().split():
        token = None
        for gene in genesA:
            if gene in word:
                token = 'pathwayA'
                break
                
        for gene in genesB:
            if gene in word:
                token = 'pathwayB'
                break
        if token is None:
            token = word
        tokenized_sentence.append(token)
    return ' '.join(tokenized_sentence)

def compute_distance_embedding(word, x):
    word_distances = np.zeros(x.shape, dtype='int')
    for i in range(x.shape[0]):
        word_positions = np.where(x[i] == word)[0]
        for j in range(x.shape[1]):
            if len(word_positions) > 0:
                word_position = word_positions[np.argmin(np.abs(word_positions - j))]
                word_distances[i][j] = word_position - j
                if word_distances[i][j]<0:
                    word_distances[i][j] = 600+word_distances[i][j]
            else:
                word_distances[i][j] = 299
    return word_distances

def compute_pos_embedding(data, vocab_processor):
    pos_emebedding = np.zeros(data.shape, dtype='int')
    for i in range(data.shape[0]):
        tags = pos_tag(word_tokenize(list(vocab_processor.reverse([data[i]]))[0].replace('<UNK>', 'XXX')))
        for j in range(data.shape[1]):
            if tags[j][1].lower() in pos_map:
                pos_emebedding[i][j] = pos_map[tags[j][1].lower()]
            else:
                pos_emebedding[i][j] = 6
    return pos_emebedding

def load_pos_embedding():
    return np.load('pos_emebedding.npy')

def load_word_distancesA():
    return np.load('word_distancesA.npy')

def load_word_distancesB():
    return np.load('word_distancesB.npy')

def load_pos_mapping():
    pos_map = {}
    with open('pos-mapping.txt', 'r') as f:
        for lines in f.readlines():
            pos, num = lines.split()
            pos_map[pos] = num
    return pos_map


pathway_to_genes_dict = pickle.load(open( "data/pathway_to_genes_dict.p", "rb" ))
sentence_support_df = pd.read_csv('data/sentence_support_v3.tsv', delimiter='\t')
sentence_support_df.drop_duplicates(inplace=True)
sentence_support_df['tokenizedSentenceFromPaper'] = sentence_support_df.apply(lambda x: tokenize_pathway_names(x.sentenceFromPaper, x.pathwayA, x.pathwayB), axis=1)