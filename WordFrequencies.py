import os
from collections import Counter
import re
from fuzzywuzzy import process
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (only needs to be done once)
nltk.download('stopwords')

# Create a set of English stopwords
stop_words = set(stopwords.words('english'))

# returns five of the closest matching words in the word set
def get_closest_words(word, word_set):
    closest_matches = process.extract(word, word_set)
    return closest_matches

def collect_word_frequencies(document):
    word_frequencies = Counter()
    word_pattern = re.compile(r'\b\w+\b')

    words = word_pattern.findall(document.lower())  # Convert to lower case for case-insensitive matching

    # Filter out stopwords and update the Counter object with the remaining words from this document
    modal_auxiliaries = {
    'can', 'could', 'may', 'might', 'will', 'would', 'shall', 'should', 'must', 'ought', 'dare', 'need'
    }
    filtered_words = [word for word in words 
                      if word not in stop_words 
                      and not word.isdigit()
                      and word not in modal_auxiliaries]
    word_frequencies.update(filtered_words)

    return word_frequencies

def get_word_frequencies_docs(data):
    data_df = data
    
    word_frequencies = Counter()
    word_dict = {}
    num_docs = data_df.shape[0]
    for index, row in data_df.iterrows():
        print(f'processing {index+1} of {num_docs}',end='\r')
        document = row['text']
        # get word frequencies
        doc_word_frequency = collect_word_frequencies(document)
        word_frequencies += doc_word_frequency
        # add document to list associated with that word
        for word in doc_word_frequency:
            if word not in word_dict:
                word_dict[word] = [row['file']]
            else:
                word_dict[word].append(row['file'])
    print('')
    return (word_dict, word_frequencies)
