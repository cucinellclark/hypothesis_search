import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch, sys
from torch import Tensor
import argparse, json

#device = "mps" if torch.backends.mps.is_available() else "cpu"
device = 'cpu'
#top_n = 5

def get_embeddings(input_texts: list, tokenizer, model) -> Tensor:
    max_length = 512
    batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, add_special_tokens = True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return embeddings

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_top_similar(model_name, query, embedding_matrix, data, word_dict, word_frequencies, top_n):
    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # print(f'top_n = {top_n}')
    # Embed the query
    query_embedding = get_embeddings([query], tokenizer, model)

    '''
    # get list of documents that contain words in the query
    documents = []
    for word in query.split():
        if word not in word_dict:
            continue
        documents += word_dict[word]
    documents = list(set(documents))

    if len(documents) == 0:
        raise ValueError('No documents found for the query')
    matching_indices = data.index[data['file'].isin(documents)].tolist()
    subset_matrix = embedding_matrix[matching_indices, :]
    '''
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, embedding_matrix)
    
    # Get the top N similar rows
    top_indices = np.argsort(similarities[0])[-top_n:]
    
    # # Print the top N similar rows
    # print(f"Top {top_n} similar rows in the embedding matrix:")
    # for index in reversed(top_indices):
    #     print(f"Row {index} with similarity {similarities[0][index]}")
    #     print(f"Hypothesis: {data.iloc[index]['text']}")
    return json.dumps([data.iloc[index]['text'] for index in reversed(top_indices)])

def get_documents(model_name, query, embedding_matrix, data, word_dict, word_frequencies, top_n):

    if data.shape[0] != embedding_matrix.shape[0]:
        raise ValueError('Data and precomputed embedding matrix have different number of rows')
    return(get_top_similar(model_name, query, embedding_matrix, data, word_dict, word_frequencies, top_n))

