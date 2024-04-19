from openai import OpenAI
import subprocess
import json, re
import sys, os
from WordFrequencies import get_word_frequencies_docs
from GetDocuments import get_documents
import pickle

# mistralai/Mixtral-8x7B-Instruct-v0.1
summary_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
openai_api_key = "EMPTY"
openai_api_base = "http://rbdgx2.cels.anl.gov:8000/v1"
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

data_dir = "/rbstor/ac.cucinell/LUCID/HypothesisWorkflow/prepared_data"

embedding_dict = {
            "biobert": {
                "hypo": os.path.join(data_dir,"biobert_hypo_embeddings.pkl"),
                "exp":os.path.join(data_dir,"biobert_exp_embeddings.pkl"),
                'model': "dmis-lab/biobert-v1.1"
                },
            "biomednlp": {
                "hypo": os.path.join(data_dir,"biomednlp_hypo_embeddings.pkl"),
                "exp":os.path.join(data_dir,"biomednlp_exp_embeddings.pkl"),
                'model': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
            }
        }
data_dict = {
            'text': os.path.join(data_dir,'all_data.pkl')
        }

def summarize_documents(query, documents):
    # prefix = "Give a comprehensive summary of the following documents and how they relate to one another"
    prefix = "Provide an overall, aggregate summary of the following documents and discuss the most relevant information to the query"
    prefix += "above the documents is the query used for searching them:\n" + query
    prompt = f"{prefix}:\n\n{documents}\n\n"
    prompt = json.dumps(prompt)
    message = {
            "role":"user",
            "content":prompt
        }
    try:
        response = client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages = [message],
                temperature=1.0,
                max_tokens=8000
            )
        print('------Response-------')
        print(response.choices[0].message.content)
        print('------End of Response-------')
    except Exception as e:
        print(f"Error in query: \n{e}")

def load_data(data_file, embedding_file):
    data, embedding = None, None
    if data_file:
        data = pickle.load(open (data_file,'rb'))
    if embedding_file:
        embedding = pickle.load(open (embedding_file, 'rb'))
    return data, embedding.values

def main():
    # Variables for script path, embedding file, and data file
    model_name = 'biobert'
    model = embedding_dict[model_name]['model']
    embedding_file = embedding_dict[model_name]['hypo'] 
    data_file = data_dict['text'] 
    data_type = 'hypo'
    # dmis-lab/biobert-v1.1, microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext, 
    top_n = 5
    print("loading data")
    all_data, embeddings = load_data(data_file, embedding_file)
    data = all_data[['file',data_type+'_text']]
    data.columns = ['file','text']
    word_dict, word_frequencies = get_word_frequencies_docs(data)

    print(f"Starting default: {data_type} search summarizing {top_n} documents with model {model}")

    while True:
        # Prompt the user for a query
        query = input("Enter your query (or type 'exit' to terminate): ")
        if query.lower() == 'exit':
            print("Terminating...")
            break

        # skip empty queries
        if query.strip() == '':
            continue
        
        # set the number of documents to summarize
        limit_set_pattern = r"^set limit (\d+)$"
        limit_set_match = re.search(limit_set_pattern, query)
        if limit_set_match:
            value = int(query.split()[-1])
            if value <= 0:
                print("Document limit less than 0, setting to default value (5)")
            else:
                top_n = value
                print("Setting document limit to {}".format(top_n))
            continue

        # switch between hypothesis and experimental search
        data_set_pattern = r"^set data (exp$|hypo$)"
        data_set_match = re.search(data_set_pattern, query.strip())
        if data_set_match:
            value = query.split()[-1]
            if value == 'hypo' or value == 'exp':
                data_type = value
                data = all_data[['file',data_type+'_text']]
                data.columns = ['file','text']
                word_dict, word_frequencies = get_word_frequencies_docs(data)
                print(f'data changed to {data_type}')
            else:
                print(f'data type not support: {value}')
            continue

        # change embedding model
        embedding_set_pattern = r"^set model (biobert$|biomednlp$)$"
        embedding_set_match = re.search(embedding_set_pattern, query.strip())
        if embedding_set_match:
            value = query.split()[-1]
            if value == 'biobert' or value == 'biomednlp':
                model_name = value
                model = embedding_dict[model_name]['model']
                _, embeddings = load_data(None, embedding_dict[value][data_type])
                print(f'embedding model changed to {model}')
            else:
                print(f'embedding model not support: {value}')
            continue

        # summarize
        summary_pattern = r"^summary$"
        summary_match = re.search(summary_pattern, query.strip())
        if summary_match:
            print(f'Query parameters: {data_type} search summarizing {top_n} documents with model {model}')
            continue

        # list commands
        help_pattern = r"^help$"
        help_match = re.search(help_pattern, query.strip())
        if help_match:
            print('Commands:')
            print('Set document summary limit: set limit [number]')
            print('Search hypotheses or experiments: set data exp|hypo')
            print('Change embedding model: set model biobert|biomednlp')
            print('Print current parameter summary: summary')
            continue

        # TODO: arrow keys to iterate through searches

        # Define the command to call GetDocuments.py with necessary arguments
        documents = json.loads(get_documents(model, query, embeddings, data, word_dict, word_frequencies, top_n))

        print('Query = {}'.format(query))
        print('Using top {} documents'.format(top_n))
        summarize_documents(query, '\n'.join(documents))

if __name__ == "__main__":
    main()
