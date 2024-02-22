from retriever.retriever import (
    RetrieverPipeline,
    setup_retriever_pipelines
)
from arguments import args
import pandas as pd
from envs import *
import tqdm

def read_input_file(filename, query_column, groundtruth_column):
    df = pd.read_csv(filename)
    groundtruth = df[groundtruth_column]
    queries = df[query_column]

    return groundtruth.to_list(), queries.to_list()

retriever_pipeline = setup_retriever_pipelines(args)

groundtruth, queries = read_input_file(filename=FAQ_FILE, 
                                       query_column="Paraphased-question", 
                                       groundtruth_column="Question")

output_data = {
    'Query': [],
    'Top 1 Retrieve': [],
    'Top 1 Score': [],
    'Groundtruth': [],
    'Groundtruth Score': []
}

# Retrieve Document with input Query
for query, truth in tqdm.tqdm(zip(queries, groundtruth), desc="Retrieving Query:..."):
    res = retriever_pipeline["query_pipeline"](query=query, 
                                 debug=True, 
                                 params={"EmbeddingRetriever": {'root_node':'Query', 'index':'faq'}})
    
    truth_score = [(i.score, i.answer) for i in res['answers'] if (i.answer == truth)]
    truth_score, truth_answer = truth_score[0] if len(truth_score) > 0 else ("", "")
    top1_doc = res['answers'][0] if len(res['answers']) > 0 else None

    output_data['Query'].append(query)

    if top1_doc != None:
        output_data['Top 1 Retrieve'].append(top1_doc.answer)
        output_data['Top 1 Score'].append(top1_doc.score)
        output_data['Groundtruth'].append(truth_answer if truth_answer != top1_doc.answer else "")
        output_data['Groundtruth Score'].append(truth_score if truth != top1_doc.answer else "")
    else:
        output_data['Top 1 Retrieve'].append("")
        output_data['Top 1 Score'].append("")
        output_data['Groundtruth'].append(truth_answer)
        output_data['Groundtruth Score'].append(truth_score)

# Output to excel file
df = pd.DataFrame(output_data)
df.to_excel("output_file.xlsx", index=False)