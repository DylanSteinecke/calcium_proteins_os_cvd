import pandas as pd
from grape import Graph
from kg_analysis import get_edge_type_to_node_types_mapping, independent_edge_evaluation, add_predictions_to_kg
import argparse


# Input
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--graph_name', type = str)
parser.add_argument('-s', '--score', default = 'CaseOLAP', type = str)
args = parser.parse_args()
category = args.graph_name
score = args.score
protein_subset_file = f'../data/{category}_ca_proteins.txt'
graph_name = f'{category}_Calcium_KG'
<<<<<<< HEAD
edge_type_to_predict = f'-{score}_Score-'
=======
edge_type_to_predict = f'-{score}_score-'
>>>>>>> bfbe69723692af14fc16537f4dee42b8f5ddc121

# Node input
node_path = f'../data/{category}_caseolap_node_list.tsv'
node_df_cols = list(pd.read_table(node_path).columns)
nodes_col = node_df_cols[0]
node_type_col = node_df_cols[1]
node_type_default = 'None'

# Edge input
edge_path = f'../data/{category}_caseolap_edge_list.tsv'
edge_df_cols = list(pd.read_table(edge_path).columns) 
start_node_col = edge_df_cols[0]
relation_col = edge_df_cols[1]
end_node_col = edge_df_cols[2]

# Output paths
pred_out_path = f'../output/{category}/predictions.csv'
eval_out_path = f'../output/{category}/eval_results.csv'


# Create graph
g = Graph.from_csv(directed = False,
                  node_path = node_path,
                  edge_path = edge_path,
                  verbose = True,
                  nodes_column = nodes_col,
                  node_list_node_types_column = node_type_col,
                  default_node_type = node_type_default,
                  sources_column = start_node_col,
                  destinations_column = end_node_col,
                  edge_list_edge_types_column = relation_col,
                  name = graph_name)
g = g.remove_disconnected_nodes()

# how many of each edge type?
print(g.get_edge_type_names_counts_hashmap())


# Evaluation Metrics
edge_type_to_node_type = get_edge_type_to_node_types_mapping(g, directed=False)
edge_pair = edge_type_to_node_type[edge_type_to_predict][0]
eval_df, pred_df = independent_edge_evaluation(g, edge_pair, pred_out_path)
eval_df.to_csv(eval_out_path)
print(eval_df)


# Predictions
with open(protein_subset_file) as fin:
    proteins = [l.strip() for l in fin.readlines()]
filt_pred = pred_df[pred_df['destinations'].isin(proteins)]
filt_pred.to_csv(pred_out_path)
print('Number of predicted edges: %f'%filt_pred.shape[0])


# Add predictions to the KG
input_edges = pd.read_csv(edge_path, sep="\t")
<<<<<<< HEAD
print(filt_pred.columns, 'filtered predictin df column')
add_predictions_to_kg(input_edges, filt_pred)
=======
add_predictions_to_kg(input_edges, filt_pred)
>>>>>>> bfbe69723692af14fc16537f4dee42b8f5ddc121
