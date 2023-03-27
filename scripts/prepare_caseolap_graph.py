'''
Prepare the graph which will be used for edge prediction.
ent: entity (e.g., protein)
cat: category (e.g., disease category)
'''
import os
import pandas as pd
import csv
import json
import requests as req
import argparse

def switch_dictset_to_dictlist(d):
    d2 = dict()
    for k,v in d.items():
        d2[k] = list(v)
    return d2


def create_category_subclass_category_edges(tree_to_tree_in_path, first_part_of_all_cats,
                                           score_col, headers, category_tree_path):
    ''' 
    Function:
    - Create edges: Category -subclass_of-> Category 
      e.g., Disease MeSH tree: Disease MeSH Tree - subclass of - Disease MeSH Tree
      Note: This could be made more versatile/abstract so it handles other input data
      such as multiple root nodes, not just one-root trees (e.g., C14)
    Args:
    - tree_to_tree_in_path (str): input file mapping the tree numbers to parents
    - first_part_of_all_cats (str): the first characters in the MeSH tree (e.g., C14 for 
      all cardiovascular diseases)
    - score_col (str): edge score column name 
    - headers (list of str): table headers of the start node, relation, end node, and score
    - category_tree_path (str): output edge file
    '''
    # Input 
    df = pd.read_csv(tree_to_tree_in_path)
    # Output
    cols = list(df.columns)
    tree_to_tree_df = df[df[cols[0]].str.startswith(first_part_of_all_cats)]
    tree_to_tree_df.columns = ['h', 't', 'r']
    tree_to_tree_df = tree_to_tree_df[['h', 'r', 't']]
    tree_to_tree_df[score_col] = [1.0]*len(tree_to_tree_df)
    tree_to_tree_df.columns = headers
    tree_to_tree_df.to_csv(category_tree_path, index=False)

    return tree_to_tree_df


def entity_score_category(entity_name_prefix, cat_name_prefix, ent_to_cat_edge,
                          ent_to_cat_in_path, entity_to_category_path, headers):
    '''
    Function:
    - Create edges for "Entity-score-Category_Name" 
      (e.g., "Protein-CaseOLAP_Score-Disease_Name"
    Args:
    - ent_to_cat_edge (str): edge name for the entity to category association
    - cat_name_prefix (str): category prefix i.e., the category type (e.g., "Disease:")     
    - entity_name_prefix (str): entity prefix i.e., the entity type (e.g., "Protein:")
    - ent_to_cat_in_path (dict): input dictionary, keys are category names, values are 
      dictionaries with keys as entity names and values as entity to category scores 
    - entity_to_category_path (str): output edge file 
    - headers (list of str): table headers of the start node, relation, end node, and score
    '''
    # Input 
    caseolap_dict = json.load(open(ent_to_cat_in_path))
    
    # Output
    with open(entity_to_category_path, 'w', newline='') as fout: 
        writer = csv.writer(fout, delimiter='\t')
        writer.writerow(headers)
        for disease, entity_to_score_d in caseolap_dict.items():
            for entity, score in entity_to_score_d.items():
                writer.writerow([cat_name_prefix+disease, ent_to_cat_edge, 
                                 entity_name_prefix+entity, score])
    caseolap_edge_df = pd.read_table(entity_to_category_path)
    
    return caseolap_edge_df


def category_name_to_tree(category_names_path, tree_txt_in_path, category_to_tree_path, headers,
                         cat_prefix, cat_tree_prefix, cat_to_tree_edge):
    ''' 
    Function:
    - Make edges for Category -is- MeSH Tree e.g., category = CVD
    Args:
    - category_names_path (str): input path of category names in a list
    - tree_txt_in_path (str): lines are space-separate MeSH tree numbers
    - category_to_tree_path (str): output path of edges
    - headers (list of str): table headers of the start node, relation, end node, and score
    - cat_prefix (str): name of cateogory node type
    - cat_tree_prefix (str): name of category tree node type
    - cat_to_tree_edge (str): name of category to tree edge type (one type)
    '''
    # Input
    category_names = json.load(open(category_names_path))
    with open(tree_txt_in_path) as fin:
        mesh_trees = [line.strip().split(' ') for line in fin.readlines()]

    # Output 
    with open(category_to_tree_path, 'w', newline='') as fout:
        writer = csv.writer(fout, delimiter = '\t')
        writer.writerow(headers)
        cat_to_trees = dict(zip(category_names, mesh_trees))
        for cat, trees in cat_to_trees.items():
            for tree in trees:
                writer.writerow([cat_prefix+cat, cat_to_tree_edge, 
                                 cat_tree_prefix+tree, 1.0])
    cat_to_mesh_edge_df = pd.read_table(category_to_tree_path)

    return cat_to_mesh_edge_df


def get_node_df(start_node_col, end_node_col, node_path, node_name, node_type_name):
    '''
    Function:
    - Get a dataframe mapping nodes to the node type. Used in GRAPE's KG edge
      prediction.
    Args:
    - start_node_col (str): columm name of start node
    - end_node_col (str): columm name of end node 
    - node_path (str): output path for node dataframe
    - node_name (str): column name of node for output dataframe
    - node_type_name (str): column name of node type for output dataframe
    '''
    # Input
    nodes = set(edge_df[start_node_col]).union(edge_df[end_node_col])
    
    # Output
    with open(node_path, 'w', newline='') as fout:
        writer = csv.writer(fout, delimiter = '\t')
        writer.writerow([node_name, node_type_name])
        for node in nodes:
            node_type = node.split(':')[0]
            writer.writerow([node, node_type])
    node_df = pd.read_table(node_path).tail()

    return node_df, nodes


def output_entity_subset_to_predict(nodes, entity_subset_outfile, entity_name_prefix):
    '''
    Function:
    - Outputs all the nodes/entities of interest. Configured to output all the nodes
      of a certain node type, typically for the entity node (e.g., protein node)
    Args:
    - nodes (list of str): list of nodes
    - entity_subset_outfile (str): output file of the entities you're interested in
    - entity_name_prefix (str): name of the node type
    '''
    entities = [node for node in nodes if node.startswith(entity_name_prefix)]
    with open(entity_subset_outfile, 'w', newline='') as fout:
        for entity in entities:
            fout.write(entity+'\n')


# Input
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--graph_name', type = str)
parser.add_argument('-s', '--score', default = 'CaseOLAP', type = str)
args = parser.parse_args()
category = args.graph_name
score = args.score

ent_to_cat_in_path = f'../input/{category}/{category}_{score.lower()}.json'
tree_txt_in_path = f'../input/{category}/categories.txt'
tree_to_tree_in_path = f'../input/{category}/mesh_tree_disease_to_mesh_tree_disease.csv'
category_names_path = f'../input/{category}/textcube_config.json'

first_part_of_all_cats = 'MeSH_Tree_Disease:C14'

# Output (intermediate and final files)
entity_to_category_path = f'../data/{category}_protein_to_disease.tsv'
category_tree_path = f'../data/{category}_mesh_disease_to_mesh_disease.csv'
category_to_tree_path = f'../data/{category}_category_to_mesh_tree.tsv'
node_path = f'../data/{category}_caseolap_node_list.tsv'
edge_path = f'../data/{category}_caseolap_edge_list.tsv'
entity_subset_outfile = f'../data/{category}_ca_proteins.txt'

# Node prefixes
cat_prefix = 'Disease:'
cat_name_prefix = 'Disease_Name:'
cat_tree_prefix = 'MeSH_Tree_Disease:'
entity_name_prefix = 'Protein:'

# Node column headers
node_name = 'node'
node_type_name = 'node_type_name'

# Edge names
cat_to_tree_edge = '-cat_is_mesh-' 
ent_to_cat_edge = '-CaseOLAP_Score-'

# Edge column headers
start_node_col = 'head'
relation_col = 'relationship'
end_node_col = 'tail'
score_col = 'score'
headers = [start_node_col, relation_col, end_node_col, score_col]


''' Edges'''
# Edges: Category -subclass_of-> Category e.g., category = Disease_MeSH_Tree 
tree_to_tree_df = create_category_subclass_category_edges(tree_to_tree_in_path, 
                                                          first_part_of_all_cats,
                                                          score_col, 
                                                          headers, 
                                                          category_tree_path)

# Edges: Entity -Score- Category_Name (e.g., Protein -CaseOLAP_Score- Disease_Name)
caseolap_edge_df = entity_score_category(entity_name_prefix,
                                         cat_name_prefix,
                                         ent_to_cat_edge,
                                         ent_to_cat_in_path, 
                                         entity_to_category_path, 
                                         headers)

# Edges: Category -is- MeSH Tree e.g., category = CVD
cat_to_mesh_edge_df = category_name_to_tree(category_names_path, 
                                            tree_txt_in_path, 
                                            category_to_tree_path, 
                                            headers,
                                            cat_prefix, 
                                            cat_tree_prefix,
                                            cat_to_tree_edge)
    
# Merge edge dataframes '''
<<<<<<< HEAD
edge_df = pd.concat([caseolap_edge_df, tree_to_tree_df, cat_to_mesh_edge_df])
=======
edge_df = caseolap_edge_df.append(tree_to_tree_df).append(cat_to_mesh_edge_df)
>>>>>>> bfbe69723692af14fc16537f4dee42b8f5ddc121
edge_df.to_csv(edge_path, sep='\t', index=False)


''' Nodes '''
node_df, nodes = get_node_df(start_node_col, 
                             end_node_col, 
                             node_path, 
                             node_name, 
                             node_type_name)


''' Entity Subset (e.g., Protein Subset) '''
<<<<<<< HEAD
output_entity_subset_to_predict(nodes, entity_subset_outfile, entity_name_prefix)
=======
output_entity_subset_to_predict(nodes, entity_subset_outfile, entity_name_prefix)
>>>>>>> bfbe69723692af14fc16537f4dee42b8f5ddc121
