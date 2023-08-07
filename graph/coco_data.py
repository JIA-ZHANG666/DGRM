import pickle
import numpy as np

def get_coco_data():
    print('obtaining coco data ...')
    with open("graph/adj_mat_coco.pickle","rb") as f:
        graph_adj_mat = pickle.load(f)

        print('the adj mat is\n',graph_adj_mat)
        print('the type is\n', type(graph_adj_mat))
        print('the shape is\n', graph_adj_mat.shape)
        print('nonzero\n', np.count_nonzero(graph_adj_mat))

        num_symbol_node = graph_adj_mat.shape[0]


    with open("graph/embed_mat_cocov2_300.pickle","rb") as f:
        fasttest_embeddings = pickle.load(f)
        fasttest_dim = fasttest_embeddings.shape[1]

        print('the fasttest_embeddings is\n',fasttest_embeddings)
        print('the type is\n', type(fasttest_embeddings))
        print('the shape is\n', fasttest_embeddings.shape)
        print('nonzero\n', np.count_nonzero(fasttest_embeddings))

    print('obtained coco data')

    return {"num_symbol_node":num_symbol_node,
            "fasttest_embeddings":fasttest_embeddings,
             "fasttest_dim": fasttest_dim,
            "graph_adj_mat": graph_adj_mat
            }



