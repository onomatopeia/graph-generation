import time
import logging
import os
import random
import shutil
from pathlib import Path
from random import shuffle
from time import gmtime, strftime
from datetime import datetime

import torch

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorboard_logger import configure

import create_graphs
from args import Args
from data import (
    Graph_sequence_sampler_pytorch,
    Graph_sequence_sampler_pytorch_canonical,
    Graph_sequence_sampler_pytorch_nobfs,
)
from model import GRU_plain, MLP_plain, MLP_VAE_conditional_plain
from train import train
from utils import save_graph_list

def main_training(args: Args):
    # All necessary arguments are defined in args.py    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if not os.path.isdir(args.logs_save_path):
        os.makedirs(args.logs_save_path)
    logging.basicConfig(filename=args.logs_save_path + args.fname + timestamp + '.log', level=logging.INFO)
    logging.info(f'Torch is available {torch.cuda.is_available()}')
    logging.info("Args:\n%s", "\n".join(f"{k} = {v}" for k, v in vars(args).items()))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)


    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    path = str(Path("tensorboard").joinpath("run" + timestamp).resolve())
    configure(path, flush_secs=5)

    graphs = create_graphs.create(args)  # list[nx.Graph]

    # split datasets
    random.seed(123)
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_train = graphs[0:int(0.8 * graphs_len)]
    graphs_validate = graphs[0:int(0.2 * graphs_len)]
    logging.info('graphs_len %d', graphs_len)
    # if use pre-saved graphs
    # dir_input = "/dfs/scratch0/jiaxuany0/graphs/"
    # fname_test = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
    #     args.hidden_size_rnn) + '_test_' + str(0) + '.dat'
    # graphs = load_graph_list(fname_test, is_real=True)
    # graphs_test = graphs[int(0.8 * graphs_len):]
    # graphs_train = graphs[0:int(0.8 * graphs_len)]
    # graphs_validate = graphs[int(0.2 * graphs_len):int(0.4 * graphs_len)]

    graph_validate_len = 0
    for graph in graphs_validate:
        graph_validate_len += graph.number_of_nodes()
    graph_validate_len /= len(graphs_validate)
    logging.info('graph_validate_len %.4f', graph_validate_len)

    graph_test_len = 0
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
    graph_test_len /= len(graphs_test)
    logging.info('graph_test_len %.4f', graph_test_len)

    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    # args.max_num_node = 2000
    # show graphs statistics
    logging.info(f'total graph num: {len(graphs)}, training set: {len(graphs_train)}')
    logging.info('max number node: {}'.format(args.max_num_node))
    logging.info('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
    logging.info('max previous node: {}'.format(args.max_prev_node))

    # save ground truth graphs
    ## To get train and test set, after loading you need to manually slice
    save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    logging.info('train and test graphs saved at: %s', args.graph_save_path + args.fname_test + '0.dat')

    ### comment when normal training, for graph completion only
    # p = 0.5
    # for graph in graphs_train:
    #     for node in list(graph.nodes()):
    #         # print('node',node)
    #         if np.random.rand()>p:
    #             graph.remove_node(node)
    # for edge in list(graph.edges()):
    #     # print('edge',edge)
    #     if np.random.rand()>p:
    #         graph.remove_edge(edge[0],edge[1])

    ### dataset initialization
    if 'nobfs' in args.note:
        logging.info('nobfs')
        dataset = Graph_sequence_sampler_pytorch_nobfs(graphs_train, max_num_node=args.max_num_node)
        args.max_prev_node = args.max_num_node - 1
    if 'barabasi_noise' in args.graph_type:
        logging.info('barabasi_noise')
        dataset = Graph_sequence_sampler_pytorch_canonical(graphs_train, max_prev_node=args.max_prev_node)
        args.max_prev_node = args.max_num_node - 1
    else:
        dataset = Graph_sequence_sampler_pytorch(
            graphs_train, max_prev_node=args.max_prev_node, max_num_node=args.max_num_node
        )
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
        [1.0 / len(dataset) for i in range(len(dataset))],
        num_samples=args.batch_size * args.batch_ratio, replacement=True
    )
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=sample_strategy
    )

    ### model initialization
    ## Graph RNN VAE model
    # lstm = LSTM_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_lstm,
    #                   hidden_size=args.hidden_size, num_layers=args.num_layers).cuda()

    if 'GraphRNN_VAE_conditional' in args.note:
        rnn = GRU_plain(
            input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
            hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
            has_output=False
        ).cuda()
        output = MLP_VAE_conditional_plain(
            h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node
        ).cuda()
    elif 'GraphRNN_MLP' in args.note:
        rnn = GRU_plain(
            input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
            hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
            has_output=False
        ).cuda()
        output = MLP_plain(
            h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node
        ).cuda()
    elif 'GraphRNN_RNN' in args.note:
        rnn = GRU_plain(
            input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
            hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
            has_output=True, output_size=args.hidden_size_rnn_output
        ).cuda()
        output = GRU_plain(
            input_size=1, embedding_size=args.embedding_size_rnn_output,
            hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
            has_output=True, output_size=1
        ).cuda()
    else:
        logging.error('Unknown model type in args.note: %s', args.note)
        raise ValueError(f'Unknown model type in args.note {args.note}')

    ### start training
    train(args, dataset_loader, rnn, output)

    ### graph completion
    # train_graph_completion(args,dataset_loader,rnn,output)

    ### nll evaluation
    # train_nll(args, dataset_loader, dataset_loader, rnn, output, max_iter = 200, graph_validate_len=graph_validate_len,graph_test_len=graph_test_len)

if __name__ == '__main__':
    args = Args()
    main_training(args)