import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument('--weights_path', nargs='?', default='./output/',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='Data/',
                        help='Input data path.')   
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.') 
    parser.add_argument('--dataset', nargs='?', default='buy',
                        help='Choose a dataset from {buy, cart, view}')                    
    parser.add_argument('--dataset1', nargs='?', default='buy',
                        help='Choose a dataset from {buy, cart, view}') #
    parser.add_argument('--dataset2', nargs='?', default='collect',
                        help='Choose a dataset from {buy, cart, view}') #
    parser.add_argument('--dataset3', nargs='?', default='cart',
                        help='Choose a dataset from {buy, cart, view}') #
    parser.add_argument('--dataset4', nargs='?', default='view',
                        help='Choose a dataset from {buy, cart, view}')  #
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.') 
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')  
    parser.add_argument('--is_norm', type=int, default=1, help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64, 64, 64]',
                        help='Output sizes of every layer') 
    parser.add_argument('--layer_size2', nargs='?', default='[64, 64, 64]',
                        help='Output sizes of every layer')
    parser.add_argument('--layer_size3', nargs='?', default='[64, 64, 64, 64, 64]',
                        help='Output sizes of every layer')
    parser.add_argument('--layer_size4', nargs='?', default='[64, 64, 64]',
                        help='Output sizes of every layer')
    parser.add_argument('--layer_size5', nargs='?', default='[64]',
                        help='Output sizes of every layer')

    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.') 

    parser.add_argument('--regs', nargs='?', default='[1e-4]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='mbcgcn',
                        help='Specify the name of model (mbcgcn).')
    parser.add_argument('--adj_type', nargs='?', default='pre',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='mbcgcn',
                        help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.') 

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[10,20,50,80]',
                        help='Top k(s) recommend')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver') 

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    
    parser.add_argument('--label_weight', type=float, default=1.0,
                        help='label_weight')

    parser.add_argument('--shencha', type=int, default=1,
                        help='check code in new test')
    
    parser.add_argument('--gumbel', type=int, default=1,
                        help='weather use gumbel')

    return parser.parse_args()
