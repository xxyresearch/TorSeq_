from argparse import ArgumentParser


def parse_train_args():

    # General arguments
    parser = ArgumentParser()

    # Dataset Argument
    parser.add_argument('--split_path', type=str, default='./data/DRUGS/split.npy', help='Path of file defining the split')
    parser.add_argument('--data_type', type=str, default='drugs', help='Path of file defining the split')
    parser.add_argument('--max_seq_length', type=int, default=100, help='Max Sequence Length')
    parser.add_argument('--conf_num', type=int, default=30, help='the number of generate conf')
    parser.add_argument('--std_dir', type=str, default='/data/DRUGS/standardized_pickles', help='Folder in which the pickle are put after standardisation/matching')

    # Training arguments
    parser.add_argument('--device', type=str, default='cuda:0', help='the device id')
    parser.add_argument('--n_epochs', type=int, default=250, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for preprocessing')
    parser.add_argument('--optimizer', type=str, default='adam', help='Adam optimiser only one supported')
    parser.add_argument('--scheduler', type=str, default='plateau', help='LR scehduler: plateau or none')
    parser.add_argument('--scheduler_patience', type=int, default=20, help='Patience of plateau scheduler')
    
    # Model type arguments
    parser.add_argument('--mpnn_conv', action='store_true', default= False, help='Whether to use mpnn conv')

    # e3nn conv model arguments
    parser.add_argument('--sigma_min', type=float, default=0.01*3.14, help='Minimum sigma used for training')
    parser.add_argument('--sigma_max', type=float, default=3.14, help='Maximum sigma used for training')
    parser.add_argument('--limit_train_mols', type=int, default=0, help='Limit to the number of molecules in dataset, 0 uses them all')
    parser.add_argument('--boltzmann_weight', action='store_true', default=False, help='Whether to sample conformers based on B.w.')
    parser.add_argument('--in_node_features', type=int, default=74, help='Dimension of node features: 74 for drugs and xl, 44 for qm9')
    parser.add_argument('--in_edge_features', type=int, default=4, help='Dimension of edge feature (do not change)')
    parser.add_argument('--sigma_embed_dim', type=int, default=32, help='Dimension of sinusoidal embedding of sigma')
    parser.add_argument('--radius_embed_dim', type=int, default=50, help='Dimension of embedding of distances')
    parser.add_argument('--num_conv_layers', type=int, default=4, help='Number of interaction layers')
    parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
    parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
    parser.add_argument('--ns', type=int, default=32, help='Number of hidden features per node of order 0')
    parser.add_argument('--nv', type=int, default=8, help='Number of hidden features per node of order >0')
    parser.add_argument('--no_residual', action='store_true', default=False, help='If set, it removes residual connection')
    parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
    parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')

    # mpnn model arguments
    parser.add_argument('--hidden_dim', type=int, default=64, help='the hidden dim of mpnn layer')
    parser.add_argument('--gnn_layer_num', type=int, default=3, help='Number of interaction layers')
    parser.add_argument('--latent_dim', type=int, default=10, help='the noise dim of mpnn input')
    parser.add_argument('--model_name', type=str, default='mpnn+lstm', help='the name of saved model')

    # lstm model arguments
    parser.add_argument('--lstm_hidden_dim', type=int, default=512, help='the hidden dim of lstm layer')
    parser.add_argument('--max_length', type=int, default=100, help='max length of torsional sequence')
    parser.add_argument('--lstm_layer_num', type=int, default=2, help='the number of lstm layer')
    parser.add_argument('--bidirection', action='store_false', default= True, help='Whether to use BiDirectional LSTM')

    # mpnn mode arguments
    parser.add_argument('--no_use_lstm', action='store_true', default = False, help='whether not use lstm for torsional angle')
    parser.add_argument('--use_motif_gcn', action='store_true', default = False, help='whether use motif-level gcn')
    parser.add_argument('--no_random_start', action='store_true', default = False, help='whether not start frin random degree')
    parser.add_argument('--no_local_feature', action='store_true', default = False, help='whether not use local feature')
    # linear head arguments
    
    args = parser.parse_args()
    return args
