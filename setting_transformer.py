import torch
import argparse
import sys


class SettingTransformer:
    """ Settings for the Transformer-based Flashback model. """

    def parse(self):
        self.guess_foursquare = any(['4sq' in argv for argv in sys.argv])
        self.guess_brightkite = any(['brightkite' in argv for argv in sys.argv])

        parser = argparse.ArgumentParser()
        if self.guess_foursquare:
            self.parse_foursquare(parser)
        elif self.guess_brightkite:
            self.parse_brightkite(parser)
        else:
            self.parse_gowalla(parser)
        self.parse_arguments(parser)
        args = parser.parse_args()

        ###### settings ######
        # training
        self.gpu = args.gpu
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.lambda_t = args.lambda_t
        self.lambda_s = args.lambda_s

        # data management
        self.dataset_file = './data/{}'.format(args.dataset)
        self.friend_file = './data/{}'.format(args.friendship)
        self.max_users = 0
        self.sequence_length = args.sequence_length
        self.batch_size = args.batch_size
        self.min_checkins = args.min_checkins

        # evaluation
        self.validate_epoch = args.validate_epoch
        self.validate_epochs = [int(x) for x in args.validate_epochs.split(',')] if args.validate_epochs else []
        self.report_user = args.report_user

        # log
        self.log_file = args.log_file

        self.trans_loc_file = args.trans_loc_file
        self.trans_loc_spatial_file = args.trans_loc_spatial_file
        self.trans_user_file = args.trans_user_file
        self.trans_interact_file = args.trans_interact_file

        self.lambda_user = args.lambda_user
        self.lambda_loc = args.lambda_loc

        self.use_weight = args.use_weight
        self.use_graph_user = args.use_graph_user
        self.use_spatial_graph = args.use_spatial_graph

        ### CUDA Setup ###
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)

    def parse_arguments(self, parser):
        # training
        parser.add_argument('--gpu', default=0, type=int, help='the gpu to use')
        parser.add_argument('--d-model', default=64, type=int, help='transformer model dimension')
        parser.add_argument('--n-heads', default=4, type=int, help='number of attention heads')
        parser.add_argument('--n-layers', default=2, type=int, help='number of transformer layers')
        parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
        parser.add_argument('--weight_decay', default=0, type=float, help='weight decay regularization')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        parser.add_argument('--epochs', default=100, type=int, help='amount of epochs')

        # data management
        parser.add_argument('--dataset', default='checkins-gowalla.txt', type=str,
                            help='the dataset under ./data/<dataset.txt> to load')
        parser.add_argument('--friendship', default='gowalla_friend.txt', type=str,
                            help='the friendship file under ../data/<edges.txt> to load')
        # evaluation
        parser.add_argument('--validate-epoch', default=5, type=int,
                            help='run each validation after this amount of epochs')
        parser.add_argument('--validate-epochs', default='', type=str,
                            help='comma-separated list of specific epochs to validate (overrides --validate-epoch)')
        parser.add_argument('--report-user', default=-1, type=int,
                            help='report every x user on evaluation (-1: ignore)')

        # log
        parser.add_argument('--log_file', default='./results/log_transformer', type=str,
                            help='log file path')
        parser.add_argument('--trans_loc_file', default='./KGE/POI_graph/gowalla_scheme2_transe_loc_temporal_100.pkl', type=str,
                            help='temporal POI transition graph from TransE')
        parser.add_argument('--trans_user_file', default='', type=str,
                            help='user transition graph from TransE')
        parser.add_argument('--trans_loc_spatial_file', default='', type=str,
                            help='spatial POI transition graph from TransE')
        parser.add_argument('--trans_interact_file', default='./KGE/POI_graph/gowalla_scheme2_transe_user-loc_100.pkl', type=str,
                            help='user-POI interaction graph from TransE')
        parser.add_argument('--use_weight', default=False, type=bool, help='use W in GCN AXW')
        parser.add_argument('--use_graph_user', default=False, type=bool, help='use user graph')
        parser.add_argument('--use_spatial_graph', default=False, type=bool, help='use spatial POI graph')

    def parse_gowalla(self, parser):
        parser.add_argument('--batch-size', default=200, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=1000, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')
        parser.add_argument('--sequence-length', default=20, type=int, help='sequence length for splitting check-ins')
        parser.add_argument('--min-checkins', default=101, type=int, help='minimum checkins per user (5*seq_len+1)')

    def parse_foursquare(self, parser):
        parser.add_argument('--batch-size', default=512, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')
        parser.add_argument('--sequence-length', default=20, type=int, help='sequence length for splitting check-ins')
        parser.add_argument('--min-checkins', default=101, type=int, help='minimum checkins per user (5*seq_len+1)')

    def parse_brightkite(self, parser):
        parser.add_argument('--batch-size', default=200, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=1000, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')
        parser.add_argument('--sequence-length', default=20, type=int, help='sequence length for splitting check-ins')
        parser.add_argument('--min-checkins', default=101, type=int, help='minimum checkins per user (5*seq_len+1)')

    def __str__(self):
        if self.guess_foursquare:
            ds = 'foursquare'
        elif self.guess_brightkite:
            ds = 'brightkite'
        else:
            ds = 'gowalla'
        return (f'TransformerFlashback with {ds} default settings\n'
                f'use device: {self.device}\n'
                f'd_model: {self.d_model}, n_heads: {self.n_heads}, '
                f'n_layers: {self.n_layers}, dropout: {self.dropout}')
