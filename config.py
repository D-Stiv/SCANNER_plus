import copy
import argparse


class Configuration:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        
        # config parsed by the default parser
        self._config = None

        # individual configurations for different runs
        self._configs = []
        
        # arguments with more than one value
        self._multivalue_args = []
        
        
        
    def parse(self):
        self._config = self._parser.parse_args()
    
        # find values with more than one entry
        dict_config = vars(self._config)
        for k in dict_config :
            if isinstance(dict_config[k], list):
                self._multivalue_args.append(k)

        self._configs.append(self._config)
        for ma in self._multivalue_args:
            new_configs = []

            # in each config
            for c in self._configs:
                # split each attribute with multiple values
                for v in dict_config[ma]:
                    current = copy.deepcopy(c)
                    setattr(current, ma, v)
                    new_configs.append(current)

            # store splitted values
            self._configs = new_configs
        
    def get_configs(self):
        return self._configs
    

def setup_config(config):
    print('Configuration setup ...')

    config._parser.add_argument('--model_name', default='SCANNER+', type=str, help='Name of the model', nargs='*')
    config._parser.add_argument('--dataset_name', default='metr-la', type=str, help='Name of the dataset', nargs='*')
    config._parser.add_argument('--seed', default=42, type=int, help='Random seed', nargs='*')
    config._parser.add_argument('--epochs', default=1, type=int, help='Number of training epochs', nargs='*')
    config._parser.add_argument('--patience', default=15, type=int, help='Number epochs without validation improvement', nargs='*')
    config._parser.add_argument('--distance', default=1, type=int, help='Use the spatial distance', nargs='*')
    config._parser.add_argument('--temporal_horizons', default=12, type=int, help='Number of lags to retrieve', nargs='*')
    config._parser.add_argument('--num_neighbors', default=2, type=int, help='Number of neighbors for each node', nargs='*')
    config._parser.add_argument('--use_neighbors', default=1, type=int, help='Use neighborhood enrichment', nargs='*')
    config._parser.add_argument('--add_tod', default=1, type=int, help='add time of the day in list of input features', nargs='*')
    config._parser.add_argument('--add_dow', default=0, type=int, help='add day of week in list of input features', nargs='*')
    config._parser.add_argument('--batch_size', default=64, type=int, help='', nargs='*')
    config._parser.add_argument('--expid', default=-1, type=int, help='', nargs='*')
    config._parser.add_argument('--expnum', default=-1, type=int, help='', nargs='*')
    config._parser.add_argument("--output_horizon", default=12, type=int, help="number of steps to predict", nargs='*')
    config._parser.add_argument("--input_horizon", default=12, type=int, help="number of historical steps", nargs='*')
    config._parser.add_argument("--snorm", default=1, type=int, help="spatial normalisation", nargs='*')
    config._parser.add_argument("--tnorm", default=1, type=int, help="temporal normalistation", nargs='*')
    config._parser.add_argument("--device", default='cuda', help="Name of the method", nargs='*')
    config._parser.add_argument("--corr_learn_type", default='sum', help="How to construct the spatiotemporal matrices C. sum, dot_product, linear_layer", nargs='*')
    config._parser.add_argument("--filter_agg_type", default='attention_block', help="How to combine horizons. attention_block, linear_layer", nargs='*')
    config._parser.add_argument("--values_equal_keys", default=0, type=int, help="Attention keys and values are identical?", nargs='*')
    config._parser.add_argument("--start_hour", default=5, type=int, help="starting hour to filter the dataset", nargs='*')
    config._parser.add_argument("--end_hour", default=23, type=int, help="ending hour to filter the dataset", nargs='*')
    config._parser.add_argument('--num_heads', type=int, default=1, help='number of heads for attention', nargs='*')
    config._parser.add_argument('--nhid', type=int, default=32, help='', nargs='*')
    config._parser.add_argument('--dropout', type=float, default=.3, help='', nargs='*')
    config._parser.add_argument('--lr', type=float, default=1e-3, help='', nargs='*')
    config._parser.add_argument('--weight_decay', type=float, default=1e-4, help='', nargs='*')
    config._parser.add_argument("--gcn_bool", default=0, type=int, help="whether to add graph convolution layer", nargs='*')
    config._parser.add_argument("--aptonly", default=0, type=int, help="whether only adaptive adj", nargs='*')
    config._parser.add_argument("--adaptadj", default=0, type=int, help="whether add adaptive adj", nargs='*')
    
    config.parse()