import numpy as np
import pandas as pd
from correlations import get_correlations, aggregate_correlations, get_keys, get_values
        
class DataLoader(object):
    def __init__(self, xs, ys, vs, batch_size, pad_with_last_sample=True): 
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            
            v_padding = np.repeat(vs[-1:], num_padding, axis=0)
            vs = np.concatenate([vs, v_padding], axis=0)
            
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.vs = vs

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys
        
        vs = self.vs[permutation]
        self.vs = vs

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size *
                              (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                v_i = self.vs[start_ind: end_ind, ...]
                yield (x_i, y_i, v_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def load_dataset(args, dataset_name='pems-bay', input_horizon=12, output_horizon=12, train_fraction=.7, test_fraction=.2, start_hour=5, end_hour=22, batch_size=64, num_spatial=1):
    print("Getting the data ...")
    data_ = {}
    
    # get data into pivoted df of shape (datetimes, num_nodes)
    if dataset_name in ['pems-bay', 'metr-la']:
        path = f'[Path to the datasets directory]/{dataset_name}.h5'
        data = pd.read_hdf(path)
    else:
        raise Exception(f"Dataset '{dataset_name}' is unknown")
    # Use linear interpolation to fill up nulls
    df = data.interpolate(method='linear', axis=0).ffill().bfill()

    print(df.shape)
    num_nodes = df.shape[1]
    df = df.sort_index(axis=1).reset_index()
    index = df.columns[0]
    df[index] = pd.to_datetime(df[index])
    df = df[(df[index].dt.hour >= start_hour)
            & (df[index].dt.hour <= end_hour)]
    
    speed_df = pd.melt(df, id_vars=df.columns[0], value_vars=df.columns[1:])
    date_time_col, sensors_col, speed_col = speed_df.columns[0], speed_df.columns[1], speed_df.columns[2]
    speed_df[sensors_col] = speed_df[sensors_col].astype('int')

    # compute dow, hour, minute, interval
    speed_df['dow'] = speed_df[date_time_col].dt.day_of_week
    speed_df['hour'] = speed_df[date_time_col].dt.hour
    speed_df['minute'] = speed_df[date_time_col].dt.minute
    rate = 5    # minutes
    speed_df['interval'] = speed_df[date_time_col].dt.minute / rate
    speed_df['interval'] = speed_df['interval'].astype(int)
    speed_df['tod_norm'] = (speed_df[date_time_col].dt.hour*60 + speed_df[date_time_col].dt.minute) / (60*24)    # time of the day
    speed_df['dow_norm'] = (speed_df[date_time_col].dt.day_of_week + 1) / 7

    # Extract training set
    train_samples = df[date_time_col].iloc[:int(train_fraction*df.shape[0])]
    train_df = speed_df[speed_df[date_time_col].isin(train_samples)]

    # compute average speed by segment, dow, hour, interval
    avg_speed = pd.DataFrame(
            {'avg_speed_id_dow_hour_interval': train_df.groupby([sensors_col, "dow", "hour", "interval"])[speed_col].mean()}).reset_index()
    speed_df = speed_df.join(avg_speed.set_index([sensors_col, "dow", "hour", "interval"]), on=[sensors_col, "dow", "hour", "interval"])

    # constrcuction of data split with speed and avg_speed
    x_offsets = np.arange(-input_horizon + 1, 1, 1)
    y_offsets = np.arange(1, output_horizon + 1, 1)
    num_samples = df.shape[0]
    data = []
    speed = pd.pivot_table(speed_df, values=speed_col, index=date_time_col, columns=[sensors_col], aggfunc=np.mean)
    data.append(speed) 
    if args.add_tod:   
        time_of_day = pd.pivot_table(speed_df, values='tod_norm', index=date_time_col, columns=[sensors_col], aggfunc=np.mean)
        time_of_day = (time_of_day - time_of_day.min())/(time_of_day.max() - time_of_day.min())
        data.append(time_of_day.fillna(0))
    if args.add_dow:
        day_of_week = pd.pivot_table(speed_df, values='dow_norm', index=date_time_col, columns=[sensors_col], aggfunc=np.mean)
        day_of_week = (day_of_week - day_of_week.min())/(day_of_week.max() - day_of_week.min())
        data.append(day_of_week.fillna(0))
    avg_speed = pd.pivot_table(speed_df, values='avg_speed_id_dow_hour_interval', index=date_time_col, columns=[sensors_col], aggfunc=np.mean)
    avg_speed = (avg_speed - avg_speed.min())/(avg_speed.max() - avg_speed.min())
    data.append(avg_speed.fillna(0))
    
    data = np.stack(data, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)  # x: (num_samples, input_horizon, num_nodes, input_features)
    y = np.stack(y, axis=0)  # y: (num_samples, output_horizon, num_nodes, output_features)    
    
    num_samples, input_horizon, num_nodes, input_features = x.shape
    values = np.zeros((num_samples, 1, num_nodes, input_features, 1))   # default n case 
    if args.use_neighbors and args.corr_learn_type in ['dot_product', 'sum']:
        correlations = get_correlations(args)
        corrs = [np.concatenate([correlations[..., [i]], correlations[..., -num_spatial:]], axis=-1) for i in range(args.temporal_horizons)]
        learned_corr = np.stack([aggregate_correlations(corrs[i], args.corr_learn_type) for i in range(args.temporal_horizons)], axis=-1)
        keys = get_keys(learned_corr, args.num_neighbors) if args.num_neighbors < num_nodes else learned_corr       
        values = get_values(keys, x.transpose(0, 3, 2, 1)).transpose(0, 3, 2, 1, 4) # shape (num_samples, num_neighbors, num_nodes, in_feats, num_keys)    

    num_samples = x.shape[0]
    num_test = round(num_samples * test_fraction)
    num_train = round(num_samples * train_fraction)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]
    
    v_train, v_val, v_test = values[:num_train], values[num_train: num_train + num_val], values[-num_test:]

    # mkdir(dir_path)
    
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        _v = locals()["v_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape, "v:", _v.shape)
        data_['x_' + cat], data_['y_' + cat] = _x, _y
        data_['v_' + cat] = _v
    scaler = StandardScaler(mean=data_['x_train'][..., 0].mean(), std=data_['x_train'][..., 0].std())
    
    # data format
    for category in ['train', 'val', 'test']:
        data_['x_' + category][..., 0] = scaler.transform(data_['x_' + category][..., 0])
    data_['train_loader'] = DataLoader(data_['x_train'], data_['y_train'], data_['v_train'], batch_size)
    data_['val_loader'] = DataLoader(data_['x_val'], data_['y_val'], data_['v_val'], batch_size)
    data_['test_loader'] = DataLoader(data_['x_test'], data_['y_test'], data_['v_test'], batch_size)
    data_['scaler'] = scaler

    return data_