from data_provider.data_loader import MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, WADISegLoader
 
from torch.utils.data import DataLoader

data_dict = {
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWaT': SWATSegLoader,
    'WADI': WADISegLoader
}


def data_provider(args, flag, trace_list=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq
    if (flag == 'test' or flag == 'TEST'):
        batch_size = 1
    if args.task_name == 'anomaly_detection':
        if args.data == 'SWaT' or args.data=='WADI':
            drop_last = False
            data_set = Data(
                args = args,
                root_path=args.root_path,
                win_size=args.seq_len,
                flag=flag,
            )
            print(flag, len(data_set))
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
            return data_set, data_loader
        else:
            drop_last = False
            data_set = Data(
                args = args,
                root_path=args.root_path,
                win_size=args.seq_len,
                trace = trace_list,
                flag=flag,
            )
            print(flag, len(data_set))
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
            return data_set, data_loader
    else:
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
