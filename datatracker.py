import numpy as np
import os
import pandas as pd
import signal
import sys
import time


class DataTracker:

    def __init__(self, data=None, y=None, id=None, output_dir=None):
        if data is None and id is None:
            raise('Must input either the data or list of ids for the data.')
        
        self.dfs = []
        self.data = data
        self.id = id if id is not None else np.arange(len(self.data))
        if 'id' not in list(self.data.columns):
            self.data['id'] = self.id
        
        self.y = y
        self.tracker_fns = {}
        self.iter = 0
        self.output_dir = os.path.join(
            os.getcwd(), output_dir if output_dir else 'datatracker_outputs')

        # Make output dir if necessary
#         if not os.path.exists(self.output_dir):
#             os.makedirs(self.output_dir)

        # Install signal handlers
        # self.install_signal_handlers()

    def install_signal_handlers(self):

        def sigint_handler(sig, frame):
            try:
                self.dump_results(include_time=True)
                sys.exit(0)
            except Exception as inst:
                print('Unable to save latest artifacts: ', inst)
                sys.exit(1)
        
        def sigstop_handler(sig, frame):
            pass

        def sigcont_handler(sig, frame):
            pass
        
        signal.signal(signal.SIGINT, sigint_handler)

    def add_tracker(self, name, tracker_fn):
        if name in self.tracker_fns:
            raise "An tracker function already exists with this name."
        self.tracker_fns[name] = tracker_fn

    def track(self, **kwargs):
        id = self.id if 'id' not in kwargs else kwargs.get('id')
        iter = self.iter if 'iter' not in kwargs else kwargs.get('iter')
        df = pd.DataFrame({'id': id})

        # Run all the tracker functions
        for name, tracker_fn in self.tracker_fns.items():
            fn_args = tracker_fn.__code__.co_varnames
            curr_kwargs = {key: kwargs.get(key) for key in fn_args}
            df[name] = tracker_fn(**curr_kwargs)
        
        # Create label column
        if self.y is not None:
            df['label'] = self.y[id]

        # Create iteration column
        df['iter'] = iter * np.ones(len(df.index), dtype='int')
        
        if isinstance(iter, int):
            self.iter = iter + 1
        self.dfs.append(df.reset_index(drop=True))

    def get_results(self):
        return pd.concat(self.dfs)

    def dump_results(self, prefix='', include_time=False):
        filename = f'{os.path.join(self.output_dir, prefix)}'
        if include_time:
            filename += f'_{time.time()}'
        if prefix == '':
            filename += f'{time.time()}'
        self.get_results().to_csv(filename + '.csv', index=False)
    
    def get_largest_false_positives(self, name, k, features_to_show=[], last_iter=True):
        if self.y is None:
            raise('Label attribute not found.')
        k_df = self._get_k(name, k=None, last_iter=last_iter, ascending=False)
        k_df = k_df[k_df['label'] != 1].head(k)
        return self._merge_with_features(k_df, features_to_show)
    
    def get_smallest_false_negatives(self, name, k, features_to_show=[], last_iter=True):
        if self.y is None:
            raise('Label attribute not found.')
        k_df = self._get_k(name, k=None, last_iter=last_iter, ascending=True)
        k_df = k_df[k_df['label'] != 0].head(k)
        return self._merge_with_features(k_df, features_to_show)
    
    def get_largest(self, name, k, features_to_show=[], last_iter=True):
        k_df = self._get_k(name, k, last_iter=last_iter, ascending=False)
        return self._merge_with_features(k_df, features_to_show)
    
    def get_smallest(self, name, k, features_to_show=[], last_iter=True):
        k_df = self._get_k(name, k, last_iter=last_iter, ascending=True)
        return self._merge_with_features(k_df, features_to_show)

    def _get_k(self, name, k=None, last_iter=True, ascending=True):
        df = self.get_results()
        candidate_df = df
        if last_iter:
            max_iter = df['iter'].max()
            candidate_df = df[df['iter'] == max_iter]
        cols = ['id', name]
        if self.y is not None:
            cols.append('label')
            
        k_df = candidate_df.sort_values(by=[name], ascending=ascending)[cols]
        if k is not None:
            k_df = k_df.head(k)
        
        return k_df
    
    def _merge_with_features(self, df, features_to_show=[], filter_criteria={}):
        if len(features_to_show) > 0 or filter_criteria is not None:
            if self.data is None:
                raise('Data attribute not found.')
                
            data_df = self.data
            # Filter if there is filter criteria
            for key, val in filter_criteria.items():
                if isinstance(val, list):
                    data_df = data_df[data_df[key].isin(val)]     
                else:
                    data_df = data_df[data_df[key] == val]
            
            cols = ['id'] + list(filter_criteria.keys()) + features_to_show
            seen = set()
            seen_add = seen.add
            cols = [c for c in cols if not (c in seen or seen_add(c))]
            
            return df.merge(data_df[cols], on='id', how='inner')
        
        return df
    
    def get_largest_drop(self, name, k):
        return self._get_drop(name, k, ascending=False)
    
    def get_smallest_drop(self, name, k):
        return self._get_drop(name, k, ascending=True)

    def get_largest_gain(self, name, k):
        return self._get_gain(name, k, ascending=False)
    
    def get_smallest_gain(self, name, k):
        return self._get_gain(name, k, ascending=True)
    
    def _get_drop(self, name, k, ascending=True):
        df = self.get_results().copy()

        # Get performance drops
        df[f'max_{name}'] = df.sort_values(by=['iter'], ascending=True).groupby('id')[name].cummax()
        df[f'min_{name}'] = df.sort_values(by=['iter'], ascending=False).groupby('id')[name].cummin()
        df['drop'] = df[f'max_{name}'] - df[f'min_{name}']
        drops = df.groupby('id').agg({'drop': 'max'}).reset_index()

        return drops.sort_values(by=['drop'], ascending=ascending).head(k)
    
    def _get_gain(self, name, k, ascending=True):
        df = self.get_results().copy()

        # Get performance drops
        df[f'min_{name}'] = df.sort_values(by=['iter'], ascending=True).groupby('id')[name].cummin()
        df[f'max_{name}'] = df.sort_values(by=['iter'], ascending=False).groupby('id')[name].cummax()

        df['gain'] = df[f'max_{name}'] - df[f'min_{name}']

        gains = df.groupby('id').agg({'gain': 'max'}).reset_index()

        return gains.sort_values(by=['gain'], ascending=ascending).head(k)
    
    def get_individual(self, id, name, features_to_show=[]):
        df = self.get_results.copy()
        cols = ['id', name]
        if self.y is not None:
            cols.append('label')
        
        results_df = df[df['id'] == id][cols]
        return self._merge_with_features(results_df, features_to_show)
    
    def get_group(self, filter_criteria, name, features_to_show=[], ascending=True):
        cols = ['id', name]
        if self.y is not None:
            cols.append('label')

        df = self.get_results().copy()[cols]
        return self._merge_with_features(df, features_to_show, filter_criteria).sort_values(by=[name], ascending=ascending)
