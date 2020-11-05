import numpy as np
import os
import pandas as pd
import signal
import sys
import time


class DataTracker:

    def __init__(self, X, y=None, output_dir=None):
        self.dfs = []
        self.id = np.arange(len(X))
        self.y = y
        self.tracker_fns = {}
        self.iter = 0
        self.output_dir = os.path.join(
            os.getcwd(), output_dir if output_dir else 'datatracker_outputs')

        # Make output dir if necessary
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

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

        # Create iteration column
        df['iter'] = iter * np.ones(len(df.index), dtype='int')
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
    
    def get_largest(self, name, k, last_iter=True):
        return self._get_k(name, k, last_iter=last_iter, ascending=False)
    
    def get_smallest(self, name, k, last_iter=True):
        return self._get_k(name, k, last_iter=last_iter, ascending=True)

    def _get_k(self, name, k, last_iter=True, ascending=True):
        df = self.get_results()
        candidate_df = df
        if last_iter:
            max_iter = df['iter'].max()
            candidate_df = df[df['iter'] == max_iter]
        return candidate_df.sort_values(by=[name], ascending=ascending)[['id', name]].head(k)
    
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
        # print(df.sort_values(by=['gain'], ascending=False).head(10))

        gains = df.groupby('id').agg({'gain': 'max'}).reset_index()

        return gains.sort_values(by=['gain'], ascending=ascending).head(k)

