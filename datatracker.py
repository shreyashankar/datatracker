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
        self.install_signal_handlers()

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
        df = pd.DataFrame({'id': id})

        # Run all the tracker functions
        for name, tracker_fn in self.tracker_fns.items():
            fn_args = tracker_fn.__code__.co_varnames
            curr_kwargs = {key: kwargs.get(key) for key in fn_args}
            df[name] = tracker_fn(**curr_kwargs)

        # Create iteration column
        df['iter'] = self.iter * np.ones(len(df.index), dtype='int')
        self.dfs.append(df.reset_index(drop=True))

    def get_results(self):
        return pd.concat(self.dfs)

    def dump_results(self, prefix='', include_time=False):
        filename = f'{os.path.join(self.output_dir, prefix)}_{self.iter}'
        if include_time:
            filename += f'_{time.time()}'
        self.get_results().to_csv(filename + '.csv', index=False)
