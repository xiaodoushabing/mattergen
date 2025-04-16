#!/lisa/mattergen/.venv/bin/python

import torch
import gc

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
