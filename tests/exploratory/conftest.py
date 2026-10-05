"""Windows synthetic-only import shim; never simulate successful Linux locks."""
import sys
import types
import torch

torch.set_num_threads(1)
if sys.platform == 'win32':
    module = types.ModuleType('fcntl')
    def forbidden_lock(*args, **kwargs):
        raise RuntimeError('Linux recovery locks are outside Windows synthetic tests')
    module.flock = forbidden_lock
    module.LOCK_EX, module.LOCK_NB, module.LOCK_UN = 2, 4, 8
    sys.modules['fcntl'] = module
