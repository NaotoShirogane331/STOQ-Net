"""Scope, once-only execution and deadline policy for a new exploratory pilot."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import sys


def candidates():
    return [dict(name=name, weight_decay=wd, router_temperature=t, seed=13, epochs=1,
                 batch_size=64, learning_rate=.001, precision='amp_float16')
            for name, wd, t in [('C0', .0001, 1.), ('C-Reg', .001, 1.), ('C-Temp', .0001, 1.25)]]


def validate_candidate(row):
    if not any(row == allowed for allowed in candidates()):
        raise ValueError('candidate differs from fixed approved pilot')
    return row


def input_path(root, split, relative):
    if split not in ('train', 'validation') or relative != split + '/index.json':
        raise ValueError('only original train/validation indices are allowed')
    root = Path(root).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError('input escapes data root')
    return path


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode()


def save(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())


def run_bounded(command, root, cap_seconds):
    if not math.isfinite(cap_seconds) or not 0 < cap_seconds <= 7200:
        raise ValueError('runtime cap must be within new two-hour allowance')
    root = Path(root)
    began = time.monotonic()
    save(root / 'started.json', dict(cap_seconds=cap_seconds, utc=time.time(),
                                   command=command, automatic_retry=False))
    timed_out = False
    parent_pid = os.getpid()
    def bind_parent():
        # Linux prctl is inherited through this non-privileged exec. Check the
        # race in which the parent exits before the child installs PDEATHSIG.
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            os._exit(125)
        if os.getppid() != parent_pid:
            os._exit(125)
    env = dict(os.environ, RACEPV_PILOT_DEADLINE_MONOTONIC=str(began + cap_seconds))
    with (root / 'worker.stdout').open('xb') as out, (root / 'worker.stderr').open('xb') as err:
        child = subprocess.Popen(command, cwd=root, stdout=out, stderr=err,
                                 start_new_session=os.name != 'nt', env=env,
                                 preexec_fn=bind_parent if sys.platform == 'linux' else None)
        try:
            child.wait(timeout=max(.001, cap_seconds - (time.monotonic() - began)))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if child.poll() is None:
                if os.name == 'nt':
                    child.kill()
                else:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
    result = dict(returncode=child.returncode, timed_out=timed_out,
                  whole_seconds=time.monotonic() - began, cap_seconds=cap_seconds,
                  ended_utc=time.time(), automatic_retry=False)
    save(root / 'terminal.json', result)
    return result
