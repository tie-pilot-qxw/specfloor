"""Resume only uncommitted AR tasks, retaining original per-task wall timers."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

from run_serving_sweep import ROOT, sha, stop_server, write
from serving_sweep_client import DATASETS, api, request, summarize


def validate_saved(rows, summaries, prepared):
    """A saved task must be whole, identity-matched, and reproduce its summary."""
    counts = dict(DATASETS)
    if set(summaries) - set(counts):
        raise ValueError("Unknown saved task")
    if {r['dataset'] for r in rows} != set(summaries):
        raise ValueError("Saved rows and summaries disagree")
    for name, summary in summaries.items():
        selected = [r for r in rows if r['dataset'] == name]
        expected = [r for r in prepared if r['dataset'] == name]
        identity = lambda r: (r['idx'], r['prompt_sha1'], r['input_ids_sha256'])
        if len(selected) != counts[name] or sorted(map(identity, selected)) != sorted(map(identity, expected)):
            raise ValueError(f"Incomplete or changed task: {name}")
        if summarize(selected, summary['elapsed_s'], 'baseline') != summary:
            raise ValueError(f"Summary differs from raw rows: {name}")
    return [(n, c) for n, c in DATASETS if n not in summaries]


def gpu_state(gpu):
    # A telemetry query timeout is distinct from model failure. Keep the same
    # exclusivity checks, but allow slow nvidia-smi up to 60s; log sample times.
    def query(option, fields):
        return subprocess.check_output(['nvidia-smi', '-i', str(gpu), option + '=' + fields,
            '--format=csv,noheader,nounits'], text=True, timeout=60).strip()
    started = time.time()
    state = query('--query-gpu', 'uuid,memory.used,utilization.gpu')
    processes = query('--query-compute-apps', 'pid,used_memory')
    return dict(time_unix=time.time(), query_started_unix=started, gpu=state,
                processes=processes, pids=[int(s.split(',')[0]) for s in processes.splitlines() if s.strip()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    out = args.run.resolve()
    lock = (out / 'recovery.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prefix = 'baseline.c1'
    if (out / f'{prefix}.complete.json').exists():
        raise RuntimeError('C=1 already complete; refusing to rerun')
    manifest = json.loads((out / 'manifest.json').read_text())
    prepared = [json.loads(s) for s in (out / 'prepared_requests.jsonl').read_text().splitlines()]
    if sha(out / 'prepared_requests.jsonl') != manifest['prepared_requests_sha256']:
        raise RuntimeError('Prepared requests changed')
    saved = [json.loads(s) for s in (out / f'{prefix}.jsonl').read_text().splitlines()]
    summaries = json.loads((out / f'{prefix}.summary.json').read_text())
    remaining = validate_saved(saved, summaries, prepared)
    gpu = manifest['gpu']
    recovery = out / ('recovery_' + time.strftime('%Y%m%dT%H%M%S', time.gmtime()))
    recovery.mkdir()
    for name in ['status.json', f'{prefix}.jsonl', f'{prefix}.summary.json', 'results.json']:
        shutil.copy2(out / name, recovery / ('original.' + name))
    provenance = dict(started_unix=time.time(), retained_tasks=list(summaries),
        remaining_tasks=[n for n, _ in remaining], reason='Original nvidia-smi 15s timeout',
        telemetry_timeout_s=60, runner_sha256=sha(__file__), original_manifest=str(out / 'manifest.json'),
        note='C=1 resumes at whole-task boundaries; no elapsed timers span server restarts.')
    write(recovery / 'manifest.json', provenance)
    write(out / 'baseline.c1.recovery.json', dict(provenance, path=str(recovery)))
    def verify_sources():
        for path, expected in manifest['source_sha256'].items():
            if sha(path) != expected:
                raise RuntimeError(f'Source changed: {path}')
    verify_sources()
    cmd = json.loads((out / 'baseline.command.json').read_text())
    port = cmd[cmd.index('--port') + 1]
    base = f'http://127.0.0.1:{port}'
    env = dict(os.environ, HF_HOME='/workspace/.cache/huggingface', HF_HUB_OFFLINE='1',
        TMPDIR='/workspace/tmp', PYTHONPATH=str(ROOT / 'refine_exp/serving_sweep_bootstrap'),
        SGLANG_RECORD_STEP_TIME='1', SGLANG_RAGGED_VERIFY_MODE='static',
        SGLANG_DFLASH_FUSE_CONV='1', SGLANG_DSPARK_FOLDED_LATTICE='1')
    proc = thread = None
    stop = threading.Event()
    violations = []
    def status(**kw):
        payload = dict(arm='baseline', gpu=gpu, concurrency=1, updated_unix=time.time(),
            recovery_path=str(recovery), **kw)
        write(out / 'status.json', payload)
        write(recovery / 'status.json', payload)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        while True:
            state = gpu_state(gpu)
            if not state['pids'] and int(state['gpu'].split(',')[1]) < 512:
                break
            status(state='waiting_for_gpu', observed=state)
            time.sleep(2)
        with (out / 'baseline.server.log').open('a') as log:
            log.write('\nAR C=1 recovery: ' + str(recovery) + '\n')
            log.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        write(recovery / 'server.process.json', dict(pid=proc.pid, command=cmd))
        status(state='starting_server', pid=proc.pid)
        deadline = time.monotonic() + 600
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f'Server exited: {proc.returncode}')
            try:
                api(base, '/health', timeout=2)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise TimeoutError('Server startup')
                time.sleep(1)
        ready = gpu_state(gpu)
        if len(ready['pids']) != 1 or int(ready['processes'].split(',')[1]) < 74000:
            raise RuntimeError(f'GPU not exclusive/near-full: {ready}')
        write(recovery / 'gpu_ready.json', ready)
        def monitor():
            with (recovery / 'gpu_samples.jsonl').open('x', buffering=1) as log:
                while not stop.is_set():
                    try:
                        observed = gpu_state(gpu)
                        log.write(json.dumps(observed) + '\n')
                        if observed['pids'] != ready['pids']:
                            raise RuntimeError(f'GPU process set changed: {observed}')
                    except Exception as exc:
                        violations.append(repr(exc))
                        return
                    stop.wait(2)
        def guard():
            if violations:
                raise RuntimeError(violations[0])
            if proc.poll() is not None:
                raise RuntimeError(f'Server exited: {proc.returncode}')
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        write(recovery / 'before.json', api(base, '/get_server_info'))
        api(base, '/flush_cache')
        by_task = {n: [r for r in prepared if r['dataset'] == n] for n, _ in DATASETS}
        warmup = [by_task[n][i] for i in range(4) for n, _ in DATASETS][:32]
        for _ in range(2):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                warm = list(pool.map(lambda row: request(base, row, 64), warmup))
            summarize(warm, sum(r['wall_s'] for r in warm), 'baseline')
        api(base, '/flush_cache')
        write(recovery / 'measured_before.json', api(base, '/get_server_info'))
        status(state='measuring', pid=proc.pid)
        for name, count in remaining:
            verify_sources()
            guard()
            status(state='measuring', pid=proc.pid, dataset=name)
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                measured = list(pool.map(lambda row: request(base, row), by_task[name]))
            elapsed = time.perf_counter() - start
            guard()
            summary = summarize(measured, elapsed, 'baseline')
            for row in measured:
                row.update(arm='baseline', concurrency=1, temperature=0)
            # Store each recovered task independently before updating canonical files.
            with (recovery / f'{name}.jsonl').open('x') as fh:
                fh.writelines(json.dumps(row) + '\n' for row in measured)
            write(recovery / f'{name}.summary.json', summary)
            with (out / f'{prefix}.jsonl').open('a') as fh:
                fh.writelines(json.dumps(row) + '\n' for row in measured)
            summaries[name] = summary
            saved.extend(measured)
            write(out / f'{prefix}.summary.json', summaries)
            print(f'RECOVERED C=1 {name}: {count} rows {elapsed:.6f}s', flush=True)
        if validate_saved(saved, summaries, prepared):
            raise RuntimeError('Missing tasks after recovery')
        guard()
        after = api(base, '/get_server_info')
        write(recovery / 'after.json', after)
        write(out / f'{prefix}.after.json', dict(after, recovery_note=str(recovery)))
        guard()
        results = json.loads((out / 'results.json').read_text())
        results['baseline']['1'] = summaries
        write(out / 'results.json', results)
        write(out / f'{prefix}.complete.json', dict(complete=True, rows=len(saved),
            completed_unix=time.time(), concurrency=1, arm='baseline', recovery_path=str(recovery)))
        status(state='complete')
    except BaseException as exc:
        status(state='failed', error=repr(exc))
        raise
    finally:
        stop.set()
        if proc:
            stop_server(proc.pid)
        if thread:
            thread.join(timeout=5)


if __name__ == '__main__':
    main()
