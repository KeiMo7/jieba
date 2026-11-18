#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Concurrent benchmark for jieba tokenization.

Usage:
    python test/bench_concurrent.py              # full report
    python test/bench_concurrent.py throughput   # 1 vs 8 threads, same corpus
    python test/bench_concurrent.py stability    # repeated concurrent runs
    python test/bench_concurrent.py memory       # shared vs per-request instance

The script only uses jieba's public API so it can run against both the
pre-change and post-change code for before/after comparison.
"""
from __future__ import print_function

import os
import resource
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jieba

jieba.setLogLevel(60)  # silence logs during benchmark

THREADS = 8


def load_corpus():
    """Short + long mixed corpus built from test/lyric.txt."""
    path = os.path.join(os.path.dirname(__file__), 'lyric.txt')
    with open(path, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    corpus = list(lines)
    for i in range(0, len(lines), 7):
        corpus.append('。'.join(lines[i:i + 7]))
    corpus.append('。'.join(lines))  # one long text
    return corpus


def peak_rss_mb():
    # Linux: ru_maxrss is in KiB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def digest(results):
    return [hash('/'.join(words)) for words in results]


def cut_corpus(tokenizer, corpus):
    return [tokenizer.lcut(s) for s in corpus]


def run_threads(worker, nthreads):
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(nthreads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def bench_throughput(rounds=30):
    corpus = load_corpus()
    print('[throughput] corpus: %d texts (short+long mixed), threads: 1 vs %d'
          % (len(corpus), THREADS))

    # single thread
    t0 = time.perf_counter()
    for _ in range(rounds):
        expected = cut_corpus(jieba, corpus)
    dt1 = time.perf_counter() - t0
    qps1 = len(corpus) * rounds / dt1

    # multi thread, same total load, results collected per round
    mismatch = [0]
    lock = threading.Lock()

    def worker(idx):
        nonlocal expected
        local = []
        for _ in range(rounds):
            got = cut_corpus(jieba, corpus)
            if digest(got) != digest(expected):
                with lock:
                    mismatch[0] += 1
            local.append(got)
        return local

    t0 = time.perf_counter()
    run_threads(worker, THREADS)
    dt8 = time.perf_counter() - t0
    qps8 = len(corpus) * rounds * THREADS / dt8

    print('[throughput] single-thread : %8.1f sentences/s (%d sentences in %.2fs)'
          % (qps1, len(corpus) * rounds, dt1))
    print('[throughput] %-2d threads   : %8.1f sentences/s (%d sentences in %.2fs), speedup x%.2f'
          % (THREADS, qps8, len(corpus) * rounds * THREADS, dt8, qps8 / qps1))
    print('[throughput] result identical to single-thread baseline: %s (mismatched rounds: %d)'
          % (mismatch[0] == 0, mismatch[0]))
    return mismatch[0] == 0


def bench_stability(rounds=5):
    corpus = load_corpus()
    expected = digest(cut_corpus(jieba, corpus))
    bad = [0]
    lock = threading.Lock()

    def worker(idx):
        for _ in range(rounds):
            got = digest(cut_corpus(jieba, corpus))
            if got != expected:
                with lock:
                    bad[0] += 1

    run_threads(worker, THREADS)
    total = rounds * THREADS
    print('[stability] %d texts x %d rounds x %d threads: %d/%d runs identical'
          % (len(corpus), rounds, THREADS, total - bad[0], total))
    return bad[0] == 0


def _memory_load_shared(requests_per_thread):
    corpus = load_corpus()[:10]
    jieba.lcut('warm up 预热')

    def worker(idx):
        for _ in range(requests_per_thread):
            for s in corpus:
                jieba.lcut(s)

    run_threads(worker, THREADS)


def _memory_load_per_request(requests_per_thread):
    corpus = load_corpus()[:10]

    def worker(idx):
        for _ in range(requests_per_thread):
            tk = jieba.Tokenizer()
            tk.initialize()
            for s in corpus:
                tk.lcut(s)

    run_threads(worker, THREADS)


def bench_memory(requests_per_thread=6):
    here = os.path.abspath(__file__)
    print('[memory] same load: %d threads x %d requests x 10 sentences, threads=%d'
          % (THREADS, requests_per_thread, THREADS))
    results = {}
    for mode, fn in (('shared', _memory_load_shared),
                     ('per-request', _memory_load_per_request)):
        code = (
            "import sys, resource; sys.argv=['x','%s-worker']; __file__=%r;\n"
            "exec(open(%r).read())\n" % (mode, here, here)
        )
        out = subprocess.run([sys.executable, '-c', code],
                             capture_output=True, text=True)
        for line in out.stdout.splitlines():
            if line.startswith('PEAK_RSS_MB='):
                results[mode] = float(line.split('=')[1])
        if mode not in results:
            print(out.stdout)
            print(out.stderr, file=sys.stderr)
            raise RuntimeError('memory worker failed for mode %s' % mode)
    print('[memory] peak RSS shared single instance : %8.1f MB' % results['shared'])
    print('[memory] peak RSS per-request instances  : %8.1f MB' % results['per-request'])
    print('[memory] per-request overhead            : %+.1f MB (x%.2f)'
          % (results['per-request'] - results['shared'],
             results['per-request'] / max(results['shared'], 1e-9)))
    return True


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode == 'shared-worker':
        _memory_load_shared(6)
        print('PEAK_RSS_MB=%.1f' % peak_rss_mb())
        return 0
    if mode == 'per-request-worker':
        _memory_load_per_request(6)
        print('PEAK_RSS_MB=%.1f' % peak_rss_mb())
        return 0

    ok = True
    if mode in ('all', 'throughput'):
        ok = bench_throughput() and ok
    if mode in ('all', 'stability'):
        ok = bench_stability() and ok
    if mode in ('all', 'memory'):
        ok = bench_memory() and ok
    print('RESULT: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
