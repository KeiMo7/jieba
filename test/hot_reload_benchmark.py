# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals

import gc
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jieba


BASE_DICTIONARY = (
    "我 1000000\n"
    "爱 1000000\n"
    "热 10000\n"
    "加 10000\n"
    "载 10000\n"
    "批 10000\n"
    "词 10000\n"
    "号 10000\n"
    "新 10000\n"
    "品 10000\n"
    "活 10000\n"
    "动 10000\n"
)


def percentile(values, point):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * point))]


def measure_cut(tokenizer, prefix, count=300, rounds=3):
    p50_values = []
    p95_values = []
    max_values = []
    for _ in range(rounds):
        values = []
        for index in range(count):
            sentence = '%s编号%s' % (prefix, index)
            started = time.perf_counter()
            tokenizer.lcut(sentence, HMM=False)
            values.append((time.perf_counter() - started) * 1000.0)
        p50_values.append(statistics.median(values))
        p95_values.append(percentile(values, 0.95))
        max_values.append(max(values))
    return {
        'p50_ms': statistics.median(p50_values),
        'p95_ms': statistics.median(p95_values),
        'max_ms': max(max_values),
    }


def main():
    tmpdir = tempfile.mkdtemp(prefix='jieba-hot-reload-bench-')
    base_path = os.path.join(tmpdir, 'base.txt')
    user_path = os.path.join(tmpdir, 'user.txt')
    try:
        with open(base_path, 'wb') as target:
            target.write(BASE_DICTIONARY.encode('utf-8'))
        with open(user_path, 'wb') as target:
            target.write(b'')

        tokenizer = jieba.Tokenizer(base_path)
        tokenizer.tmp_dir = tmpdir
        info = tokenizer.get_dict_info()
        print('initial: version=%s entries=%s' %
              (info['version'], info['entry_count']))

        # Warm the bounded result cache, then show only dependent entries are
        # removed when the new words are published.
        for index in range(200):
            tokenizer.lcut('我爱热加载编号%s' % index, HMM=False)
        for index in range(200):
            tokenizer.lcut('我爱活动编号%s' % index, HMM=False)

        before_latency = measure_cut(tokenizer, '我爱活动')
        batch = ['热加载', '新品活动', '秒级生效', '原子切换']
        with open(user_path, 'wb') as target:
            target.write(''.join(
                '%s 50000\n' % word for word in batch).encode('utf-8'))
        result = tokenizer.reload_userdict(user_path)
        after_latency = measure_cut(tokenizer, '我爱活动')
        print('hot_reload: version %s -> %s entries %s -> %s delta=%s '
              'cache_invalidated=%s duration_ms=%.3f switch_ms=%.3f' % (
                  result['version_before'], result['version_after'],
                  result['entry_count_before'], result['entry_count_after'],
                  result['entry_count_delta'],
                  result['invalidated_cache_entries'],
                  result['duration_seconds'] * 1000.0,
                  result['switch_seconds'] * 1000.0))
        print('latency_after_hot_reload: before p50=%.4f p95=%.4f '
              'after p50=%.4f p95=%.4f' % (
                  before_latency['p50_ms'], before_latency['p95_ms'],
                  after_latency['p50_ms'], after_latency['p95_ms']))

        # Keep a request stream running while a larger next batch is parsed.
        large_path = os.path.join(tmpdir, 'large.txt')
        with open(large_path, 'wb') as target:
            for index in range(30000):
                target.write(('扩容词%s 50000\n' % index).encode('utf-8'))
        request_latencies = []
        stop = threading.Event()

        def request_worker():
            counter = 0
            while not stop.is_set():
                started = time.perf_counter()
                tokenizer.lcut('我爱活动编号%s' % counter, HMM=False)
                request_latencies.append(
                    (time.perf_counter() - started) * 1000.0)
                counter += 1

        threads = [threading.Thread(target=request_worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        concurrent_load = tokenizer.reload_userdict(large_path)
        stop.set()
        for thread in threads:
            thread.join()
        print('concurrent_load: duration_ms=%.3f switch_ms=%.3f '
              'requests=%s max_request_ms=%.4f p95_request_ms=%.4f' % (
                  concurrent_load['duration_seconds'] * 1000.0,
                  concurrent_load['switch_seconds'] * 1000.0,
                  len(request_latencies), max(request_latencies),
                  percentile(request_latencies, 0.95)))

        # Fill the bounded cache before the memory baseline so the five-batch
        # measurement isolates dictionary revisions rather than cache warm-up.
        for index in range(1500):
            tokenizer.lcut('我爱活动预热门%s' % index, HMM=False)
        gc.collect()
        tracemalloc.start()
        baseline_memory = tracemalloc.get_traced_memory()[0]
        all_lines = []
        batch_memories = []
        batch_p95 = []
        for batch_index in range(5):
            words = ['连%s批%s号' % (batch_index, index) for index in range(80)]
            all_lines.extend('%s 50000\n' % word for word in words)
            with open(user_path, 'wb') as target:
                target.write(''.join(all_lines).encode('utf-8'))
            result = tokenizer.reload_userdict(user_path)
            latency = measure_cut(
                tokenizer, '我爱连%s批79号' % batch_index, 200)
            gc.collect()
            memory = tracemalloc.get_traced_memory()[0]
            batch_memories.append(memory - baseline_memory)
            batch_p95.append(latency['p95_ms'])
            print('batch_%s: version=%s entries=%s delta=%s memory_kb=%.1f '
                  'p95_ms=%.4f' % (
                      batch_index + 1, result['version'],
                      result['entry_count_after'],
                      result['entry_count_delta'],
                      (memory - baseline_memory) / 1024.0,
                      latency['p95_ms']))
        print('five_batch_summary: memory_after_stabilization_kb=%.1f '
              'p95_first_ms=%.4f p95_last_ms=%.4f' % (
                  (batch_memories[-1] - batch_memories[1]) / 1024.0,
                  batch_p95[0], batch_p95[-1]))
        tracemalloc.stop()
    finally:
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    main()
