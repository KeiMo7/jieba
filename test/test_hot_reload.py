# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import gc
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
import tracemalloc
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jieba


BASE_DICTIONARY = (
    "我 100000\n"
    "爱 100000\n"
    "热 10000\n"
    "加 10000\n"
    "载 10000\n"
    "新 10000\n"
    "词 10000\n"
    "组 10000\n"
    "旧 10000\n"
    "版 10000\n"
    "本 10000\n"
    "北 10000\n"
    "京 10000\n"
    "北京 100000\n"
    "今 10000\n"
    "天 10000\n"
    "气 10000\n"
    "不 10000\n"
    "错 10000\n"
)


class SlowDictionaryFile(object):

    def __init__(self, lines, ready, delay=0.08):
        self.lines = lines
        self.ready = ready
        self.delay = delay

    def __iter__(self):
        for index, line in enumerate(self.lines):
            yield line
            if index == 0:
                self.ready.set()
            time.sleep(self.delay)


class HotReloadTestCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='jieba-hot-reload-')
        self.base_path = os.path.join(self.tmpdir, 'base.txt')
        self.user_path = os.path.join(self.tmpdir, 'user.txt')
        with open(self.base_path, 'wb') as target:
            target.write(BASE_DICTIONARY.encode('utf-8'))
        self.tokenizer = jieba.Tokenizer(self.base_path)
        self.tokenizer.tmp_dir = self.tmpdir
        self.tokenizer.initialize()

    def tearDown(self):
        self.tokenizer.disable_auto_reload()
        shutil.rmtree(self.tmpdir)

    def write_userdict(self, content):
        mode = 'ab' if os.path.exists(self.user_path) else 'wb'
        with open(self.user_path, mode) as target:
            target.write(content.encode('utf-8'))

    def test_incremental_load_atomic_observability_and_cache_invalidation(self):
        old_version = self.tokenizer.get_dict_info()['version']
        old_entries = self.tokenizer.get_dict_info()['entry_count']
        self.assertEqual(
            self.tokenizer.lcut('我爱热加载新词组', HMM=False),
            ['我', '爱', '热', '加', '载', '新', '词', '组'])
        self.tokenizer.lcut('我爱北京', HMM=False)
        self.assertGreater(self.tokenizer.get_dict_info()['cache_entries'], 0)

        request_started_before_reload = self.tokenizer.cut(
            '我爱热加载', HMM=False)
        self.write_userdict('热加载 50000\n新词组 50000\n')
        result = self.tokenizer.reload_userdict(self.user_path)

        self.assertEqual(result['old_version'], old_version)
        self.assertEqual(result['version'], old_version + 1)
        self.assertEqual(result['added'], 2)
        self.assertGreaterEqual(result['entry_count_delta'], 2)
        self.assertGreaterEqual(result['invalidated_cache_entries'], 1)
        self.assertEqual(result['cache_entries'], 1)
        self.assertEqual(
            list(request_started_before_reload),
            ['我', '爱', '热', '加', '载'])
        self.assertEqual(
            self.tokenizer.lcut('我爱热加载新词组', HMM=False),
            ['我', '爱', '热加载', '新词组'])
        self.assertEqual(
            self.tokenizer.lcut('我爱北京', HMM=False),
            ['我', '爱', '北京'])

        info = self.tokenizer.get_dict_info()
        self.assertEqual(info['version'], old_version + 1)
        self.assertEqual(
            info['entry_count'], old_entries + result['entry_count_delta'])
        self.assertGreater(info['last_load_time'], 0)
        self.assertEqual(info['last_load']['added'], 2)

    def test_concurrent_cut_uses_old_or_new_snapshot_without_blocking(self):
        ready = threading.Event()
        lines = [
            '热加载 50000\n',
            '新词组 50000\n',
            '并发词 50000\n',
            '原子切换 50000\n',
            '无半成品 50000\n',
            '完整快照 50000\n',
        ]
        slow_file = SlowDictionaryFile(lines, ready, delay=0.08)
        errors = []

        def reload_worker():
            try:
                self.tokenizer.reload_userdict(slow_file, force=True)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=reload_worker)
        thread.start()
        self.assertTrue(ready.wait(2.0))

        old_result = ('我', '爱', '热', '加', '载', '新', '词', '组')
        new_result = ('我', '爱', '热加载', '新词组')
        observed = set()
        latencies = []
        deadline = time.time() + 0.45
        while time.time() < deadline:
            started = time.time()
            observed.add(tuple(self.tokenizer.lcut(
                '我爱热加载新词组', HMM=False)))
            latencies.append(time.time() - started)

        thread.join(2.0)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertLess(max(latencies), 0.25)
        self.assertTrue(observed.issubset(set([old_result, new_result])))
        self.assertEqual(
            self.tokenizer.lcut('我爱热加载新词组', HMM=False),
            list(new_result))

    def test_format_error_rejects_whole_file_and_keeps_old_dictionary(self):
        self.write_userdict('热加载 50000\n')
        self.tokenizer.reload_userdict(self.user_path)
        before = self.tokenizer.get_dict_info()
        before_cut = self.tokenizer.lcut('我爱热加载', HMM=False)

        with open(self.user_path, 'ab') as target:
            target.write(b'\xff\xfe')
        with self.assertRaises(jieba.DictionaryLoadError) as context:
            self.tokenizer.reload_userdict(self.user_path, force=True)

        self.assertEqual(context.exception.line_number, 2)
        self.assertIsNotNone(context.exception.entry)
        after = self.tokenizer.get_dict_info()
        self.assertEqual(after['version'], before['version'])
        self.assertEqual(after['entry_count'], before['entry_count'])
        self.assertIn('Line 2', after['last_error']['message'])
        self.assertEqual(
            self.tokenizer.lcut('我爱热加载', HMM=False), before_cut)

        with open(self.user_path, 'wb') as target:
            target.write('热加载 50000\n坏词 -1\n'.encode('utf-8'))
        with self.assertRaises(jieba.DictionaryLoadError) as context:
            self.tokenizer.reload_userdict(self.user_path, force=True)
        self.assertEqual(context.exception.line_number, 2)
        self.assertIn('坏词 -1', context.exception.entry)
        self.assertEqual(
            self.tokenizer.get_dict_info()['version'], before['version'])

    def test_empty_file_is_no_change_and_duplicates_last_wins(self):
        self.write_userdict('')
        result = self.tokenizer.reload_userdict(self.user_path)
        self.assertEqual(result['status'], 'no_change')
        old_version = result['version']

        with open(self.user_path, 'wb') as target:
            target.write('重复词 1\n重复词 9\n'.encode('utf-8'))
        result = self.tokenizer.reload_userdict(self.user_path)
        self.assertEqual(result['status'], 'loaded')
        self.assertEqual(result['duplicates'], 1)
        self.assertEqual(result['version'], old_version + 1)
        self.assertEqual(self.tokenizer.FREQ.get('重复词'), 9)

    def test_rollback_restores_previous_version_and_cache(self):
        old_version = self.tokenizer.get_dict_info()['version']
        self.write_userdict('热加载 50000\n')
        self.tokenizer.reload_userdict(self.user_path)
        self.assertEqual(
            self.tokenizer.lcut('我爱热加载', HMM=False),
            ['我', '爱', '热加载'])

        result = self.tokenizer.rollback()
        self.assertEqual(result['version'], old_version)
        self.assertEqual(
            self.tokenizer.lcut('我爱热加载', HMM=False),
            ['我', '爱', '热', '加', '载'])
        self.assertFalse(self.tokenizer.get_dict_info()['can_rollback'])
        with self.assertRaises(ValueError):
            self.tokenizer.rollback()

    def test_concurrent_reload_triggers_are_serialized(self):
        self.write_userdict('热加载 50000\n新词组 50000\n')
        results = []
        errors = []
        barrier = threading.Barrier(8)

        def worker():
            try:
                barrier.wait()
                results.append(self.tokenizer.reload_userdict(
                    self.user_path, force=True))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5.0)

        self.assertFalse(errors)
        self.assertEqual(len(results), 8)
        self.assertEqual(
            sorted(result['status'] for result in results),
            ['loaded'] + ['no_change'] * 7)
        self.assertEqual(self.tokenizer.get_dict_info()['version'], 2)

    def test_auto_reload_is_configurable_and_uses_same_reload_path(self):
        self.write_userdict('热加载 50000\n')
        self.tokenizer.reload_userdict(self.user_path)
        self.tokenizer.enable_auto_reload(self.user_path, interval=0.05)
        try:
            self.write_userdict('新词组 50000\n')
            deadline = time.time() + 3.0
            found = False
            while time.time() < deadline:
                result = self.tokenizer.lcut(
                    '我爱热加载新词组', HMM=False)
                if result == ['我', '爱', '热加载', '新词组']:
                    found = True
                    break
                time.sleep(0.02)
            self.assertTrue(found)
            info = self.tokenizer.get_dict_info()
            self.assertTrue(info['auto_reload_enabled'])
            self.assertEqual(info['auto_reload_interval'], 0.05)
            self.assertEqual(info['version'], 3)
        finally:
            self.tokenizer.disable_auto_reload()

    def test_five_incremental_batches_keep_memory_and_latency_bounded(self):
        gc.collect()
        tracemalloc.start()
        baseline_memory = tracemalloc.get_traced_memory()[0]
        self.tokenizer.set_cache_size(256)

        sentences = []
        userdict_lines = []
        for batch in range(5):
            words = []
            for item in range(60):
                word = '批%s词%s号' % (batch, item)
                words.append(word)
                userdict_lines.append('%s 50000\n' % word)
            if batch == 0:
                self.write_userdict(''.join(userdict_lines))
            else:
                self.write_userdict(''.join(
                    '%s 50000\n' % word for word in words))
            result = self.tokenizer.reload_userdict(self.user_path)
            self.assertEqual(result['version'], batch + 2)
            sentence = '我爱%s' % words[-1]
            sentences.append(sentence)
            self.assertIn(words[-1], self.tokenizer.lcut(
                sentence, HMM=False))

        latencies = []
        for index in range(500):
            sentence = '我爱批4词59号编号%s' % index
            started = time.perf_counter()
            self.tokenizer.lcut(sentence, HMM=False)
            latencies.append(time.perf_counter() - started)
        stats = {
            'p50': statistics.median(latencies),
            'p95': sorted(latencies)[int(len(latencies) * 0.95)],
        }
        gc.collect()
        current_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        self.assertLess(current_memory - baseline_memory, 8 * 1024 * 1024)
        self.assertLess(stats['p95'], 0.05)
        self.assertLessEqual(
            self.tokenizer.get_dict_info()['cache_entries'], 256)


if __name__ == '__main__':
    unittest.main()
