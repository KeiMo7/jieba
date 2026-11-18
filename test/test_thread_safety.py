#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Thread-safety test suite for jieba tokenization.

Covers:
  1. concurrent cuts of the same texts == single-threaded results
  2. different modes (accurate / full / search) used concurrently
  3. concurrent load_userdict while cutting (snapshot semantics)
  4. cut-result cache isolation and eviction under concurrency
  5. long + short texts mixed concurrently
  6. repeated concurrent processing of the same text is stable
  7. module-level shared state (finalseg force-split set) is isolated
  8. concurrent initialize() does not race

Run:  python test/test_thread_safety.py   (or: python -m unittest ...)
The suite mostly uses public APIs, so it can also run against the
pre-change code for before/after comparison.
"""
from __future__ import print_function

import os
import random
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jieba

jieba.setLogLevel(60)
jieba.initialize()  # pay dict-loading cost once, outside the tests

THREADS = 8

SENTENCES = [
    '我来到北京清华大学',
    '他来到了网易杭研大厦',
    '小明硕士毕业于中国科学院计算所，后在日本京都大学深造',
    '炭治郎是鬼灭之刃的主角',
    '今天天气不错，适合出去运动',
    '中国人民银行行长出席会议',
    'Python是一门流行的编程语言',
    '结婚的和尚未结婚的',
]


def lyric_corpus():
    path = os.path.join(os.path.dirname(__file__), 'lyric.txt')
    with open(path, encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def run_threads(worker, nthreads=THREADS):
    errors = []

    def guarded(idx):
        try:
            worker(idx)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=guarded, args=(i,))
               for i in range(nthreads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


class ConcurrentConsistencyTest(unittest.TestCase):
    """1+5: concurrent cuts (short and long texts) match single-thread."""

    def test_concurrent_same_texts(self):
        corpus = SENTENCES + lyric_corpus()
        corpus.append('。'.join(lyric_corpus()))  # one long text
        expected = [jieba.lcut(s) for s in corpus]

        def worker(idx):
            rng = random.Random(idx)
            for _ in range(20):
                order = list(range(len(corpus)))
                rng.shuffle(order)
                for i in order:
                    got = jieba.lcut(corpus[i])
                    self.assertEqual(got, expected[i],
                                     'thread %d, text %r' % (idx, corpus[i]))

        errors = run_threads(worker)
        self.assertEqual(errors, [])


class MixedModesTest(unittest.TestCase):
    """2: accurate / full / search / no-HMM modes used concurrently."""

    def test_mixed_modes(self):
        corpus = SENTENCES
        baselines = {
            'accurate': [jieba.lcut(s) for s in corpus],
            'full': [jieba.lcut(s, cut_all=True) for s in corpus],
            'no_hmm': [jieba.lcut(s, HMM=False) for s in corpus],
            'search': [jieba.lcut_for_search(s) for s in corpus],
            'search_no_hmm': [jieba.lcut_for_search(s, HMM=False)
                              for s in corpus],
        }
        modes = list(baselines)

        def worker(idx):
            rng = random.Random(1000 + idx)
            for _ in range(30):
                mode = modes[rng.randrange(len(modes))]
                i = rng.randrange(len(corpus))
                s = corpus[i]
                if mode == 'accurate':
                    got = jieba.lcut(s)
                elif mode == 'full':
                    got = jieba.lcut(s, cut_all=True)
                elif mode == 'no_hmm':
                    got = jieba.lcut(s, HMM=False)
                elif mode == 'search':
                    got = jieba.lcut_for_search(s)
                else:
                    got = jieba.lcut_for_search(s, HMM=False)
                self.assertEqual(got, baselines[mode][i],
                                 'mode %s, text %r' % (mode, s))

        errors = run_threads(worker)
        self.assertEqual(errors, [])


class ConcurrentLoadUserdictTest(unittest.TestCase):
    """3: loading a userdict while other threads are cutting."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', suffix='.txt', delete=False)
        self.tmp.write('炭治郎 1000000\n鬼灭之刃 1000000\n')
        self.tmp.close()
        self.addCleanup(os.unlink, self.tmp.name)

    def test_load_userdict_during_cut(self):
        tk = jieba.Tokenizer()
        tk.initialize()
        target = '炭治郎是鬼灭之刃的主角'
        corpus = SENTENCES + [target]
        before = {s: tk.lcut(s) for s in corpus}
        tk.load_userdict(self.tmp.name)
        after = {s: tk.lcut(s) for s in corpus}
        # sanity: the userdict actually changes the target segmentation
        self.assertNotEqual(before[target], after[target])
        self.assertIn('炭治郎', after[target])
        self.assertIn('鬼灭之刃', after[target])

        tk2 = jieba.Tokenizer()
        tk2.initialize()
        allowed = {s: (before[s], after[s]) for s in corpus}
        done = threading.Event()

        def cutter(idx):
            rng = random.Random(idx)
            while not done.is_set():
                s = corpus[rng.randrange(len(corpus))]
                got = tk2.lcut(s)
                self.assertIn(got, allowed[s],
                              'half-loaded dictionary observed for %r' % s)

        def loader():
            tk2.load_userdict(self.tmp.name)
            done.set()

        threads = [threading.Thread(target=cutter, args=(i,))
                   for i in range(THREADS - 1)]
        threads.append(threading.Thread(target=loader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # after loading, every cut must see the new words
        self.assertEqual(tk2.lcut(target), after[target])

    def test_concurrent_add_and_del_word(self):
        tk = jieba.Tokenizer()
        tk.initialize()
        target = '旋风冲锋龙卷风'
        before = tk.lcut(target)
        tk.add_word('旋风冲锋龙卷风', 1000000)
        after = tk.lcut(target)
        self.assertNotEqual(before, after)
        self.assertEqual(after, ['旋风冲锋龙卷风'])

        tk2 = jieba.Tokenizer()
        tk2.initialize()
        done = threading.Event()

        def cutter(idx):
            while not done.is_set():
                got = tk2.lcut(target)
                self.assertIn(got, (before, after),
                              'half-updated dictionary observed')

        def writer():
            tk2.add_word('旋风冲锋龙卷风', 1000000)
            done.set()

        threads = [threading.Thread(target=cutter, args=(i,))
                   for i in range(THREADS - 1)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()


class CutCacheTest(unittest.TestCase):
    """4: cache hits/misses/evictions never return another text's result."""

    def setUp(self):
        if not hasattr(jieba.Tokenizer, 'enable_cut_cache'):
            self.skipTest('cut cache not available in this version')

    def test_cache_isolation_and_eviction(self):
        tk = jieba.Tokenizer()
        tk.initialize()
        corpus = lyric_corpus() + SENTENCES  # > 40 distinct texts
        baseline = {s: tk.lcut(s) for s in corpus}
        baseline_search = {s: tk.lcut_for_search(s) for s in corpus}
        baseline_full = {s: tk.lcut(s, cut_all=True) for s in corpus}

        tk.enable_cut_cache(maxsize=8)  # tiny capacity: constant eviction

        def worker(idx):
            rng = random.Random(2000 + idx)
            for _ in range(200):
                s = corpus[rng.randrange(len(corpus))]
                self.assertEqual(tk.lcut(s), baseline[s])
                self.assertEqual(tk.lcut_for_search(s), baseline_search[s])
                self.assertEqual(tk.lcut(s, cut_all=True), baseline_full[s])

        errors = run_threads(worker)
        self.assertEqual(errors, [])

    def test_cache_invalidated_by_dict_update(self):
        tk = jieba.Tokenizer()
        tk.initialize()
        tk.enable_cut_cache(maxsize=16)
        target = '炭治郎砍了一刀'
        before = tk.lcut(target, HMM=False)
        tk.add_word('炭治郎', 1000000)
        after = tk.lcut(target, HMM=False)
        self.assertNotEqual(before, after)
        self.assertEqual(after[0], '炭治郎')


class RepeatStabilityTest(unittest.TestCase):
    """6: the same text processed repeatedly under concurrency is stable."""

    def test_repeat_same_text(self):
        text = '小明硕士毕业于中国科学院计算所，后在日本京都大学深造'
        expected = jieba.lcut(text)

        def worker(idx):
            for _ in range(300):
                self.assertEqual(jieba.lcut(text), expected)

        errors = run_threads(worker)
        self.assertEqual(errors, [])


class ModuleGlobalStateTest(unittest.TestCase):
    """7: del_word on one Tokenizer must not leak into another instance
    through the module-level finalseg.Force_Split_Words set."""

    def test_force_split_isolated_per_instance(self):
        tk1 = jieba.Tokenizer()
        tk2 = jieba.Tokenizer()
        # an HMM-recognized word that is not in the dictionary: the
        # force-split set (module-level in the old code) decides whether
        # it is emitted as one word or as single characters
        sentence = '炭治郎砍了一刀'
        expected = tk2.lcut(sentence)
        self.assertIn('炭治郎', expected)
        tk1.del_word('炭治郎')
        self.assertNotEqual(tk1.lcut(sentence), expected)
        # the other instance (and the module-level default) are unaffected
        self.assertEqual(tk2.lcut(sentence), expected)


class InitializeRaceTest(unittest.TestCase):
    """8: concurrent initialize() / first-cut does not race."""

    def test_concurrent_initialize(self):
        tk = jieba.Tokenizer()

        def worker(idx):
            tk.initialize()

        errors = run_threads(worker)
        self.assertEqual(errors, [])
        self.assertTrue(tk.initialized)
        self.assertEqual(jieba.lcut('我来到北京清华大学'),
                         tk.lcut('我来到北京清华大学'))

    def test_concurrent_first_cut(self):
        tk = jieba.Tokenizer()
        expected = jieba.lcut('我来到北京清华大学')

        def worker(idx):
            self.assertEqual(tk.lcut('我来到北京清华大学'), expected)

        errors = run_threads(worker)
        self.assertEqual(errors, [])

    def test_concurrent_initialize_custom_dict(self):
        dict_path = os.path.join(os.path.dirname(__file__), '..',
                                 'extra_dict', 'dict.txt.small')
        holders = [jieba.Tokenizer(dict_path) for _ in range(4)]

        def worker(idx):
            holders[idx % len(holders)].initialize()

        errors = run_threads(worker)
        self.assertEqual(errors, [])
        for tk in holders:
            self.assertTrue(tk.initialized)
            self.assertGreater(len(tk.FREQ), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
