#encoding=utf-8
"""
词典版本记录 / 差异 / 回放 的验收测试。

运行方式（仓库根目录或 test 目录下均可）：
    python -m unittest test_dicttrack -v
"""
from __future__ import print_function, unicode_literals

import os
import shutil
import sys
import tempfile
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import jieba
from jieba import dicttrack


def write_file(path, content):
    with open(path, "wb") as f:
        f.write(content.encode("utf-8"))


class DictTrackTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jieba_dicttrack_test_")
        self.store = os.path.join(self.tmp, "store")
        # 每个用例一个独立的分词器，避免污染全局 dt
        self.tok = jieba.Tokenizer(track_versions=True,
                                   version_store_dir=self.store)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_dict(self, name, content):
        path = os.path.join(self.tmp, name)
        write_file(path, content)
        return path


class TestNormalRecording(DictTrackTestBase):
    """用例一：正常记录——来源、条目数、内容摘要、加载时间俱全。"""

    def test_load_records(self):
        d1 = self.make_dict("d1.txt", "云计算 5\n李小福 2 nr\n")
        self.tok.load_userdict(d1)
        info = self.tok.version_info()
        self.assertTrue(info["tracking"])
        self.assertIsNotNone(info["version"])
        kinds = [r["kind"] for r in info["loads"]]
        self.assertEqual(kinds, ["main_dict", "userdict"])
        rec = info["loads"][-1]
        self.assertEqual(rec["source"], d1)
        self.assertEqual(rec["entry_count"], 2)
        self.assertTrue(rec["digest"])
        self.assertEqual(rec["digest_status"], "ok")
        self.assertGreater(rec["time"], 0)
        # 快照已按内容寻址落盘
        self.assertTrue(os.path.isfile(
            self.tok._tracker.store.snapshot_path(rec["digest"])))

    def test_result_carries_version(self):
        d1 = self.make_dict("d1.txt", "云计算 5\n")
        self.tok.load_userdict(d1)
        out = self.tok.lcut_with_version("云计算方面的专家")
        self.assertEqual(out["dict_version"], self.tok.current_dict_version())
        self.assertIn("云计算", out["words"])

    def test_digest_depends_on_content_not_mtime(self):
        # 内容相同、mtime 不同 -> 摘要相同；内容变化 -> 摘要变化
        d1 = self.make_dict("same.txt", "云计算 5\n李小福 2 nr\n")
        d2 = self.make_dict("same2.txt", "云计算 5\n李小福 2 nr\n")
        os.utime(d1, (1000000000, 1000000000))
        os.utime(d2, (1600000000, 1600000000))
        e1 = dicttrack.parse_dict_file(d1)
        e2 = dicttrack.parse_dict_file(d2)
        self.assertEqual(dicttrack.compute_entries_digest(e1),
                         dicttrack.compute_entries_digest(e2))
        d3 = self.make_dict("diff.txt", "云计算 6\n李小福 2 nr\n")
        e3 = dicttrack.parse_dict_file(d3)
        self.assertNotEqual(dicttrack.compute_entries_digest(e1),
                            dicttrack.compute_entries_digest(e3))


class TestDiff(DictTrackTestBase):
    """用例二：两次词典的差异——新增、删除、词频从多少到多少。"""

    def test_diff_two_dicts(self):
        d1 = self.make_dict("v1.txt", "云计算 5\n李小福 2 nr\n创新办 3 i\n")
        d2 = self.make_dict("v2.txt", "云计算 9\n李小福 2 nr\n韩玉赏鉴 3 nz\n")
        diff = jieba.dict_diff_files(d1, d2)
        self.assertEqual(diff["summary"], {"added": 1, "removed": 1, "changed": 1})
        self.assertIn("韩玉赏鉴", diff["added"])
        self.assertIn("创新办", diff["removed"])
        chg = diff["changed"]["云计算"]
        self.assertEqual(chg["from"][0], 5)
        self.assertEqual(chg["to"][0], 9)

    def test_diff_two_versions(self):
        d1 = self.make_dict("v1.txt", "云计算 5\n")
        d2 = self.make_dict("v2.txt", "云计算 9\n新词甲 7\n")
        self.tok.load_userdict(d1)
        v1 = self.tok.current_dict_version()
        self.tok.load_userdict(d2)
        v2 = self.tok.current_dict_version()
        diff = jieba.dict_diff(v1, v2, store_dir=self.store)
        self.assertIn("新词甲", diff["added"])
        self.assertEqual(diff["changed"]["云计算"]["from"][0], 5)
        self.assertEqual(diff["changed"]["云计算"]["to"][0], 9)


class TestConflict(DictTrackTestBase):
    """用例三：多词典叠加时冲突条目能检测并报出最终生效值。"""

    def test_conflict_reports_effective_value(self):
        d1 = self.make_dict("a.txt", "云计算 5\n")
        d2 = self.make_dict("b.txt", "云计算 500\n")
        self.tok.load_userdict(d1)
        self.tok.load_userdict(d2)
        rec = self.tok.load_history()[-1]
        self.assertEqual(rec["conflict_count"], 1)
        conflict = rec["conflicts"][0]
        self.assertEqual(conflict["word"], "云计算")
        self.assertEqual(conflict["previous_freq"], 5)
        self.assertEqual(conflict["incoming_freq"], 500)
        # 最终生效值：后加载覆盖先加载
        self.assertEqual(conflict["effective_freq"], 500)
        self.assertEqual(self.tok.FREQ["云计算"], 500)


class TestReplay(DictTrackTestBase):
    """用例四：给定历史版本，重放当时的分词结果。"""

    SENT = "李小福是创新办主任也是云计算方面的专家"

    def test_replay_historical_result(self):
        d1 = self.make_dict("v1.txt", "云计算 5\n李小福 2 nr\n创新办 3 i\n")
        d2 = self.make_dict("v2.txt", "云计算 5000000\n")
        self.tok.load_userdict(d1)
        v1 = self.tok.current_dict_version()
        r1 = self.tok.lcut(self.SENT)
        self.tok.load_userdict(d2)
        v2 = self.tok.current_dict_version()
        self.assertNotEqual(v1, v2)
        # 用 v1 版本重放，结果与当时一致
        replayed = jieba.replay_lcut(self.SENT, v1, store_dir=self.store)
        self.assertEqual(replayed, r1)

    def test_replay_missing_version(self):
        with self.assertRaises(jieba.DictVersionNotFoundError) as ctx:
            jieba.replay_lcut(self.SENT, "0" * 32, store_dir=self.store)
        # 报错信息里要给出可行的替代做法
        self.assertIn("可行做法", str(ctx.exception))


class TestErrorsAndDegradation(DictTrackTestBase):
    def test_format_error_points_line_and_rejects(self):
        bad = self.make_dict("bad.txt", "甲乙丙丁戊 5\n新词 3.5\n甲乙丙丁己 2\n")
        self.tok.check_initialized()  # 先让主词典加载记录落账
        n0 = len(self.tok.load_history())
        with self.assertRaises(jieba.DictFormatError) as ctx:
            self.tok.load_userdict(bad)
        self.assertEqual(ctx.exception.lineno, 2)
        # 拒绝加载是原子的：坏行前后的词条都不会被应用，也不产生记录
        self.assertNotIn("甲乙丙丁戊", self.tok.FREQ)
        self.assertNotIn("甲乙丙丁己", self.tok.FREQ)
        self.assertEqual(len(self.tok.load_history()), n0)

    def test_format_error_too_many_fields(self):
        bad = self.make_dict("bad2.txt", "新词 10 nz extra\n")
        with self.assertRaises(jieba.DictFormatError):
            self.tok.load_userdict(bad)

    def test_digest_unknown_when_too_large(self):
        old_limit = dicttrack.DIGEST_MAX_ENTRIES
        dicttrack.DIGEST_MAX_ENTRIES = 1
        try:
            d1 = self.make_dict("big.txt", "云计算 5\n李小福 2 nr\n")
            self.tok.load_userdict(d1)
            rec = self.tok.load_history()[-1]
            self.assertIsNone(rec["digest"])
            self.assertEqual(rec["digest_status"], "unknown")
        finally:
            dicttrack.DIGEST_MAX_ENTRIES = old_limit


class TestTrackingOffUnchanged(DictTrackTestBase):
    """不开启记录时行为与现在一致；开启与否分词结果一致。"""

    SENT = "李小福是创新办主任也是云计算方面的专家"

    def test_off_by_default(self):
        tok = jieba.Tokenizer()
        self.assertIsNone(tok.current_dict_version())
        self.assertEqual(tok.load_history(), [])
        d1 = self.make_dict("d1.txt", "云计算 5\n李小福 2 nr\n创新办 3 i\n")
        tok.load_userdict(d1)  # 不报错、不记录
        self.assertEqual(tok.load_history(), [])

    def test_same_result_with_and_without_tracking(self):
        d1 = self.make_dict("d1.txt", "云计算 5\n李小福 2 nr\n创新办 3 i\n")
        plain = jieba.Tokenizer()
        plain.load_userdict(d1)
        tracked = jieba.Tokenizer(track_versions=True,
                                  version_store_dir=self.store)
        tracked.load_userdict(d1)
        self.assertEqual(plain.lcut(self.SENT), tracked.lcut(self.SENT))
        self.assertEqual(plain.lcut_for_search(self.SENT),
                         tracked.lcut_for_search(self.SENT))

    def test_add_word_and_del_word_recorded(self):
        self.tok.check_initialized()
        n0 = len(self.tok.load_history())
        self.tok.add_word("测试词甲", 100)
        self.tok.del_word("测试词甲")
        kinds = [r["kind"] for r in self.tok.load_history()[n0:]]
        self.assertEqual(kinds, ["add_word", "add_word"])
        self.assertEqual(self.tok.load_history()[-1]["freq"], 0)


if __name__ == "__main__":
    unittest.main()
