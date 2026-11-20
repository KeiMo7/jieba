# -*- coding: utf-8 -*-
"""
词典版本记录、差异对比与历史回放（默认关闭，需显式开启）。

设计要点：
* 摘要基于词典内容（规范化后的 词条/词频/词性 序列），与文件修改时间无关；
* 加载顺序即版本顺序：同一状态版本由有序的事件摘要链合成，
  同一词条后加载的覆盖先加载的（与 jieba 既有行为一致）；
* 快照按内容寻址（文件名即摘要）落盘，回放/对比时按版本号重建；
* 不开启记录时，本模块不会被触发，分词行为与性能与原来一致。
"""
from __future__ import absolute_import, unicode_literals

import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
import time

from ._compat import text_type, string_types, iteritems, strdecode, resolve_filename

logger = logging.getLogger(__name__)

# 条目数超过该阈值时不再计算内容摘要（防止超大词典拖慢加载），
# 对应版本标记为 unknown 并告警。
DIGEST_MAX_ENTRIES = 2000000

# 冲突列表写入历史文件时的最大条数（内存中的记录保留全量）。
MAX_CONFLICTS_IN_HISTORY = 10000

SNAPSHOT_FORMAT_VERSION = 1

# 与 jieba/__init__.py 中 re_userdict 保持一致（避免循环引用，在此复制）。
_re_userdict = re.compile('^(.+?)( [0-9]+)?( [a-z]+)?$', re.U)
# 词本身若以 " 数字" 或 " 小数" 结尾，说明词频位置很可能写错了（如 "新词 3.5"）。
_re_tail_numlike = re.compile(' [0-9.]+$')
# 已经解析出词性后，词仍以 " 小写字母" 结尾，说明一行里字段过多（如 "词 10 nz extra"）。
_re_tail_tag = re.compile(' [a-z]+$')


class DictFormatError(ValueError):
    """词典格式错误。携带文件名、行号、原始行与原因；抛出时本次加载被拒绝。"""

    def __init__(self, f_name, lineno, line, reason):
        self.f_name = f_name
        self.lineno = lineno
        self.line = line
        self.reason = reason
        super(DictFormatError, self).__init__(
            'invalid dictionary entry in %s at Line %s: %r (%s)'
            % (f_name, lineno, line, reason))


class DictVersionNotFoundError(Exception):
    """回放/对比所需的词典版本或快照不存在。"""
    pass


def default_store_dir():
    """版本快照与历史的默认存储目录，可用环境变量 JIEBA_DICT_VERSION_DIR 覆盖。"""
    return (os.environ.get('JIEBA_DICT_VERSION_DIR') or
            os.path.join(os.path.expanduser('~'), '.jieba_dict_versions'))


def compute_entries_digest(entries):
    """
    基于内容计算摘要。entries: dict, word -> (freq, tag)。

    摘要对 排序后的 "word\\tfreq\\ttag" 序列取 md5，因此：
    * 内容变则摘要变，内容不变（哪怕文件 mtime 不同）摘要不变；
    * 与条目在文件中的先后顺序无关（同一文件内后者覆盖前者，取生效值）。

    条目数超过 DIGEST_MAX_ENTRIES 时返回 None（版本未知，调用方应告警）。
    """
    if len(entries) > DIGEST_MAX_ENTRIES:
        return None
    h = hashlib.md5()
    for word in sorted(entries):
        freq, tag = entries[word]
        h.update(('%s\t%s\t%s\n' % (word, freq, tag or '')).encode('utf-8'))
    return h.hexdigest()


def combine_event_digests(digests):
    """把有序的事件摘要链合成一个状态版本号。加载顺序不同则版本不同。"""
    h = hashlib.md5()
    for d in digests:
        h.update((d or 'unknown').encode('ascii'))
        h.update(b'\n')
    return h.hexdigest()


def parse_userdict_line(line, f_name, lineno):
    """
    严格解析一行用户词典，返回 (word, freq_or_None, tag_or_None)。

    与历史行为（re_userdict 尽量匹配、异常静默吞掉）不同，严格模式下
    格式可疑的行会抛出 DictFormatError 并拒绝整个加载，调用方拿到行号与原因。
    """
    word, freq, tag = _re_userdict.match(line).groups()
    if not word:
        raise DictFormatError(f_name, lineno, line, 'empty word')
    if _re_tail_numlike.search(word):
        raise DictFormatError(
            f_name, lineno, line,
            'word ends with a number-like field; freq must be a plain integer, '
            'expected format: "word freq(optional integer) tag(optional)"')
    if tag is not None and _re_tail_tag.search(word):
        raise DictFormatError(
            f_name, lineno, line,
            'too many fields or misplaced tag, '
            'expected format: "word freq(optional integer) tag(optional)"')
    if freq is not None:
        freq = freq.strip()
        # 正则已保证 freq 全为数字，这里再兜底一次，拒绝而不是静默跳过。
        if not freq.isdigit():
            raise DictFormatError(f_name, lineno, line,
                                  'freq must be a non-negative integer')
    if tag is not None:
        tag = tag.strip()
    return word, freq, tag


def parse_dict_file(f, f_name=None):
    """严格解析整个用户词典文件，返回 dict: word -> (freq_or_None, tag)。"""
    if isinstance(f, string_types):
        f_name = f
        f = io.open(f, 'rb')
    if f_name is None:
        f_name = resolve_filename(f)
    entries = {}
    for lineno, ln in enumerate(f, 1):
        line = ln.strip()
        if not isinstance(line, text_type):
            try:
                line = line.decode('utf-8').lstrip('\ufeff')
            except UnicodeDecodeError:
                raise DictFormatError(f_name, lineno, '<binary data>',
                                      'dictionary file must be utf-8')
        if not line:
            continue
        word, freq, tag = parse_userdict_line(line, f_name, lineno)
        entries[word] = (int(freq) if freq is not None else None, tag)
    return entries


def diff_entries(old, new):
    """
    对比两份词条表（word -> (freq, tag)），返回新增/删除/词频与词性调整。

    changed 的值为 {'from': (freq, tag), 'to': (freq, tag)}，可看出词频从多少到多少。
    """
    old = old or {}
    new = new or {}
    added = dict((w, new[w]) for w in new if w not in old)
    removed = dict((w, old[w]) for w in old if w not in new)
    changed = dict(
        (w, {'from': old[w], 'to': new[w]})
        for w in old
        if w in new and old[w] != new[w])
    return {
        'added': added,
        'removed': removed,
        'changed': changed,
        'summary': {
            'added': len(added),
            'removed': len(removed),
            'changed': len(changed),
        },
    }


class VersionStore(object):
    """
    版本存储：快照按内容寻址保存为 snapshots/<digest>.json，
    加载历史追加写入 history.jsonl（单行 JSON，便于外部工具消费）。
    """

    def __init__(self, store_dir=None):
        self.dir = store_dir or default_store_dir()
        self.snap_dir = os.path.join(self.dir, 'snapshots')
        for d in (self.dir, self.snap_dir):
            if not os.path.isdir(d):
                try:
                    os.makedirs(d)
                except OSError:
                    # 并发创建时目录可能已存在
                    if not os.path.isdir(d):
                        raise
        self._hist_lock = threading.Lock()

    @property
    def history_path(self):
        return os.path.join(self.dir, 'history.jsonl')

    def snapshot_path(self, digest):
        return os.path.join(self.snap_dir, '%s.json' % digest)

    def save_snapshot(self, digest, entries):
        """entries: word -> (freq, tag)。内容寻址，同摘要只写一次；临时文件+改名保证原子性。"""
        return self.save_snapshot_items(
            digest, ((w, f, t) for w, (f, t) in iteritems(entries)),
            len(entries))

    def save_snapshot_items(self, digest, items, entry_count):
        """流式写快照：items 为 (word, freq, tag) 可迭代对象，避免在内存里再复制一份条目表。"""
        path = self.snapshot_path(digest)
        if os.path.exists(path):
            return path
        fd, tmp = tempfile.mkstemp(dir=self.snap_dir, prefix='.tmp-')
        try:
            with io.open(fd, 'w', encoding='utf-8') as fp:
                fp.write('{"format": %d, "digest": "%s", "entry_count": %d, "entries": ['
                         % (SNAPSHOT_FORMAT_VERSION, digest, entry_count))
                first = True
                for w, f, t in items:
                    if not first:
                        fp.write(',')
                    first = False
                    fp.write(json.dumps([w, f, t], ensure_ascii=False))
                fp.write(']}')
            os.rename(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    def load_snapshot(self, digest):
        path = self.snapshot_path(digest)
        if not os.path.isfile(path):
            raise DictVersionNotFoundError(
                '词典版本快照 %s 不存在于 %s。\n'
                '可行做法：1) 确认 store_dir / 环境变量 JIEBA_DICT_VERSION_DIR '
                '指向当时开启记录时使用的目录；2) 若保留有当时的词典文件，'
                '可用 jieba.dict_diff_files() 直接对比、或重新 load_userdict 加载；'
                '3) 在开启记录的环境重新运行一次以重建快照。'
                % (digest, path))
        with io.open(path, 'r', encoding='utf-8') as fp:
            payload = json.load(fp)
        return dict((w, (f, t)) for w, f, t in payload['entries'])

    def append_history(self, record):
        line = json.dumps(record, ensure_ascii=False)
        with self._hist_lock:
            with io.open(self.history_path, 'a', encoding='utf-8') as fp:
                fp.write(line + '\n')

    def read_history(self):
        if not os.path.isfile(self.history_path):
            return []
        records = []
        with io.open(self.history_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError:
                    # 多进程追加可能造成行交错，跳过坏行而不是整体失败
                    logger.warning('skip corrupted history line in %s',
                                   self.history_path)
        return records


class DictTracker(object):
    """
    挂在 Tokenizer 上的记录器。只负责记录与落盘，不改变分词行为。

    线程安全：记录变更由内部 RLock 保护；历史文件追加由 VersionStore 保护。
    多进程（enable_parallel）下，子进程为 fork 出的副本，其记录不会回传，
    请在主进程加载完词典后再开启并行。
    """

    def __init__(self, tokenizer, store_dir=None):
        self.tokenizer = tokenizer
        self.store = VersionStore(store_dir)
        self.records = []
        self._event_digests = []
        self._lock = threading.RLock()
        # 用词典路径标识分词器，回放/对比时按它过滤出同一条加载链
        dictionary = getattr(tokenizer, 'dictionary', None)
        self.tokenizer_id = repr(dictionary) if dictionary is not None else 'default'

    def state_version(self):
        with self._lock:
            if not self._event_digests:
                return None
            return combine_event_digests(self._event_digests)

    def _append_record(self, rec):
        rec['seq'] = len(self.records) + 1
        rec['time'] = time.time()
        rec['tokenizer'] = self.tokenizer_id
        rec['state_version'] = combine_event_digests(self._event_digests)
        self.records.append(rec)
        hist = dict(rec)
        conflicts = hist.get('conflicts')
        if conflicts and len(conflicts) > MAX_CONFLICTS_IN_HISTORY:
            hist['conflicts'] = conflicts[:MAX_CONFLICTS_IN_HISTORY]
            hist['conflicts_truncated'] = True
        try:
            self.store.append_history(hist)
        except Exception:
            # 记录落盘失败不应影响分词主流程
            logger.exception('failed to append dictionary history')

    def _record_entries_event(self, kind, source, entries, extra=None):
        """entries: word -> (freq, tag)。返回记录。"""
        digest = compute_entries_digest(entries)
        if digest is None:
            digest_status = 'unknown'
            logger.warning(
                'dictionary %s has more than %d entries; '
                'content digest is unavailable, version marked as unknown',
                source, DIGEST_MAX_ENTRIES)
        else:
            digest_status = 'ok'
            try:
                self.store.save_snapshot(digest, entries)
            except Exception:
                digest_status = 'snapshot_failed'
                logger.exception('failed to save snapshot for %s', source)
        with self._lock:
            self._event_digests.append(digest or 'unknown')
            rec = {
                'kind': kind,
                'source': source,
                'entry_count': len(entries),
                'digest': digest,
                'digest_status': digest_status,
            }
            if extra:
                rec.update(extra)
            self._append_record(rec)
            return rec

    def record_main_dict(self, source, from_cache=False):
        """记录主词典加载。词条取 FREQ 中词频大于 0 的项（前缀碎片词频为 0，自动排除）。

        为控制内存峰值，不复制条目表：只对词表排序（引用列表），
        摘要与快照都按排序后的键流式计算/写盘。
        """
        tok = self.tokenizer
        source = source or 'default:dict.txt'
        words = [w for w, f in iteritems(tok.FREQ) if f > 0]
        count = len(words)
        if count > DIGEST_MAX_ENTRIES:
            digest = None
            digest_status = 'unknown'
            logger.warning(
                'dictionary %s has more than %d entries; '
                'content digest is unavailable, version marked as unknown',
                source, DIGEST_MAX_ENTRIES)
        else:
            words.sort()
            h = hashlib.md5()
            for w in words:
                h.update(('%s\t%s\t\n' % (w, tok.FREQ[w])).encode('utf-8'))
            digest = h.hexdigest()
            digest_status = 'ok'
            try:
                self.store.save_snapshot_items(
                    digest, ((w, tok.FREQ[w], None) for w in words), count)
            except Exception:
                digest_status = 'snapshot_failed'
                logger.exception('failed to save snapshot for %s', source)
        with self._lock:
            self._event_digests.append(digest or 'unknown')
            self._append_record({
                'kind': 'main_dict',
                'source': source,
                'entry_count': count,
                'digest': digest,
                'digest_status': digest_status,
                'from_cache': bool(from_cache),
            })
            return self.records[-1]

    def record_userdict(self, source, entries, conflicts):
        """记录一次用户词典加载。conflicts 为冲突列表（含最终生效值）。"""
        rec = self._record_entries_event(
            'userdict', source, entries,
            {'conflicts': conflicts, 'conflict_count': len(conflicts)})
        if conflicts:
            logger.warning(
                'userdict %s: %d conflicting entr%s overwritten '
                '(later load overrides earlier), e.g. %s',
                source, len(conflicts), 'y was' if len(conflicts) == 1 else 'ies were',
                ', '.join('%s: %s -> %s' % (c['word'], c['previous_freq'],
                                            c['effective_freq'])
                          for c in conflicts[:5]))
        return rec

    def record_add_word(self, word, freq, tag):
        """记录单独的 add_word/del_word 调用（del_word 即 freq=0）。"""
        d = hashlib.md5(
            ('add\t%s\t%s\t%s' % (word, freq, tag or '')).encode('utf-8')
        ).hexdigest()
        with self._lock:
            self._event_digests.append(d)
            self._append_record({
                'kind': 'add_word',
                'source': None,
                'word': word,
                'freq': freq,
                'tag': tag,
                'entry_count': 1,
                'digest': d,
                'digest_status': 'ok',
            })


def _events_for_version(store, version):
    """从历史中找出达到指定状态版本的完整事件链（按分词器过滤）。"""
    history = store.read_history()
    last_idx = None
    for i, rec in enumerate(history):
        if rec.get('state_version') == version:
            last_idx = i
    if last_idx is None:
        raise DictVersionNotFoundError(
            '找不到状态版本 %s 的加载历史（存储目录：%s）。\n'
            '可行做法：1) 确认 store_dir / 环境变量 JIEBA_DICT_VERSION_DIR '
            '指向当时开启记录时使用的目录；2) 用 jieba.dict_history() 查看'
            '当前可见的版本列表；3) 若只有当时的词典文件，可重新加载后用 '
            'jieba.dict_diff_files() 对比。'
            % (version, store.dir))
    tok_id = history[last_idx].get('tokenizer')
    return [r for r in history[:last_idx + 1]
            if r.get('tokenizer') == tok_id]


def entries_for_version(store, version):
    """按事件链重建该版本的生效词条表：word -> (freq, tag)。后加载覆盖先加载。"""
    merged = {}
    for ev in _events_for_version(store, version):
        kind = ev.get('kind')
        if kind in ('main_dict', 'userdict'):
            digest = ev.get('digest')
            if not digest or ev.get('digest_status') == 'unknown':
                raise DictVersionNotFoundError(
                    '版本 %s 中的词典 %s 摘要为 unknown（条目数超过上限），'
                    '无法精确重建。请调大 dicttrack.DIGEST_MAX_ENTRIES 后重新记录，'
                    '或使用当时的词典文件。'
                    % (version, ev.get('source')))
            merged.update(store.load_snapshot(digest))
        elif kind == 'add_word':
            merged[ev['word']] = (ev.get('freq'), ev.get('tag'))
    return merged


def diff_versions(store, version_a, version_b):
    """对比两个状态版本的生效词条差异。"""
    old = entries_for_version(store, version_a)
    new = entries_for_version(store, version_b)
    result = diff_entries(old, new)
    result['version_a'] = version_a
    result['version_b'] = version_b
    return result


def diff_files(file_a, file_b):
    """直接对比两个用户词典文件（不需要开启记录）。"""
    old = parse_dict_file(file_a)
    new = parse_dict_file(file_b)
    result = diff_entries(old, new)
    result['file_a'] = file_a if isinstance(file_a, string_types) else resolve_filename(file_a)
    result['file_b'] = file_b if isinstance(file_b, string_types) else resolve_filename(file_b)
    return result
