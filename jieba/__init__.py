from __future__ import absolute_import, unicode_literals

__version__ = '0.42.1'
__license__ = 'MIT'

import marshal
import re
import tempfile
import threading
import time
from collections import OrderedDict
from hashlib import md5
from math import log

from . import finalseg
from ._compat import *

if os.name == 'nt':
    from shutil import move as _replace_file
else:
    _replace_file = os.rename

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

DEFAULT_DICT = None
DEFAULT_DICT_NAME = "dict.txt"

log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(log_console)

DICT_WRITING = {}

pool = None

re_userdict = re.compile('^(.+?)( [0-9]+)?( [a-z]+)?$', re.U)
re_userdict_tag = re.compile('^[a-z]+$', re.U)
re_userdict_freq = re.compile('^-?[0-9]+$', re.U)

re_eng = re.compile('[a-zA-Z0-9]', re.U)

# \u4E00-\u9FD5a-zA-Z0-9+#&\._ : All non-space characters. Will be handled with re_han
# \r\n|\s : whitespace characters. Will not be handled.
# re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)", re.U)
# Adding "-" symbol in re_han_default
re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)

re_skip_default = re.compile("(\r\n|\s)", re.U)

DEFAULT_AUTO_RELOAD_INTERVAL = 60.0
DEFAULT_CACHE_SIZE = 1024
DEFAULT_CACHEABLE_SENTENCE_LENGTH = 2048

_CACHE_MISS = object()


class DictionaryLoadError(ValueError):
    """Dictionary load failure with enough context for diagnostics."""

    def __init__(self, message, filename=None, line_number=None, entry=None):
        ValueError.__init__(self, message)
        self.filename = filename
        self.line_number = line_number
        self.entry = entry


class _DeltaFreq(object):
    """Read-only logical view of a base dictionary plus a small delta."""

    __slots__ = ('base', 'delta')

    def __init__(self, base, delta):
        self.base = base
        self.delta = delta

    def __contains__(self, key):
        return key in self.delta or key in self.base

    def __getitem__(self, key):
        if key in self.delta:
            return self.delta[key]
        return self.base[key]

    def get(self, key, default=None):
        if key in self.delta:
            return self.delta[key]
        return self.base.get(key, default)

    def __iter__(self):
        for key in self.base:
            if key not in self.delta:
                yield key
        for key in self.delta:
            yield key

    def __len__(self):
        return len(self.base) + sum(1 for key in self.delta if key not in self.base)

    def keys(self):
        return list(iter(self))

    def values(self):
        return [self[key] for key in self]

    def items(self):
        return [(key, self[key]) for key in self]

    def copy(self):
        result = dict(self.base)
        result.update(self.delta)
        return result

    def __eq__(self, other):
        try:
            return self.copy() == other
        except Exception:
            return False

    def __repr__(self):
        return '<_DeltaFreq base=%d delta=%d>' % (len(self.base), len(self.delta))


class _DictionaryState(object):
    """One complete, atomically published dictionary revision."""

    __slots__ = ('version', 'freq', 'total', 'entry_count', 'loaded_at',
                 'user_entries', 'user_tags', 'force_splits')

    def __init__(self, version, freq, total, entry_count, loaded_at,
                 user_entries, user_tags, force_splits=None):
        self.version = version
        self.freq = freq
        self.total = total
        self.entry_count = entry_count
        self.loaded_at = loaded_at
        self.user_entries = user_entries
        self.user_tags = user_tags
        self.force_splits = frozenset(force_splits or ())


def setLogLevel(log_level):
    default_logger.setLevel(log_level)


class Tokenizer(object):

    def __init__(self, dictionary=DEFAULT_DICT):
        self.lock = threading.RLock()
        self._reload_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._auto_lock = threading.RLock()
        if dictionary == DEFAULT_DICT:
            self.dictionary = dictionary
        else:
            self.dictionary = _get_abs_path(dictionary)
        self.FREQ = {}
        self.total = 0
        self.user_word_tag_tab = {}
        self.initialized = False
        self.tmp_dir = None
        self.cache_file = None
        self._base_freq = {}
        self._state = None
        self._previous_state = None
        self._max_version = 0
        self._result_cache = OrderedDict()
        self._cache_maxsize = DEFAULT_CACHE_SIZE
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_load_info = None
        self._last_error = None
        self._last_attempt_source = None
        self.userdict_path = None
        self.auto_reload_interval = DEFAULT_AUTO_RELOAD_INTERVAL
        self.max_reload_block_seconds = 0.2
        self._auto_reload_thread = None
        self._auto_reload_stop = None

    def __repr__(self):
        return '<Tokenizer dictionary=%r>' % self.dictionary

    def _snapshot(self):
        state = self._state
        if state is None or not self.initialized:
            self.check_initialized()
            state = self._state
        return state

    def _publish_state(self, state):
        self._state = state
        self.FREQ = state.freq
        self.total = state.total
        self.initialized = True
        if state.version > self._max_version:
            self._max_version = state.version

    def _cache_get(self, key):
        if self._cache_maxsize <= 0:
            return _CACHE_MISS
        with self._state_lock:
            try:
                result = self._result_cache.pop(key)
            except KeyError:
                self._cache_misses += 1
                return _CACHE_MISS
            self._result_cache[key] = result
            self._cache_hits += 1
            return result

    def _cache_put(self, key, state, result):
        if self._cache_maxsize <= 0:
            return
        with self._state_lock:
            # A reload may have happened while this request was computing.
            # Never publish a result from an older snapshot after the switch.
            if self._state is not state:
                return
            try:
                self._result_cache.pop(key)
            except KeyError:
                pass
            self._result_cache[key] = result
            while len(self._result_cache) > self._cache_maxsize:
                self._result_cache.popitem(last=False)

    def _match_cache_keys(self, keys, words):
        words = tuple(word for word in words if word)
        if not words or not keys:
            return set()
        if len(words) <= 32:
            return set(
                key for key in keys
                if any(word in key[0] for word in words))

        word_set = set(words)
        first_chars = set(word[0] for word in words)
        max_word_len = max(len(word) for word in words)
        affected = set()
        for key in keys:
            sentence = key[0]
            if not any(ch in first_chars for ch in sentence):
                continue
            length = len(sentence)
            if length * min(length, max_word_len) > 200000:
                if any(word in sentence for word in words):
                    affected.add(key)
                continue
            found = False
            for start in xrange(length):
                if sentence[start] not in first_chars:
                    continue
                limit = min(length, start + max_word_len)
                for end in xrange(start + 1, limit + 1):
                    if sentence[start:end] in word_set:
                        affected.add(key)
                        found = True
                        break
                if found:
                    break
        return affected

    def _plan_cache_invalidation(self, words):
        with self._state_lock:
            snapshot_keys = frozenset(self._result_cache.keys())
        return snapshot_keys, self._match_cache_keys(snapshot_keys, words)

    def _apply_cache_invalidation(self, snapshot_keys, affected):
        invalidated = 0
        for key in affected:
            if self._result_cache.pop(key, None) is not None:
                invalidated += 1
        # Results inserted by requests racing with the load were computed from
        # the old snapshot. Remove them conservatively rather than risk a
        # stale hit; normal requests keep using the old state without blocking.
        new_keys = [
            key for key in self._result_cache if key not in snapshot_keys]
        for key in new_keys:
            if self._result_cache.pop(key, None) is not None:
                invalidated += 1
        return invalidated

    def _apply_force_splits(self, add_words, remove_words):
        for word in remove_words:
            if word:
                finalseg.remove_force_split(word)
        for word in add_words:
            if word:
                finalseg.add_force_split(word)

    def _record_load_error(self, error, path, duration, old_version):
        entry = getattr(error, 'entry', None)
        line_number = getattr(error, 'line_number', None)
        filename = getattr(error, 'filename', None) or path
        error_info = {
            'message': str(error),
            'filename': filename,
            'file': filename,
            'line_number': line_number,
            'line': line_number,
            'entry': entry,
            'snippet': entry,
            'timestamp': time.time(),
        }
        self._last_error = error_info
        self._last_load_info = {
            'status': 'rejected',
            'old_version': old_version,
            'version': old_version,
            'duration_seconds': duration,
            'error': dict(error_info),
            'timestamp': error_info['timestamp'],
        }

    @staticmethod
    def _source_signature(path):
        stat = os.stat(path)
        mtime_ns = getattr(stat, 'st_mtime_ns', None)
        if mtime_ns is None:
            mtime_ns = int(stat.st_mtime * 1000000000)
        return (os.path.abspath(path), mtime_ns, stat.st_size)

    @staticmethod
    def _entry_snippet(value, limit=120):
        if isinstance(value, bytes):
            value = value.decode('utf-8', 'replace')
        value = value.replace('\r', '\\r').replace('\n', '\\n')
        if len(value) > limit:
            value = value[:limit] + '...'
        return value

    def _parse_userdict(self, f, f_name):
        entries = OrderedDict()
        duplicates = 0
        parsed_entries = 0
        signature = None

        if isinstance(f, string_types):
            abs_path = _get_abs_path(f)
            f_name = abs_path
            signature = self._source_signature(abs_path)
            with open(abs_path, 'rb') as source:
                data = source.read()
                after = self._source_signature(abs_path)
            if signature != after:
                raise DictionaryLoadError(
                    'dictionary file changed while being read: %s' % f_name,
                    filename=f_name)
            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError as exc:
                line_number = data[:exc.start].count(b'\n') + 1
                line_start = data.rfind(b'\n', 0, exc.start) + 1
                line_end = data.find(b'\n', exc.start)
                if line_end < 0:
                    line_end = len(data)
                snippet = self._entry_snippet(data[line_start:line_end])
                raise DictionaryLoadError(
                    'dictionary file %s must be utf-8 at Line %s: %s' %
                    (f_name, line_number, snippet),
                    filename=f_name, line_number=line_number, entry=snippet)
            lines = text.splitlines()
        else:
            lines = f

        for lineno, raw_line in enumerate(lines, 1):
            if not isinstance(raw_line, text_type):
                try:
                    line = raw_line.decode('utf-8')
                except UnicodeDecodeError:
                    snippet = self._entry_snippet(raw_line)
                    raise DictionaryLoadError(
                        'dictionary file %s must be utf-8 at Line %s: %s' %
                        (f_name, lineno, snippet),
                        filename=f_name, line_number=lineno, entry=snippet)
            else:
                line = raw_line
            if lineno == 1:
                line = line.lstrip('\ufeff')
            line = line.strip()
            if not line:
                continue
            parsed_entries += 1

            body = line
            tag = None
            last_space = body.rfind(' ')
            if last_space >= 0:
                possible_tag = body[last_space + 1:]
                if re_userdict_tag.match(possible_tag):
                    tag = possible_tag
                    body = body[:last_space]

            freq = None
            last_space = body.rfind(' ')
            if last_space >= 0:
                possible_freq = body[last_space + 1:]
                if re_userdict_freq.match(possible_freq):
                    try:
                        freq = int(possible_freq)
                    except ValueError:
                        snippet = self._entry_snippet(raw_line)
                        raise DictionaryLoadError(
                            'invalid word frequency in %s at Line %s: %s' %
                            (f_name, lineno, snippet),
                            filename=f_name, line_number=lineno, entry=snippet)
                    body = body[:last_space]
                elif possible_freq[:1].isdigit() or (
                        possible_freq[:1] == '-' and
                        possible_freq[1:2].isdigit()):
                    snippet = self._entry_snippet(line)
                    raise DictionaryLoadError(
                        'invalid word frequency in %s at Line %s: %s' %
                        (f_name, lineno, snippet),
                        filename=f_name, line_number=lineno, entry=snippet)

            word = body.strip()
            if not word or any(ord(ch) < 32 for ch in word):
                snippet = self._entry_snippet(line)
                raise DictionaryLoadError(
                    'invalid dictionary entry in %s at Line %s: %s' %
                    (f_name, lineno, snippet),
                    filename=f_name, line_number=lineno, entry=snippet)
            if freq is not None and freq < 0:
                snippet = self._entry_snippet(line)
                raise DictionaryLoadError(
                    'negative word frequency in %s at Line %s: %s' %
                    (f_name, lineno, snippet),
                    filename=f_name, line_number=lineno, entry=snippet)
            if word in entries:
                duplicates += 1
            entries[word] = (freq, tag)

        return entries, duplicates, parsed_entries, signature

    def _suggest_freq_state(self, word, state):
        ftotal = float(state.total)
        freq = 1
        for seg in self._cut_raw(word, False, False, False, state):
            freq *= state.freq.get(seg, 1) / ftotal
        return max(int(freq * state.total) + 1, state.freq.get(word, 1))

    def _build_candidate(self, entries, duplicates, path):
        state = self._state
        delta = dict(state.freq.delta)
        freq = _DeltaFreq(self._base_freq, delta)
        user_entries = dict(state.user_entries)
        user_tags = dict(state.user_tags)
        force_splits = set(state.force_splits)
        total = state.total
        entry_count = state.entry_count
        added = 0
        updated = 0
        unchanged = 0
        changed_words = []
        force_splits_changed = []

        for word, (raw_freq, raw_tag) in iteritems(entries):
            old_entry = user_entries.get(word)
            if raw_freq is None:
                if old_entry is not None:
                    word_freq = old_entry[0]
                else:
                    temp_state = _DictionaryState(
                        state.version, freq, total, entry_count,
                        state.loaded_at, user_entries, user_tags,
                        force_splits)
                    word_freq = self._suggest_freq_state(word, temp_state)
            else:
                word_freq = raw_freq

            if raw_tag is None and old_entry is not None:
                word_tag = old_entry[1]
            else:
                word_tag = raw_tag
            new_entry = (word_freq, word_tag)

            if old_entry == new_entry:
                unchanged += 1
                if raw_tag is not None:
                    user_tags[word] = raw_tag
                continue

            if old_entry is None:
                added += 1
            else:
                updated += 1

            old_freq = freq.get(word) or 0
            total += word_freq - old_freq

            if word not in freq:
                entry_count += 1
            delta[word] = word_freq
            for ch in xrange(len(word)):
                wfrag = word[:ch + 1]
                if wfrag not in freq:
                    freq.delta[wfrag] = 0
                    entry_count += 1

            user_entries[word] = new_entry
            if raw_tag is not None:
                user_tags[word] = raw_tag
            changed_words.append(word)
            if word_freq == 0:
                if word not in force_splits:
                    force_splits_changed.append(word)
                force_splits.add(word)
            elif word in force_splits:
                force_splits.remove(word)
                force_splits_changed.append(word)

        if not changed_words:
            return None, {
                'added': added,
                'updated': updated,
                'unchanged': unchanged,
                'duplicates': duplicates,
                'entry_count': state.entry_count,
                'entry_count_delta': 0,
                'changed_words': [],
                'force_splits': [],
            }

        candidate = _DictionaryState(
            version=self._max_version + 1,
            freq=freq,
            total=total,
            entry_count=entry_count,
            loaded_at=time.time(),
            user_entries=user_entries,
            user_tags=user_tags,
            force_splits=force_splits)
        return candidate, {
            'added': added,
            'updated': updated,
            'unchanged': unchanged,
            'duplicates': duplicates,
            'entry_count': entry_count,
            'entry_count_delta': entry_count - state.entry_count,
            'changed_words': changed_words,
            'force_splits': force_splits_changed,
        }

    def _commit_candidate(self, candidate, metrics):
        old_state = self._state
        snapshot_keys, affected = self._plan_cache_invalidation(
            metrics['changed_words'])
        switch_start = time.time()
        with self._state_lock:
            invalidated = self._apply_cache_invalidation(
                snapshot_keys, affected)
            self._publish_state(candidate)
            self._previous_state = old_state
            self.user_word_tag_tab.clear()
            self.user_word_tag_tab.update(candidate.user_tags)
        switch_seconds = time.time() - switch_start
        removed = old_state.force_splits.difference(candidate.force_splits)
        added = candidate.force_splits.difference(old_state.force_splits)
        self._apply_force_splits(added, removed)
        if switch_seconds > self.max_reload_block_seconds:
            default_logger.warning(
                'Dictionary switch blocked requests for %.6fs '
                '(configured maximum %.6fs).' %
                (switch_seconds, self.max_reload_block_seconds))
        return invalidated, switch_seconds

    @staticmethod
    def gen_pfdict(f):
        lfreq = {}
        ltotal = 0
        f_name = resolve_filename(f)
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip().decode('utf-8')
                word, freq = line.split(' ')[:2]
                freq = int(freq)
                lfreq[word] = freq
                ltotal += freq
                for ch in xrange(len(word)):
                    wfrag = word[:ch + 1]
                    if wfrag not in lfreq:
                        lfreq[wfrag] = 0
            except ValueError:
                raise ValueError(
                    'invalid dictionary entry in %s at Line %s: %s' % (f_name, lineno, line))
        f.close()
        return lfreq, ltotal

    def initialize(self, dictionary=None):
        with self._reload_lock:
            if dictionary:
                abs_path = _get_abs_path(dictionary)
                if self.dictionary == abs_path and self.initialized:
                    return
                else:
                    self.dictionary = abs_path
                    self.initialized = False
            else:
                abs_path = self.dictionary

            with self.lock:
                try:
                    with DICT_WRITING[abs_path]:
                        pass
                except KeyError:
                    pass
                if self.initialized:
                    return

                default_logger.debug("Building prefix dict from %s ..." % (abs_path or 'the default dictionary'))
                t1 = time.time()
                if self.cache_file:
                    cache_file = self.cache_file
                # default dictionary
                elif abs_path == DEFAULT_DICT:
                    cache_file = "jieba.cache"
                # custom dictionary
                else:
                    cache_file = "jieba.u%s.cache" % md5(
                        abs_path.encode('utf-8', 'replace')).hexdigest()
                cache_file = os.path.join(
                    self.tmp_dir or tempfile.gettempdir(), cache_file)
                # prevent absolute path in self.cache_file
                tmpdir = os.path.dirname(cache_file)

                load_from_cache_fail = True
                if os.path.isfile(cache_file) and (abs_path == DEFAULT_DICT or
                                                   os.path.getmtime(cache_file) > os.path.getmtime(abs_path)):
                    default_logger.debug(
                        "Loading model from cache %s" % cache_file)
                    try:
                        with open(cache_file, 'rb') as cf:
                            freq, total = marshal.load(cf)
                        load_from_cache_fail = False
                    except Exception:
                        load_from_cache_fail = True

                if load_from_cache_fail:
                    wlock = DICT_WRITING.get(abs_path, threading.RLock())
                    DICT_WRITING[abs_path] = wlock
                    with wlock:
                        freq, total = self.gen_pfdict(self.get_dict_file())
                        default_logger.debug(
                            "Dumping model to file cache %s" % cache_file)
                        try:
                            # prevent moving across different filesystems
                            fd, fpath = tempfile.mkstemp(dir=tmpdir)
                            with os.fdopen(fd, 'wb') as temp_cache_file:
                                marshal.dump(
                                    (freq, total), temp_cache_file)
                            _replace_file(fpath, cache_file)
                        except Exception:
                            default_logger.exception("Dump cache file failed.")

                    try:
                        del DICT_WRITING[abs_path]
                    except KeyError:
                        pass

                self._base_freq = freq
                state = _DictionaryState(
                    version=1,
                    freq=_DeltaFreq(freq, {}),
                    total=total,
                    entry_count=len(freq),
                    loaded_at=time.time(),
                    user_entries={},
                    user_tags={},
                    force_splits=())
                with self._state_lock:
                    self._publish_state(state)
                    self._previous_state = None
                    self._max_version = 1
                    self._result_cache.clear()
                    self.user_word_tag_tab.clear()
                self._last_attempt_source = None
                self._last_load_info = {
                    'status': 'initialized',
                    'old_version': None,
                    'version': 1,
                    'duration_seconds': time.time() - t1,
                    'entry_count': len(freq),
                    'entry_count_delta': len(freq),
                    'added': 0,
                    'updated': 0,
                    'duplicates': 0,
                    'invalidated_cache_entries': 0,
                    'switch_seconds': 0.0,
                    'timestamp': state.loaded_at,
                }
                default_logger.debug(
                    "Loading model cost %.3f seconds." % (time.time() - t1))
                default_logger.debug("Prefix dict has been built successfully.")

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def calc(self, sentence, DAG, route, state=None):
        if state is None:
            state = self._snapshot()
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = log(state.total)
        for idx in xrange(N - 1, -1, -1):
            route[idx] = max((log(state.freq.get(sentence[idx:x + 1]) or 1) -
                              logtotal + route[x + 1][0], x) for x in DAG[idx])

    def get_DAG(self, sentence, state=None):
        if state is None:
            state = self._snapshot()
        DAG = {}
        N = len(sentence)
        for k in xrange(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N and frag in state.freq:
                if state.freq[frag]:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

    def __cut_all(self, sentence, state):
        dag = self.get_DAG(sentence, state)
        old_j = -1
        eng_scan = 0
        eng_buf = u''
        for k, L in iteritems(dag):
            if eng_scan == 1 and not re_eng.match(sentence[k]):
                eng_scan = 0
                yield eng_buf
            if len(L) == 1 and k > old_j:
                word = sentence[k:L[0] + 1]
                if re_eng.match(word):
                    if eng_scan == 0:
                        eng_scan = 1
                        eng_buf = word
                    else:
                        eng_buf += word
                if eng_scan == 0:
                    yield word
                old_j = L[0]
            else:
                for j in L:
                    if j > k:
                        yield sentence[k:j + 1]
                        old_j = j
        if eng_scan == 1:
            yield eng_buf

    def __cut_DAG_NO_HMM(self, sentence, state):
        DAG = self.get_DAG(sentence, state)
        route = {}
        self.calc(sentence, DAG, route, state)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if re_eng.match(l_word) and len(l_word) == 1:
                buf += l_word
                x = y
            else:
                if buf:
                    yield buf
                    buf = ''
                yield l_word
                x = y
        if buf:
            yield buf
            buf = ''

    def __cut_DAG(self, sentence, state):
        DAG = self.get_DAG(sentence, state)
        route = {}
        self.calc(sentence, DAG, route, state)
        x = 0
        buf = ''
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield buf
                        buf = ''
                    else:
                        if not state.freq.get(buf):
                            recognized = finalseg.cut(buf)
                            for t in recognized:
                                yield t
                        else:
                            for elem in buf:
                                yield elem
                        buf = ''
                yield l_word
            x = y

        if buf:
            if len(buf) == 1:
                yield buf
            elif not state.freq.get(buf):
                recognized = finalseg.cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

    def _cut_raw(self, sentence, cut_all, HMM, use_paddle, state):
        is_paddle_installed = check_paddle_install['is_paddle_installed']
        if use_paddle and is_paddle_installed:
            # if sentence is null, it will raise core exception in paddle.
            if sentence is None or len(sentence) == 0:
                return
            import jieba.lac_small.predict as predict
            results = predict.get_sent(sentence)
            for sent in results:
                if sent is None:
                    continue
                yield sent
            return
        re_han = re_han_default
        re_skip = re_skip_default
        if cut_all:
            cut_block = self.__cut_all
        elif HMM:
            cut_block = self.__cut_DAG
        else:
            cut_block = self.__cut_DAG_NO_HMM
        blocks = re_han.split(sentence)
        for blk in blocks:
            if not blk:
                continue
            if re_han.match(blk):
                for word in cut_block(blk, state):
                    yield word
            else:
                tmp = re_skip.split(blk)
                for x in tmp:
                    if re_skip.match(x):
                        yield x
                    elif not cut_all:
                        for xx in x:
                            yield xx
                    else:
                        yield x

    def cut(self, sentence, cut_all=False, HMM=True, use_paddle=False):
        """
        The main function that segments an entire sentence that contains
        Chinese characters into separated words.

        Parameter:
            - sentence: The str(unicode) to be segmented.
            - cut_all: Model type. True for full pattern, False for accurate pattern.
            - HMM: Whether to use the Hidden Markov Model.
        """
        sentence = strdecode(sentence)
        state = self._snapshot()

        def iter_cut():
            cacheable = (
                not use_paddle and
                len(sentence) <= DEFAULT_CACHEABLE_SENTENCE_LENGTH and
                self._cache_maxsize > 0)
            key = (sentence, bool(cut_all), bool(HMM))
            if cacheable:
                cached = self._cache_get(key)
                if cached is not _CACHE_MISS:
                    for word in cached:
                        yield word
                    return
            result = tuple(self._cut_raw(
                sentence, cut_all, HMM, use_paddle, state))
            if cacheable:
                self._cache_put(key, state, result)
            for word in result:
                yield word

        return iter_cut()

    def cut_for_search(self, sentence, HMM=True):
        """
        Finer segmentation for search engines.
        """
        sentence = strdecode(sentence)
        state = self._snapshot()

        def iter_cut_for_search():
            words = self._cut_raw(sentence, False, HMM, False, state)
            for w in words:
                if len(w) > 2:
                    for i in xrange(len(w) - 1):
                        gram2 = w[i:i + 2]
                        if state.freq.get(gram2):
                            yield gram2
                if len(w) > 3:
                    for i in xrange(len(w) - 2):
                        gram3 = w[i:i + 3]
                        if state.freq.get(gram3):
                            yield gram3
                yield w

        return iter_cut_for_search()

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))

    def lcut_for_search(self, *args, **kwargs):
        return list(self.cut_for_search(*args, **kwargs))

    _lcut = lcut
    _lcut_for_search = lcut_for_search

    def _lcut_no_hmm(self, sentence):
        return self.lcut(sentence, False, False)

    def _lcut_all(self, sentence):
        return self.lcut(sentence, True)

    def _lcut_for_search_no_hmm(self, sentence):
        return self.lcut_for_search(sentence, False)

    def get_dict_file(self):
        if self.dictionary == DEFAULT_DICT:
            return get_module_res(DEFAULT_DICT_NAME)
        else:
            return open(self.dictionary, 'rb')

    def load_userdict(self, f):
        '''
        Load personalized dict to improve detect rate.

        Parameter:
            - f : A plain text file contains words and their ocurrences.
                  Can be a file-like object, or the path of the dictionary file,
                  whose encoding must be utf-8.

        Structure of dict file:
        word1 freq1 word_type1
        word2 freq2 word_type2
        ...
        Word type may be ignored
        '''
        return self.reload_userdict(f, force=True)

    def reload_userdict(self, f=None, force=False):
        """Incrementally load a user dictionary and atomically publish it.

        ``f`` may be a path or a file-like object. When omitted, the path used
        by the previous explicit/automatic reload is reused.
        """
        self.check_initialized()
        if f is None:
            f = self.userdict_path
            if f is None:
                raise ValueError('jieba: no user dictionary path configured')

        path = None
        signature = None
        if isinstance(f, string_types):
            path = _get_abs_path(f)
            try:
                signature = self._source_signature(path)
            except Exception as exc:
                with self._reload_lock:
                    duration = 0.0
                    self._record_load_error(
                        exc, path, duration, self._state.version)
                raise
        else:
            resolve_filename(f)

        with self._reload_lock:
            state = self._state
            if path is not None:
                self.userdict_path = path
                attempt_source = signature
                last_status = (self._last_load_info or {}).get('status')
                if not force and self._last_attempt_source == attempt_source \
                        and last_status != 'rejected':
                    info = {
                        'status': 'no_change',
                        'old_version': state.version,
                        'version': state.version,
                        'version_before': state.version,
                        'version_after': state.version,
                        'duration_seconds': 0.0,
                        'load_duration_seconds': 0.0,
                        'entry_count': state.entry_count,
                        'entry_count_before': state.entry_count,
                        'entry_count_after': state.entry_count,
                        'entry_count_delta': 0,
                        'added': 0,
                        'added_entries': 0,
                        'updated': 0,
                        'updated_entries': 0,
                        'duplicates': 0,
                        'duplicate_entries': 0,
                        'invalidated_cache_entries': 0,
                        'cache_invalidated': 0,
                        'switch_seconds': 0.0,
                        'timestamp': time.time(),
                    }
                    self._last_load_info = info
                    return dict(info)
                self._last_attempt_source = attempt_source

            start = time.time()
            old_version = state.version
            f_name = path if path is not None else resolve_filename(f)
            try:
                entries, duplicates, parsed_entries, parsed_signature = \
                    self._parse_userdict(f, f_name)
                if parsed_signature is not None:
                    signature = parsed_signature
                    self._last_attempt_source = signature
                candidate, metrics = self._build_candidate(
                    entries, duplicates, path)
            except Exception as exc:
                duration = time.time() - start
                self._record_load_error(exc, path, duration, old_version)
                raise

            if candidate is None:
                info = {
                    'status': 'no_change',
                    'old_version': old_version,
                    'version': old_version,
                    'version_before': old_version,
                    'version_after': old_version,
                    'duration_seconds': time.time() - start,
                    'load_duration_seconds': time.time() - start,
                    'entry_count': state.entry_count,
                    'entry_count_before': state.entry_count,
                    'entry_count_after': state.entry_count,
                    'entry_count_delta': 0,
                    'parsed_entries': parsed_entries,
                    'added': metrics['added'],
                    'added_entries': metrics['added'],
                    'updated': metrics['updated'],
                    'updated_entries': metrics['updated'],
                    'unchanged': metrics['unchanged'],
                    'duplicates': metrics['duplicates'],
                    'duplicate_entries': metrics['duplicates'],
                    'invalidated_cache_entries': 0,
                    'cache_invalidated': 0,
                    'switch_seconds': 0.0,
                    'timestamp': time.time(),
                }
                self._last_load_info = info
                self._last_error = None
                return dict(info)

            invalidated, switch_seconds = self._commit_candidate(
                candidate, metrics)
            duration = time.time() - start
            info = {
                'status': 'loaded',
                'old_version': old_version,
                'version': candidate.version,
                'version_before': old_version,
                'version_after': candidate.version,
                'duration_seconds': duration,
                'load_duration_seconds': duration,
                'parse_and_build_seconds': duration - switch_seconds,
                'switch_seconds': switch_seconds,
                'entry_count_before': state.entry_count,
                'entry_count': candidate.entry_count,
                'entry_count_after': candidate.entry_count,
                'entry_count_delta':
                    candidate.entry_count - state.entry_count,
                'parsed_entries': parsed_entries,
                'added': metrics['added'],
                'added_entries': metrics['added'],
                'updated': metrics['updated'],
                'updated_entries': metrics['updated'],
                'unchanged': metrics['unchanged'],
                'duplicates': metrics['duplicates'],
                'duplicate_entries': metrics['duplicates'],
                'changed_words': len(metrics['changed_words']),
                'invalidated_cache_entries': invalidated,
                'cache_invalidated': invalidated,
                'cache_entries': len(self._result_cache),
                'timestamp': candidate.loaded_at,
            }
            self._last_load_info = info
            self._last_error = None
            default_logger.debug(
                'Hot reload published version %s: +%s entries, '
                '%s cache entries invalidated in %.6fs.' %
                (candidate.version, info['entry_count_delta'],
                 invalidated, duration))
            return dict(info)

    def add_word(self, word, freq=None, tag=None):
        """
        Add a word to dictionary.

        freq and tag can be omitted, freq defaults to be a calculated value
        that ensures the word can be cut out.
        """
        self.check_initialized()
        word = strdecode(word)
        freq = int(freq) if freq is not None else None
        with self._reload_lock:
            old_version = self._state.version
            candidate, metrics = self._build_candidate(
                OrderedDict([(word, (freq, tag))]), 0, None)
            if candidate is None:
                return
            invalidated, switch_seconds = self._commit_candidate(
                candidate, metrics)
            self._last_load_info = {
                'status': 'loaded',
                'old_version': old_version,
                'version': candidate.version,
                'version_before': old_version,
                'version_after': candidate.version,
                'duration_seconds': switch_seconds,
                'load_duration_seconds': switch_seconds,
                'switch_seconds': switch_seconds,
                'entry_count_before': self._previous_state.entry_count,
                'entry_count': candidate.entry_count,
                'entry_count_after': candidate.entry_count,
                'entry_count_delta':
                    candidate.entry_count - self._previous_state.entry_count,
                'added': metrics['added'],
                'added_entries': metrics['added'],
                'updated': metrics['updated'],
                'updated_entries': metrics['updated'],
                'duplicates': metrics['duplicates'],
                'duplicate_entries': metrics['duplicates'],
                'invalidated_cache_entries': invalidated,
                'cache_invalidated': invalidated,
                'cache_entries': len(self._result_cache),
                'timestamp': candidate.loaded_at,
            }
            self._last_error = None

    def del_word(self, word):
        """
        Convenient function for deleting a word.
        """
        self.add_word(word, 0)

    def suggest_freq(self, segment, tune=False):
        """
        Suggest word frequency to force the characters in a word to be
        joined or splitted.

        Parameter:
            - segment : The segments that the word is expected to be cut into,
                        If the word should be treated as a whole, use a str.
            - tune : If True, tune the word frequency.

        Note that HMM may affect the final result. If the result doesn't change,
        set HMM=False.
        """
        self.check_initialized()
        state = self._snapshot()
        ftotal = float(state.total)
        freq = 1
        if isinstance(segment, string_types):
            word = segment
            for seg in self._cut_raw(word, False, False, False, state):
                freq *= state.freq.get(seg, 1) / ftotal
            freq = max(int(freq * state.total) + 1, state.freq.get(word, 1))
        else:
            segment = tuple(map(strdecode, segment))
            word = ''.join(segment)
            for seg in segment:
                freq *= state.freq.get(seg, 1) / ftotal
            freq = min(int(freq * state.total), state.freq.get(word, 0))
        if tune:
            self.add_word(word, freq)
        return freq

    def tokenize(self, unicode_sentence, mode="default", HMM=True):
        """
        Tokenize a sentence and yields tuples of (word, start, end)

        Parameter:
            - sentence: the str(unicode) to be segmented.
            - mode: "default" or "search", "search" is for finer segmentation.
            - HMM: whether to use the Hidden Markov Model.
        """
        if not isinstance(unicode_sentence, text_type):
            raise ValueError("jieba: the input parameter should be unicode.")
        state = self._snapshot()

        def iter_tokenize():
            start = 0
            if mode == 'default':
                for w in self._cut_raw(
                        unicode_sentence, False, HMM, False, state):
                    width = len(w)
                    yield (w, start, start + width)
                    start += width
            else:
                for w in self._cut_raw(
                        unicode_sentence, False, HMM, False, state):
                    width = len(w)
                    if len(w) > 2:
                        for i in xrange(len(w) - 1):
                            gram2 = w[i:i + 2]
                            if state.freq.get(gram2):
                                yield (gram2, start + i, start + i + 2)
                    if len(w) > 3:
                        for i in xrange(len(w) - 2):
                            gram3 = w[i:i + 3]
                            if state.freq.get(gram3):
                                yield (gram3, start + i, start + i + 3)
                    yield (w, start, start + width)
                    start += width

        return iter_tokenize()

    def set_dictionary(self, dictionary_path):
        with self._reload_lock:
            abs_path = _get_abs_path(dictionary_path)
            if not os.path.isfile(abs_path):
                raise Exception("jieba: file does not exist: " + abs_path)
            self.dictionary = abs_path
            self.initialized = False
            with self._state_lock:
                self._state = None

    def _changed_user_words(self, old_state, new_state):
        words = set(old_state.user_entries)
        words.update(new_state.user_entries)
        return [word for word in words
                if old_state.user_entries.get(word) !=
                new_state.user_entries.get(word)]

    def rollback(self):
        """Atomically restore the dictionary revision before the last load."""
        with self._reload_lock:
            current = self._state
            previous = self._previous_state
            if previous is None:
                raise ValueError('jieba: no previous dictionary version')
            changed_words = self._changed_user_words(current, previous)
            start = time.time()
            snapshot_keys, affected = self._plan_cache_invalidation(
                changed_words)
            with self._state_lock:
                invalidated = self._apply_cache_invalidation(
                    snapshot_keys, affected)
                self._publish_state(previous)
                self._previous_state = None
                self.user_word_tag_tab.clear()
                self.user_word_tag_tab.update(previous.user_tags)
            switch_seconds = time.time() - start
            removed = current.force_splits.difference(previous.force_splits)
            added = previous.force_splits.difference(current.force_splits)
            self._apply_force_splits(added, removed)
            info = {
                'status': 'rolled_back',
                'old_version': current.version,
                'version': previous.version,
                'version_before': current.version,
                'version_after': previous.version,
                'duration_seconds': switch_seconds,
                'load_duration_seconds': switch_seconds,
                'switch_seconds': switch_seconds,
                'entry_count': previous.entry_count,
                'entry_count_before': current.entry_count,
                'entry_count_after': previous.entry_count,
                'invalidated_cache_entries': invalidated,
                'cache_invalidated': invalidated,
                'cache_entries': len(self._result_cache),
                'timestamp': time.time(),
            }
            self._last_load_info = info
            default_logger.debug(
                'Rolled dictionary back from version %s to %s.' %
                (current.version, previous.version))
            return dict(info)

    def get_dict_info(self):
        """Return current dictionary and hot-reload observability data."""
        state = self._snapshot()
        with self._state_lock:
            cache_entries = len(self._result_cache)
            cache_hits = self._cache_hits
            cache_misses = self._cache_misses
        last_load = dict(self._last_load_info) if self._last_load_info else None
        last_error = dict(self._last_error) if self._last_error else None
        last_action_time = (
            (last_load or {}).get('timestamp') or state.loaded_at)
        return {
            'version': state.version,
            'dict_version': state.version,
            'current_version': state.version,
            'entry_count': state.entry_count,
            'dict_entry_count': state.entry_count,
            'total': state.total,
            'last_load_time': last_action_time,
            'last_reload_time': last_action_time,
            'last_load_time_iso': time.strftime(
                '%Y-%m-%d %H:%M:%S', time.localtime(last_action_time)),
            'dictionary_loaded_at': state.loaded_at,
            'last_load_duration': (
                (last_load or {}).get('duration_seconds')),
            'last_invalidated_cache_entries': (
                (last_load or {}).get('invalidated_cache_entries')),
            'last_duplicate_entries': (
                (last_load or {}).get('duplicates')),
            'dictionary': self.dictionary,
            'userdict_path': self.userdict_path,
            'can_rollback': self._previous_state is not None,
            'last_load': last_load,
            'last_error': last_error,
            'last_load_error': last_error,
            'cache_entries': cache_entries,
            'cache_maxsize': self._cache_maxsize,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'auto_reload_enabled': bool(
                self._auto_reload_thread is not None and
                self._auto_reload_thread.is_alive()),
            'auto_reload_interval': self.auto_reload_interval,
            'max_reload_block_seconds': self.max_reload_block_seconds,
        }

    get_dict_status = get_dict_info
    get_hot_reload_status = get_dict_info

    def set_cache_size(self, maxsize):
        """Set the bounded segmentation cache size; zero disables it."""
        maxsize = int(maxsize)
        if maxsize < 0:
            raise ValueError('jieba: cache size must be non-negative')
        with self._state_lock:
            self._cache_maxsize = maxsize
            while len(self._result_cache) > maxsize:
                self._result_cache.popitem(last=False)

    def _auto_reload_loop(self, stop_event):
        while not stop_event.is_set():
            interval = max(0.05, float(self.auto_reload_interval))
            if stop_event.wait(interval):
                break
            path = self.userdict_path
            if not path:
                continue
            try:
                signature = self._source_signature(path)
                attempt = signature
                if self._last_attempt_source == attempt:
                    continue
                self.reload_userdict(path)
            except Exception as exc:
                default_logger.warning(
                    'Automatic dictionary reload failed: %s', exc)

    def enable_auto_reload(self, path=None, interval=None):
        """Watch a user dictionary and load changes on a fixed interval."""
        with self._auto_lock:
            if path is not None:
                path = _get_abs_path(path)
                if not os.path.isfile(path):
                    raise ValueError(
                        'jieba: file does not exist: %s' % path)
                self.userdict_path = path
            elif self.userdict_path is None:
                raise ValueError('jieba: no user dictionary path configured')
            if interval is not None:
                interval = float(interval)
                if interval <= 0:
                    raise ValueError(
                        'jieba: auto reload interval must be positive')
                self.auto_reload_interval = interval
            if self._auto_reload_thread is not None and \
                    self._auto_reload_thread.is_alive():
                return
            self._auto_reload_stop = threading.Event()
            thread = threading.Thread(
                target=self._auto_reload_loop,
                args=(self._auto_reload_stop,))
            thread.daemon = True
            self._auto_reload_thread = thread
            thread.start()

    def disable_auto_reload(self):
        """Stop the background dictionary file watcher."""
        with self._auto_lock:
            thread = self._auto_reload_thread
            stop_event = self._auto_reload_stop
            self._auto_reload_thread = None
            self._auto_reload_stop = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive() and \
                thread is not threading.current_thread():
            thread.join(1.0)

    hot_reload = reload_userdict
    reload_dictionary = reload_userdict
    reload = reload_userdict
    start_auto_reload = enable_auto_reload
    stop_auto_reload = disable_auto_reload


# default Tokenizer instance

dt = Tokenizer()

# global functions

get_FREQ = lambda k, d=None: dt.FREQ.get(k, d)
add_word = dt.add_word
calc = dt.calc
cut = dt.cut
lcut = dt.lcut
cut_for_search = dt.cut_for_search
lcut_for_search = dt.lcut_for_search
del_word = dt.del_word
get_DAG = dt.get_DAG
get_dict_file = dt.get_dict_file
initialize = dt.initialize
load_userdict = dt.load_userdict
reload_userdict = dt.reload_userdict
reload_dictionary = dt.reload_userdict
hot_reload = dt.reload_userdict
reload = dt.reload_userdict
rollback = dt.rollback
get_dict_info = dt.get_dict_info
get_dict_status = dt.get_dict_status
get_hot_reload_status = dt.get_hot_reload_status
set_cache_size = dt.set_cache_size
enable_auto_reload = dt.enable_auto_reload
disable_auto_reload = dt.disable_auto_reload
start_auto_reload = dt.enable_auto_reload
stop_auto_reload = dt.disable_auto_reload
set_dictionary = dt.set_dictionary
suggest_freq = dt.suggest_freq
tokenize = dt.tokenize
user_word_tag_tab = dt.user_word_tag_tab


def get_dict_version():
    return dt.get_dict_info()['version']


def get_dict_entry_count():
    return dt.get_dict_info()['entry_count']


def _lcut_all(s):
    return dt._lcut_all(s)


def _lcut(s):
    return dt._lcut(s)


def _lcut_no_hmm(s):
    return dt._lcut_no_hmm(s)


def _lcut_all(s):
    return dt._lcut_all(s)


def _lcut_for_search(s):
    return dt._lcut_for_search(s)


def _lcut_for_search_no_hmm(s):
    return dt._lcut_for_search_no_hmm(s)


def _pcut(sentence, cut_all=False, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if cut_all:
        result = pool.map(_lcut_all, parts)
    elif HMM:
        result = pool.map(_lcut, parts)
    else:
        result = pool.map(_lcut_no_hmm, parts)
    for r in result:
        for w in r:
            yield w


def _pcut_for_search(sentence, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if HMM:
        result = pool.map(_lcut_for_search, parts)
    else:
        result = pool.map(_lcut_for_search_no_hmm, parts)
    for r in result:
        for w in r:
            yield w


def enable_parallel(processnum=None):
    """
    Change the module's `cut` and `cut_for_search` functions to the
    parallel version.

    Note that this only works using dt, custom Tokenizer
    instances are not supported.
    """
    global pool, dt, cut, cut_for_search
    from multiprocessing import cpu_count
    if os.name == 'nt':
        raise NotImplementedError(
            "jieba: parallel mode only supports posix system")
    else:
        from multiprocessing import Pool
    dt.check_initialized()
    if processnum is None:
        processnum = cpu_count()
    pool = Pool(processnum)
    cut = _pcut
    cut_for_search = _pcut_for_search


def disable_parallel():
    global pool, dt, cut, cut_for_search
    if pool:
        pool.close()
        pool = None
    cut = dt.cut
    cut_for_search = dt.cut_for_search
