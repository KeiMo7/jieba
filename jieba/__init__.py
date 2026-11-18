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
_DICT_WRITING_GUARD = threading.Lock()

pool = None

re_userdict = re.compile('^(.+?)( [0-9]+)?( [a-z]+)?$', re.U)

re_eng = re.compile('[a-zA-Z0-9]', re.U)

# \u4E00-\u9FD5a-zA-Z0-9+#&\._ : All non-space characters. Will be handled with re_han
# \r\n|\s : whitespace characters. Will not be handled.
# re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)", re.U)
# Adding "-" symbol in re_han_default
re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)

re_skip_default = re.compile("(\r\n|\s)", re.U)


def setLogLevel(log_level):
    default_logger.setLevel(log_level)


class _DictState(object):
    """
    Immutable snapshot of the mutable dictionary state of a Tokenizer.

    A cut operation captures one _DictState at its start and only reads
    from it, so concurrent dictionary updates never affect in-flight cuts.
    Writers build a new state and swap it in atomically (copy-on-write).
    """
    __slots__ = ('freq', 'total', 'force_split_words', 'version')

    def __init__(self, freq, total, force_split_words, version):
        self.freq = freq
        self.total = total
        self.force_split_words = force_split_words
        self.version = version


class _CutResultCache(object):
    """
    Bounded thread-safe LRU cache for cut results.

    Entries are tagged with the dictionary-state version they were computed
    from; entries from a stale version never hit and are evicted lazily.
    Eviction only drops references, so cuts already in flight are unaffected.
    """

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._data = OrderedDict()  # key -> (version, tuple_of_words)

    def get(self, key, version):
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry[0] != version:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return entry[1]

    def put(self, key, version, value):
        with self._lock:
            self._data[key] = (version, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()


class Tokenizer(object):

    def __init__(self, dictionary=DEFAULT_DICT):
        self.lock = threading.RLock()
        # serializes writers (add_word / load_userdict / initialize swap);
        # the read (cut) path never takes this lock.
        self._write_lock = threading.RLock()
        if dictionary == DEFAULT_DICT:
            self.dictionary = dictionary
        else:
            self.dictionary = _get_abs_path(dictionary)
        self._state_version = 0
        self._dict_state = _DictState({}, 0, frozenset(), 0)
        self.user_word_tag_tab = {}
        self.initialized = False
        self.tmp_dir = None
        self.cache_file = None
        self._cut_cache = None
        self._active_cuts = 0
        self._active_cuts_lock = threading.Lock()

    def __repr__(self):
        return '<Tokenizer dictionary=%r>' % self.dictionary

    # Backward-compatible views of the current dictionary snapshot.
    # Reads return the dict of the current snapshot; assignments replace
    # the whole snapshot (kept for API compatibility, not used internally).

    def _get_FREQ(self):
        return self._dict_state.freq

    def _set_FREQ(self, value):
        state = self._dict_state
        self._dict_state = _DictState(value, state.total,
                                      state.force_split_words, state.version)

    FREQ = property(_get_FREQ, _set_FREQ)

    def _get_total(self):
        return self._dict_state.total

    def _set_total(self, value):
        state = self._dict_state
        self._dict_state = _DictState(state.freq, value,
                                      state.force_split_words, state.version)

    total = property(_get_total, _set_total)

    def _enter_cut(self):
        with self._active_cuts_lock:
            self._active_cuts += 1

    def _exit_cut(self):
        with self._active_cuts_lock:
            self._active_cuts -= 1

    def _on_dict_changed(self, log_conflict=True):
        """Must be called under self._write_lock after the state was swapped."""
        cache = self._cut_cache
        if cache is not None:
            cache.clear()
        if log_conflict:
            with self._active_cuts_lock:
                active = self._active_cuts
            if active:
                default_logger.warning(
                    "jieba: dictionary of %r changed while %d cut(s) are in "
                    "flight; in-flight cuts finish with the previous "
                    "dictionary snapshot.", self, active)

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
            with _DICT_WRITING_GUARD:
                wlock = DICT_WRITING.get(abs_path)
            if wlock is not None:
                # another thread is loading the same dictionary; wait for it
                with wlock:
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
                with _DICT_WRITING_GUARD:
                    wlock = DICT_WRITING.get(abs_path)
                    if wlock is None:
                        wlock = DICT_WRITING[abs_path] = threading.RLock()
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

                with _DICT_WRITING_GUARD:
                    DICT_WRITING.pop(abs_path, None)

            # publish the new dictionary atomically; in-flight cuts keep
            # using the snapshot they captured when they started.
            with self._write_lock:
                was_initialized = bool(self._dict_state.freq)
                self._state_version += 1
                self._dict_state = _DictState(
                    freq, total, self._dict_state.force_split_words,
                    self._state_version)
                self._on_dict_changed(log_conflict=was_initialized)
            self.initialized = True
            default_logger.debug(
                "Loading model cost %.3f seconds." % (time.time() - t1))
            default_logger.debug("Prefix dict has been built successfully.")

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def calc(self, sentence, DAG, route, _state=None):
        state = _state if _state is not None else self._dict_state
        freq = state.freq
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = log(state.total)
        for idx in xrange(N - 1, -1, -1):
            route[idx] = max((log(freq.get(sentence[idx:x + 1]) or 1) -
                              logtotal + route[x + 1][0], x) for x in DAG[idx])

    def get_DAG(self, sentence, _state=None):
        self.check_initialized()
        state = _state if _state is not None else self._dict_state
        FREQ = state.freq
        DAG = {}
        N = len(sentence)
        for k in xrange(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N and frag in FREQ:
                if FREQ[frag]:
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
        freq = state.freq
        force_split_words = state.force_split_words
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
                        if not freq.get(buf):
                            recognized = finalseg.cut(buf, force_split_words)
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
            elif not freq.get(buf):
                recognized = finalseg.cut(buf, force_split_words)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

    def cut(self, sentence, cut_all=False, HMM=True, use_paddle=False, _state=None):
        """
        The main function that segments an entire sentence that contains
        Chinese characters into separated words.

        Parameter:
            - sentence: The str(unicode) to be segmented.
            - cut_all: Model type. True for full pattern, False for accurate pattern.
            - HMM: Whether to use the Hidden Markov Model.
        """
        self._enter_cut()
        try:
            is_paddle_installed = check_paddle_install['is_paddle_installed']
            sentence = strdecode(sentence)
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
            # Capture one immutable dictionary snapshot for the whole cut so
            # that concurrent dictionary updates cannot change the result
            # half-way through the sentence.
            if _state is None:
                self.check_initialized()
                state = self._dict_state
            else:
                state = _state
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
        finally:
            self._exit_cut()

    def cut_for_search(self, sentence, HMM=True, _state=None):
        """
        Finer segmentation for search engines.
        """
        if _state is None:
            self.check_initialized()
            state = self._dict_state
        else:
            state = _state
        freq = state.freq
        words = self.cut(sentence, HMM=HMM, _state=state)
        for w in words:
            if len(w) > 2:
                for i in xrange(len(w) - 1):
                    gram2 = w[i:i + 2]
                    if freq.get(gram2):
                        yield gram2
            if len(w) > 3:
                for i in xrange(len(w) - 2):
                    gram3 = w[i:i + 3]
                    if freq.get(gram3):
                        yield gram3
            yield w

    def _cut_cache_key(self, for_search, args, kwargs):
        if for_search:
            names = ('sentence', 'HMM')
        else:
            names = ('sentence', 'cut_all', 'HMM', 'use_paddle')
        values = []
        for idx, name in enumerate(names):
            if idx < len(args):
                values.append(args[idx])
            else:
                values.append(kwargs.get(name, None))
        sentence = values[0]
        if not isinstance(sentence, (text_type, bytes)):
            return None
        if not for_search and values[3]:
            # paddle results are not cached
            return None
        return (for_search,) + tuple(values)

    def enable_cut_cache(self, maxsize=1024):
        """
        Enable a bounded thread-safe LRU cache for lcut/lcut_for_search
        results. Dictionary updates invalidate cached entries automatically.
        """
        if maxsize <= 0:
            raise ValueError("jieba: cut cache maxsize must be positive")
        with self._write_lock:
            self._cut_cache = _CutResultCache(maxsize)

    def disable_cut_cache(self):
        with self._write_lock:
            self._cut_cache = None

    def lcut(self, *args, **kwargs):
        cache = self._cut_cache
        if cache is not None:
            key = self._cut_cache_key(False, args, kwargs)
            if key is not None:
                self.check_initialized()
                state = self._dict_state
                hit = cache.get(key, state.version)
                if hit is not None:
                    return list(hit)
                kwargs = dict(kwargs)
                kwargs['_state'] = state
                result = tuple(self.cut(*args, **kwargs))
                cache.put(key, state.version, result)
                return list(result)
        return list(self.cut(*args, **kwargs))

    def lcut_for_search(self, *args, **kwargs):
        cache = self._cut_cache
        if cache is not None:
            key = self._cut_cache_key(True, args, kwargs)
            if key is not None:
                self.check_initialized()
                state = self._dict_state
                hit = cache.get(key, state.version)
                if hit is not None:
                    return list(hit)
                kwargs = dict(kwargs)
                kwargs['_state'] = state
                result = tuple(self.cut_for_search(*args, **kwargs))
                cache.put(key, state.version, result)
                return list(result)
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

        The whole file is applied as one atomic update: cuts that are in
        flight while this method runs keep using the previous dictionary
        snapshot, cuts that start afterwards see all of the new words.
        A half-loaded dictionary is never visible.
        '''
        self.check_initialized()
        if isinstance(f, string_types):
            f_name = f
            f = open(f, 'rb')
        else:
            f_name = resolve_filename(f)
        entries = []
        for lineno, ln in enumerate(f, 1):
            line = ln.strip()
            if not isinstance(line, text_type):
                try:
                    line = line.decode('utf-8').lstrip('\ufeff')
                except UnicodeDecodeError:
                    raise ValueError('dictionary file %s must be utf-8' % f_name)
            if not line:
                continue
            # match won't be None because there's at least one character
            word, freq, tag = re_userdict.match(line).groups()
            if freq is not None:
                freq = freq.strip()
            if tag is not None:
                tag = tag.strip()
            entries.append((word, freq, tag))
        # Resolve default freqs against the current snapshot, then apply
        # the whole batch as a single copy-on-write swap.
        resolved = []
        for word, freq, tag in entries:
            if freq is None:
                freq = self.suggest_freq(word, False)
            else:
                freq = int(freq)
            resolved.append((word, freq, tag))
        with self._write_lock:
            state = self._dict_state
            new_freq = state.freq.copy()
            total = state.total
            force_split_words = state.force_split_words
            for word, freq, tag in resolved:
                new_freq[word] = freq
                total += freq
                if tag:
                    self.user_word_tag_tab[word] = tag
                for ch in xrange(len(word)):
                    wfrag = word[:ch + 1]
                    if wfrag not in new_freq:
                        new_freq[wfrag] = 0
                if freq == 0:
                    force_split_words = force_split_words | frozenset((word,))
            self._state_version += 1
            self._dict_state = _DictState(
                new_freq, total, force_split_words, self._state_version)
            self._on_dict_changed()

    def add_word(self, word, freq=None, tag=None):
        """
        Add a word to dictionary.

        freq and tag can be omitted, freq defaults to be a calculated value
        that ensures the word can be cut out.

        The update is applied atomically (copy-on-write); cuts already in
        flight keep using the previous dictionary snapshot.
        """
        self.check_initialized()
        word = strdecode(word)
        freq = int(freq) if freq is not None else self.suggest_freq(word, False)
        with self._write_lock:
            state = self._dict_state
            new_freq = state.freq.copy()
            new_freq[word] = freq
            for ch in xrange(len(word)):
                wfrag = word[:ch + 1]
                if wfrag not in new_freq:
                    new_freq[wfrag] = 0
            force_split_words = state.force_split_words
            if freq == 0:
                force_split_words = force_split_words | frozenset((word,))
            if tag:
                self.user_word_tag_tab[word] = tag
            self._state_version += 1
            self._dict_state = _DictState(
                new_freq, state.total + freq, force_split_words,
                self._state_version)
            self._on_dict_changed()

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
        ftotal = float(self.total)
        freq = 1
        if isinstance(segment, string_types):
            word = segment
            for seg in self.cut(word, HMM=False):
                freq *= self.FREQ.get(seg, 1) / ftotal
            freq = max(int(freq * self.total) + 1, self.FREQ.get(word, 1))
        else:
            segment = tuple(map(strdecode, segment))
            word = ''.join(segment)
            for seg in segment:
                freq *= self.FREQ.get(seg, 1) / ftotal
            freq = min(int(freq * self.total), self.FREQ.get(word, 0))
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
        start = 0
        self.check_initialized()
        state = self._dict_state
        freq = state.freq
        if mode == 'default':
            for w in self.cut(unicode_sentence, HMM=HMM, _state=state):
                width = len(w)
                yield (w, start, start + width)
                start += width
        else:
            for w in self.cut(unicode_sentence, HMM=HMM, _state=state):
                width = len(w)
                if len(w) > 2:
                    for i in xrange(len(w) - 1):
                        gram2 = w[i:i + 2]
                        if freq.get(gram2):
                            yield (gram2, start + i, start + i + 2)
                if len(w) > 3:
                    for i in xrange(len(w) - 2):
                        gram3 = w[i:i + 3]
                        if freq.get(gram3):
                            yield (gram3, start + i, start + i + 3)
                yield (w, start, start + width)
                start += width

    def set_dictionary(self, dictionary_path):
        with self.lock:
            abs_path = _get_abs_path(dictionary_path)
            if not os.path.isfile(abs_path):
                raise Exception("jieba: file does not exist: " + abs_path)
            self.dictionary = abs_path
            self.initialized = False
            with self._active_cuts_lock:
                active = self._active_cuts
            if active:
                default_logger.warning(
                    "jieba: dictionary of %r switched while %d cut(s) are in "
                    "flight; in-flight cuts finish with the previous "
                    "dictionary snapshot.", self, active)


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
set_dictionary = dt.set_dictionary
suggest_freq = dt.suggest_freq
tokenize = dt.tokenize
user_word_tag_tab = dt.user_word_tag_tab


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
