# jieba 多线程并发安全说明

本文档说明分词在多线程下的并发安全改造：改造范围（哪些状态只读、哪些按线程隔离、哪些加锁）、
并发词典加载的语义、缓存行为、冲突日志，以及如何复现验证结果。

## 结论（TL;DR）

- 同一个 `Tokenizer` 实例（含模块级默认实例 `jieba.dt`）可以被多个线程安全地并发分词，
  结果与单线程逐条处理完全一致，**不需要**为每个请求新建实例。
- 分词读路径**不加任何锁**：每次 `cut` 在开始时捕获一份不可变词典快照，之后只读该快照。
- 词典写操作（`add_word` / `del_word` / `load_userdict` / `initialize`）采用**写时复制**
  （copy-on-write）原子换入新快照，进行中的切分不受影响。
- 单线程默认调用路径行为与性能不变（见文末压测数字）。

## 改造范围：共享状态分类

### 改为只读（快照化）

- `Tokenizer` 的词典状态 `FREQ` / `total` / 强制切分词集合：合并为不可变快照对象
  `_DictState(freq, total, force_split_words, version)`（`jieba/__init__.py`）。
  快照发布后不再被修改；`cut` / `cut_for_search` / `tokenize` / `posseg` 在调用开始时
  捕获一份快照并贯穿整个切分过程（`get_DAG`、`calc`、私有切分函数均接收该快照参数），
  因此一次切分内部看到的词典永远是一致的。
- `jieba.posseg` 的词性表 `word_tag_tab`：先在局部变量构建完成后整体换入，
  不再出现"边填边读"的半加载表。
- `jieba.analyse.tfidf` 的 IDF 表 `idf_freq` / `median_idf`：同样改为构建完成后原子换入。

### 按实例隔离（消除模块级共享）

- `jieba.finalseg.Force_Split_Words`（模块级全局集合）：`Tokenizer` 不再读写它，
  改为每个实例自己的 `force_split_words`（保存在 `_DictState` 内，随快照一起换入）。
  `finalseg.cut(sentence, force_split_words=None)` 新增可选参数；不传时仍回退到模块级集合，
  保持向后兼容。这修复了"一个实例 `del_word` 会污染同进程其他实例"的串扰。
- `DAG`、`route`、缓冲区等本来就是调用局部变量，天然按线程隔离，未改动。

### 仍然加锁（但不串行化分词）

- `Tokenizer._write_lock`：只串行化**写操作**（词典变更、缓存开关），读路径不碰它。
- `Tokenizer.lock`（原有）：仅用于 `initialize` / `set_dictionary` 的初始化串行化。
- `_DICT_WRITING_GUARD`（模块级）：保护模块级 `DICT_WRITING` 字典本身的
  查-改-删，修复多线程同时 `initialize` 时各自创建不同锁、重复加载词典的竞态。
- `_CutResultCache._lock`：仅保护缓存索引的短临界区（get/put/淘汰），
  切分计算在锁外进行。
- `finalseg._force_split_lock`：保护遗留的模块级 `add_force_split` API。

没有引入任何覆盖整个切分流程的大锁；并发分词吞吐随线程数近似线性提升（见压测）。

## 并发词典加载的语义

- `load_userdict` 先把整个文件解析完、算出缺省词频，然后**一次性原子生效**：
  不会出现"一半新词已生效"的中间状态被读到。
- 在变更发生前已经开始的切分，继续使用旧快照直到结束（结果与变更前完全一致）；
  变更完成后开始的切分，看到完整的新词典。单个 `lcut` 调用不会跨快照。
- `add_word` / `del_word` 同理：单次调用原子生效。
- 多个线程同时写词典时，写操作按 `_write_lock` 串行生效，每个中间状态都是完整快照。

## 分词结果缓存

- `Tokenizer.enable_cut_cache(maxsize)` 开启可选的 LRU 结果缓存（默认关闭，
  默认路径行为与性能完全不变）；`disable_cut_cache()` 关闭。
- 缓存键包含 `(是否搜索模式, 句子, cut_all, HMM)`，不同文本/不同模式互不串结果。
- 每个缓存条目带有生成它的词典快照版本号；词典一旦变更，旧版本条目**永不命中**
  （懒淘汰），并同步清空缓存。
- 容量达到上限时按 LRU 淘汰，淘汰只是移除索引引用，不影响正在进行的切分
  （进行中的切分持有自己的结果对象）。

## 并发冲突日志

词典在"仍有切分在进行"时被修改（`add_word` / `load_userdict` / `set_dictionary` /
重新 `initialize`），会记录一条 WARNING 日志，包含实例与在飞切分数，便于定位：

```
jieba: dictionary of <Tokenizer dictionary=None> changed while 4 cut(s) are in flight; \
in-flight cuts finish with the previous dictionary snapshot.
```

该行为本身不是错误（在飞切分会用旧快照正常完成），日志用于追踪结果变更的来源。

## 不在本次范围内

- `jieba.enable_parallel()` / `disable_parallel()`（多进程模式）是进程级开关，
  不应与其他调用并发切换。
- `jieba.analyse.set_stop_words` 运行期修改停用词集合不做快照隔离（建议启动期配置）。
- paddle 分支（`use_paddle=True`）的线程安全性取决于 paddle 本身，结果不进入缓存。

## 验证方式

一致性测试（11 个用例：多线程一致、多模式并发、并发加载词典、缓存隔离与失效、
长短文本混合、重复稳定性、模块级状态隔离、初始化竞态）：

```bash
python test/test_thread_safety.py
```

压测（单线程 vs 8 线程吞吐与结果比对、重复稳定性、共享实例 vs 每请求实例的峰值内存）：

```bash
python test/bench_concurrent.py
```
