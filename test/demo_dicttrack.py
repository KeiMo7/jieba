#encoding=utf-8
"""
词典版本记录与回放演示：模拟"运营两周后往自定义词典加了新词"的场景。

运行：python demo_dicttrack.py
"""
from __future__ import print_function, unicode_literals

import json
import os
import shutil
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import jieba

SENT = "李小福是创新办主任也是云计算方面的专家，标题里有韩玉赏鉴和八一双鹿"

store_dir = os.path.join(tempfile.gettempdir(), "jieba_dicttrack_demo")
shutil.rmtree(store_dir, ignore_errors=True)

# 两周前的词典
dict_v1 = os.path.join(store_dir, "userdict_v1.txt")
# 两周后的词典：运营加了新词、调了词频
dict_v2 = os.path.join(store_dir, "userdict_v2.txt")
os.makedirs(store_dir)
with open(dict_v1, "wb") as f:
    f.write("云计算 5\n李小福 2 nr\n创新办 3 i\n".encode("utf-8"))
with open(dict_v2, "wb") as f:
    f.write("云计算 5\n李小福 2 nr\n创新办 3 i\n韩玉赏鉴 3 nz\n八一双鹿 3 nz\n云计算 500\n".encode("utf-8"))

jieba.enable_dict_tracking(store_dir)

print("=" * 60)
print("1) 两周前：加载 v1 词典并分词")
print("=" * 60)
jieba.load_userdict(dict_v1)
v1 = jieba.dict_version()
r1 = jieba.lcut_with_version(SENT)
print("分词结果:", "/".join(r1["words"]))
print("词典版本:", r1["dict_version"])

print()
print("=" * 60)
print("2) 两周后：运营更新词典（加词 + 调词频），加载 v2")
print("=" * 60)
jieba.load_userdict(dict_v2)
v2 = jieba.dict_version()
r2 = jieba.lcut_with_version(SENT)
print("分词结果:", "/".join(r2["words"]))
print("词典版本:", r2["dict_version"])
print("两次结果是否基于同一词典状态:", r1["dict_version"] == r2["dict_version"])

print()
print("=" * 60)
print("3) 词典版本记录（每次加载：来源/条目数/摘要/时间/冲突）")
print("=" * 60)
for rec in jieba.dict_history():
    print(json.dumps(rec, ensure_ascii=False, indent=2, sort_keys=True))

print()
print("=" * 60)
print("4) 两个版本的差异（新增/删除/词频从多少到多少）")
print("=" * 60)
diff = jieba.dict_diff(v1, v2)
print(json.dumps({
    "summary": diff["summary"],
    "added": diff["added"],
    "removed": diff["removed"],
    "changed": diff["changed"],
}, ensure_ascii=False, indent=2, sort_keys=True))

print()
print("=" * 60)
print("5) 用 v1 版本回放当时的分词结果")
print("=" * 60)
replayed = jieba.replay_lcut(SENT, v1)
print("回放结果:", "/".join(replayed))
print("与当时结果一致:", replayed == r1["words"])
