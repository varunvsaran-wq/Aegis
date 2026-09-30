"""Study 1 data: labelled texts for prompt-injection screening.

Three sources, each a list of :class:`Example` with label 1 = injection:

- ``deepset``: deepset/prompt-injections (Apache-2.0). Direct injections and
  benign requests, English and German. Its own train/test files are pooled and
  re-split by hash, because its test file alone (116 texts) is too small.
- ``notinject``: leolee99/NotInject (MIT). 339 *benign* prompts built around
  trigger words ("ignore", "bypass"...). Every flag on it is a false alarm, so it
  measures over-defence and Jev's documented weakness with negation.
- ``indirect``: HotpotQA paragraphs, each included twice: clean (label 0) and
  with an answer-override note planted after the first sentence (label 1), using
  the Aegis templates in :mod:`aegis.eval.poison`. Dev uses the ``dev`` templates
  and test the ``heldout`` ones, so no system is tuned on the test phrasings.

Splits are deterministic: a text goes to dev when the first 8 hex digits of
``sha256(source + key)`` fall below ``DEV_FRACTION``. Clean/poisoned pairs share
a key, so both copies of a paragraph land in the same split.
"""

from __future__ import annotations

import hashlib
import random
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache"
DEV_FRACTION = 0.3

HF = "https://huggingface.co/datasets"
FILES = {
    "deepset_train.parquet": f"{HF}/deepset/prompt-injections/resolve/main/data/"
                             "train-00000-of-00001-9564e8b05b4757ab.parquet",
    "deepset_test.parquet": f"{HF}/deepset/prompt-injections/resolve/main/data/"
                            "test-00000-of-00001-701d16158af87368.parquet",
    "notinject_one.parquet": f"{HF}/leolee99/NotInject/resolve/main/data/"
                             "NotInject_one-00000-of-00001.parquet",
    "notinject_two.parquet": f"{HF}/leolee99/NotInject/resolve/main/data/"
                             "NotInject_two-00000-of-00001.parquet",
    "notinject_three.parquet": f"{HF}/leolee99/NotInject/resolve/main/data/"
                               "NotInject_three-00000-of-00001.parquet",
}

SOURCES = ("deepset", "notinject", "indirect")


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: int
    source: str
    category: str
    split: str

    def to_dict(self) -> dict:
        return asdict(self)


def split_of(source: str, key: str, dev_fraction: float = DEV_FRACTION) -> str:
    h = int(hashlib.sha256(f"{source}\x00{key}".encode()).hexdigest()[:8], 16)
    return "dev" if h / 0xFFFFFFFF < dev_fraction else "test"


def _fetch(name: str) -> Path:
    path = CACHE_DIR / name
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FILES[name], path)
    return path


def _read(name: str):
    import pandas as pd

    return pd.read_parquet(_fetch(name))


def _id(source: str, key: str) -> str:
    return f"{source}:{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def load_deepset() -> list[Example]:
    import pandas as pd

    df = pd.concat([_read("deepset_train.parquet"), _read("deepset_test.parquet")])
    out, seen = [], set()
    for text, label in zip(df["text"], df["label"]):
        text = str(text).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(Example(id=_id("deepset", text), text=text, label=int(label),
                           source="deepset", category="direct",
                           split=split_of("deepset", text)))
    return out


def load_notinject() -> list[Example]:
    out, seen = [], set()
    for level in ("one", "two", "three"):
        df = _read(f"notinject_{level}.parquet")
        for text, cat in zip(df["prompt"], df["category"]):
            text = str(text).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(Example(id=_id("notinject", text), text=text, label=0,
                               source="notinject", category=f"{cat} ({level} trigger)",
                               split=split_of("notinject", text)))
    return out


def plant(text: str, note: str) -> str:
    """Insert ``note`` after the first sentence, as the Aegis poisoner does."""
    first, _, rest = text.partition(". ")
    return f"{first}. {note} {rest}".strip() if rest else f"{text} {note}"


def build_indirect(paragraphs: list[tuple[str, str]], seed: int = 0) -> list[Example]:
    """Clean + poisoned copies of each ``(title, text)`` paragraph."""
    from aegis.eval.poison import FAKE_ANSWERS, HELDOUT_OVERRIDE_TEMPLATES, OVERRIDE_TEMPLATES

    rng = random.Random(seed)
    counters = {"dev": 0, "test": 0}
    out = []
    for title, text in sorted(paragraphs):
        split = split_of("indirect", title)
        templates = OVERRIDE_TEMPLATES if split == "dev" else HELDOUT_OVERRIDE_TEMPLATES
        t_idx = counters[split] % len(templates)
        counters[split] += 1
        note = templates[t_idx].format(title=title, fake=rng.choice(FAKE_ANSWERS))
        tag = "dev" if split == "dev" else "heldout"
        out.append(Example(id=_id("indirect", f"clean|{title}"), text=text, label=0,
                           source="indirect", category="clean paragraph", split=split))
        out.append(Example(id=_id("indirect", f"poison|{title}"), text=plant(text, note),
                           label=1, source="indirect", category=f"{tag}-{t_idx}",
                           split=split))
    return out


def load_indirect(n_paragraphs: int = 200, seed: int = 0) -> list[Example]:
    from aegis.eval.benchmarks.hotpotqa import build_corpus, load_hotpotqa

    questions = load_hotpotqa(n=max(20, n_paragraphs // 5), seed=seed)
    chunks = [c for c in build_corpus(questions) if len(c.sentences) >= 2
              and 150 <= len(c.text) <= 1500]
    chunks = sorted(chunks, key=lambda c: c.id)
    random.Random(seed).shuffle(chunks)
    return build_indirect([(c.title, c.text) for c in chunks[:n_paragraphs]], seed=seed)


LOADERS = {"deepset": load_deepset, "notinject": load_notinject, "indirect": load_indirect}


def load(sources: tuple[str, ...] | list[str] = SOURCES, split: str | None = None) -> list[Example]:
    out: list[Example] = []
    for s in sources:
        out.extend(LOADERS[s]())
    if split:
        out = [e for e in out if e.split == split]
    return out
