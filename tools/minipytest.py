"""A very small pytest-compatible runner.

Only exists so the suite can be executed where pytest is not installed. It implements
the subset the tests use: fixtures (including autouse), parametrize, raises and approx.
Use real pytest when you have it -- `python -m pytest` from the package root.
"""

from __future__ import annotations

import inspect
import re
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path


# ---------------------------------------------------------------- pytest shim
class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(et, self.exc):
            return False
        if self.match and not re.search(self.match, str(ev)):
            raise AssertionError(f"pattern {self.match!r} not found in {str(ev)!r}")
        self.value = ev
        return True


class _Approx:
    def __init__(self, v, abs=None, rel=None):
        self.v, self.abs, self.rel = v, abs, rel

    def __eq__(self, other):
        tol = self.abs if self.abs is not None else 1e-6
        return abs(float(other) - float(self.v)) <= tol


class _MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value=None):
        if value is None and isinstance(target, str):
            mod, _, name = target.rpartition(".")
            target = sys.modules[mod]
        old = getattr(target, name)
        self._undo.append(lambda: setattr(target, name, old))
        setattr(target, name, value)

    def setitem(self, dic, key, value):
        had, old = key in dic, dic.get(key)
        self._undo.append(lambda: dic.__setitem__(key, old) if had else dic.pop(key, None))
        dic[key] = value

    def undo(self):
        while self._undo:
            self._undo.pop()()


def _make_pytest_module():
    m = types.ModuleType("pytest")
    m.raises = _Raises
    m.approx = lambda v, abs=None, rel=None: _Approx(v, abs, rel)
    m.skip = lambda msg="": (_ for _ in ()).throw(_Skip(msg))

    registry = {}

    def fixture(func=None, **kw):
        def wrap(f):
            f._is_fixture = True
            f._autouse = kw.get("autouse", False)
            registry[f.__name__] = f
            return f
        return wrap(func) if func else wrap

    m.fixture = fixture
    m._registry = registry

    class _Mark:
        @staticmethod
        def parametrize(argnames, argvalues):
            names = [a.strip() for a in argnames.split(",")]

            def deco(f):
                f._params = (names, argvalues)
                return f
            return deco

        def __getattr__(self, _):
            return lambda *a, **k: (lambda f: f)

    m.mark = _Mark()
    return m


class _Skip(Exception):
    pass


# ---------------------------------------------------------------- runner
def run(test_dir: Path, pattern="test_*.py", filt=None):
    pytest_mod = _make_pytest_module()
    sys.modules["pytest"] = pytest_mod
    sys.path.insert(0, str(test_dir))
    sys.path.insert(0, str(test_dir.parent))

    fixtures = {}
    conftest = test_dir / "conftest.py"
    if conftest.exists():
        spec = _load(conftest, "conftest")
        for name, obj in vars(spec).items():
            if callable(obj) and getattr(obj, "_is_fixture", False):
                fixtures[name] = obj

    passed = failed = skipped = 0
    failures = []
    for path in sorted(test_dir.glob(pattern)):
        mod = _load(path, path.stem)
        local = dict(fixtures)
        for name, obj in vars(mod).items():
            if callable(obj) and getattr(obj, "_is_fixture", False):
                local[name] = obj
        tests = [(n, f) for n, f in vars(mod).items()
                 if n.startswith("test_") and callable(f)]
        for name, fn in tests:
            cases = _expand(fn)
            for label, kwargs in cases:
                full = f"{path.stem}::{name}{label}"
                if filt and filt not in full:
                    continue
                try:
                    _run_one(fn, kwargs, local)
                    passed += 1
                except _Skip:
                    skipped += 1
                except Exception:
                    failed += 1
                    failures.append((full, traceback.format_exc()))
                    print(f"F {full}")
    for name, tb in failures:
        print(f"\n{'=' * 70}\nFAIL {name}\n{'-' * 70}\n{tb}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return failed == 0


def _expand(fn):
    params = getattr(fn, "_params", None)
    if not params:
        return [("", {})]
    names, values = params
    out = []
    for v in values:
        v = v if isinstance(v, (tuple, list)) else (v,)
        out.append(("[" + ",".join(str(x)[:18] for x in v) + "]", dict(zip(names, v))))
    return out


def _run_one(fn, params, fixtures):
    sig = inspect.signature(fn)
    tmp = Path(tempfile.mkdtemp())
    finalisers = []
    values = dict(params)
    try:
        auto = [f for f in fixtures.values() if getattr(f, "_autouse", False)]
        pending = list(sig.parameters) + [f.__name__ for f in auto]
        resolved = {}

        def resolve(name):
            if name in values:
                return values[name]
            if name in resolved:
                return resolved[name]
            if name == "tmp_path":
                resolved[name] = tmp
            elif name == "monkeypatch":
                mp = _MonkeyPatch()
                finalisers.append(mp.undo)
                resolved[name] = mp
            elif name in fixtures:
                f = fixtures[name]
                kw = {p: resolve(p) for p in inspect.signature(f).parameters}
                if inspect.isgeneratorfunction(f):
                    gen = f(**kw)
                    resolved[name] = next(gen)
                    finalisers.append(lambda g=gen: next(g, None))
                else:
                    resolved[name] = f(**kw)
            else:
                raise RuntimeError(f"unknown fixture {name!r}")
            return resolved[name]

        for p in pending:
            resolve(p)
        fn(**{p: resolve(p) for p in sig.parameters})
    finally:
        for f in reversed(finalisers):
            try:
                f()
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    ok = run(root / "tests", filt=sys.argv[1] if len(sys.argv) > 1 else None)
    sys.exit(0 if ok else 1)
