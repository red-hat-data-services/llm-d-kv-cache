# Minimal torch stub for protocol-only vllm usage (no GPU).
# Intercepts all `import torch*` with stub modules so vllm can be
# imported without PyTorch. CUDA op registration becomes a no-op.
# Remove once https://github.com/vllm-project/vllm/issues/38925 is fixed.

import importlib.abc
import importlib.machinery
import importlib.metadata
import sys
import types


class _NeverMatch(type):
    """Metaclass whose isinstance/issubclass always return False."""

    def __instancecheck__(cls, instance):
        return False

    def __subclasscheck__(cls, subclass):
        return False


class device(metaclass=_NeverMatch):
    pass


class dtype(metaclass=_NeverMatch):
    pass


class Tensor(metaclass=_NeverMatch):
    pass


class _Proxy:
    """Recursive no-op proxy for chained attribute access."""

    def __getattr__(self, name):
        return _Proxy()

    def __call__(self, *a, **kw):
        return _Proxy()

    def __bool__(self):
        return False

    def __or__(self, other):
        return _Proxy()

    def __ror__(self, other):
        return _Proxy()

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __mro_entries__(self, bases):
        return (object,)

    def __str__(self):
        return ""

    def __repr__(self):
        return ""

    def __int__(self):
        return 0

    def __float__(self):
        return 0.0

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __mul__(self, other):
        return _Proxy()

    def __rmul__(self, other):
        return _Proxy()

    def __contains__(self, item):
        return False

    def __eq__(self, other):
        return False

    def __hash__(self):
        return 0

    def __fspath__(self):
        return ""

    def __getitem__(self, key):
        return _Proxy()

    def __setitem__(self, key, value):
        pass

    def __class_getitem__(cls, item):
        return _Proxy()


class _StubModule(types.ModuleType):
    """Module that returns _Proxy() for any unknown attribute."""

    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []
        self.__package__ = name
        self.__spec__ = importlib.machinery.ModuleSpec(name, None)

    def __getattr__(self, name):
        return _Proxy()


# Packages to stub: torch/torchvision/torchaudio (no GPU), cv2/xgrammar (native-only, stripped)
_STUB_PREFIXES = ("torch", "torchvision", "torchaudio", "cv2", "xgrammar")


class _TorchFinder(importlib.abc.MetaPathFinder):
    """Import hook: stub packages return no-op modules."""

    def find_spec(self, fullname, path, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in _STUB_PREFIXES):
            return importlib.machinery.ModuleSpec(fullname, _TorchLoader())
        return None


class _TorchLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


def _build_stub():
    sys.meta_path.insert(0, _TorchFinder())

    mod = _StubModule("torch")
    mod.device = device
    mod.dtype = dtype
    mod.Tensor = Tensor
    mod.__version__ = "0.0.0"
    mod._C = _Proxy()
    mod.__spec__ = importlib.machinery.ModuleSpec("torch", _TorchLoader())
    sys.modules["torch"] = mod

    # Fake package metadata so importlib.metadata.version("torch") works
    _orig_from_name = importlib.metadata.Distribution.from_name

    class _TorchDist(importlib.metadata.Distribution):
        def read_text(self, filename):
            if filename == "METADATA":
                return "Metadata-Version: 2.1\nName: torch\nVersion: 0.0.0\n"
            return None

        def locate_file(self, path):
            return path

    @classmethod
    def _patched_from_name(cls, name):
        if name == "torch":
            return _TorchDist()
        return _orig_from_name(name)

    importlib.metadata.Distribution.from_name = _patched_from_name


_build_stub()
