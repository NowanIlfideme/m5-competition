"""M5 competition: time series models."""

from pathlib import Path


__all__ = ["__root__", "__version__", "__data__"]


__root__ = Path(__file__).parent.absolute()
__data__ = __root__.parent / "data"

with (__root__ / "VERSION").open() as f:
    __version__ = f.read()
