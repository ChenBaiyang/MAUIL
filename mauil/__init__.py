"""MAUIL: multilevel attribute embedding and regularized CCA alignment."""

from .model import MAUIL
from .evaluation import hit_precision, split_anchors
from .pretrained import load_pretrained_vectors, load_word_vectors

__version__ = "2.0"
__all__ = ["MAUIL", "hit_precision", "split_anchors", "load_pretrained_vectors", "load_word_vectors"]
