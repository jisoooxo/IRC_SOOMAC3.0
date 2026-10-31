import sys
sys.path.insert(0, sys.argv[1])
from collections import Counter
import transformers
from transformers import AutoConfig, DynamicCache
config = AutoConfig.from_pretrained("/home/roma/Desktop/sLLM/gemma-4-12B-it", local_files_only=True)
cache = DynamicCache(config=config)
print("Transformers:", transformers.__version__, transformers.__file__)
print("Cache layers:", len(cache.layers), dict(Counter(type(l).__name__ for l in cache.layers)))
