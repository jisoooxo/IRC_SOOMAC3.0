"""Run the existing closed-loop scenario harness with prefix cache OFF or ON (no harness changes)."""
import sys

ROOT = "/home/roma/IRC_SOOMAC3.0"
sys.path[:0] = [f"{ROOT}/src/soomac_irc", f"{ROOT}/src/soomac_irc/test"]

mode = sys.argv.pop(1)
assert mode in {"off", "on"}
import soomac_irc.prefix_cache as prefix_cache
prefix_cache.PREFIX_CACHE_ENABLED = mode == "on"
print(f"PREFIX_CACHE_ENABLED={prefix_cache.PREFIX_CACHE_ENABLED}", flush=True)

import manual_scenario_stack_probe as probe
probe.main()
