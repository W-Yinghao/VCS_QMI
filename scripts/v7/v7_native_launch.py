"""Run an EvalMuse@45c8bb4 entry script unmodified from the read-only tree: put the tree on sys.path (the scripts import `lavis`), keep the
caller's working directory (it holds the 'bert-base-uncased' link and, for the preprocessing scripts, their 'dataset/' files), and execute
the script as __main__ with the given arguments.  Nothing is written into the tree.
    python [-I] scripts/v7/v7_native_launch.py <train.py | eval.py | process/process_train.py | process/element2mask.py> [args ...]"""
import runpy
import sys

FGA = "/projects/EEG-foundation-model/yinghao/third_party/EvalMuse@45c8bb4"

if __name__ == "__main__":
    script = sys.argv[1]; sys.path.insert(0, FGA); sys.argv = [f"{FGA}/{script}"] + sys.argv[2:]
    runpy.run_path(f"{FGA}/{script}", run_name="__main__")
