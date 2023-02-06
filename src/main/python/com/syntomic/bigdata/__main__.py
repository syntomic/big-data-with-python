import importlib
import json
import sys
from typing import Dict

from src.main.python.com.syntomic.bigdata.abstract_job import AbstractJob


def main():
    """统一模块入口"""
    conf: Dict[str, str] = json.loads(sys.argv[1])

    module_name = f"src.main.python.com.syntomic.bigdata.impls.{conf.get('module.name')}"
    class_name = conf.get("job.name")

    job: AbstractJob = getattr(importlib.import_module(module_name), class_name)(conf)
    job.run()


if __name__ == "__main__":
    main()