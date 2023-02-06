from abc import ABC
from typing import Dict

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

class AbstractJob(ABC):

    def __init__(self, conf: Dict[str, str]):
        self.conf = conf
        self.env = StreamExecutionEnvironment.get_execution_environment()
        self.env.get_config().set_global_job_parameters(conf)
        self.tableEnv = StreamTableEnvironment.create(stream_execution_environment=self.env)

    def run(self):
        pass