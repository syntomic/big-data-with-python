from pyflink.common import Row, Types
from pyflink.datastream import RuntimeContext, ProcessFunction

from src.main.python.com.syntomic.bigdata.abstract_job import AbstractJob

class PurePyFlink(AbstractJob):

    def run(self):

        self.tableEnv.get_config().get_configuration().set_integer("parallelism.default", 1)
        self.create_tables()

        source = self.tableEnv.to_data_stream(self.tableEnv.from_path("source"))
        transform = source.process(MyProcessFunction(), Types.ROW([Types.INT()]))
        self.tableEnv.from_data_stream(transform).execute_insert("sink").wait()

    def create_tables(self):
        self.tableEnv.execute_sql("""
            CREATE TABLE IF NOT EXISTS `origin` (
                `sepal_length` FLOAT,
                `sepal_width` FLOAT,
                `petal_length` FLOAT,
                `petal_width` FLOAT
            ) WITH (
                'connector' = 'datagen',
                'rows-per-second' = '1',
                'fields.sepal_length.max' = '8',
                'fields.sepal_length.min' = '4',
                'fields.sepal_width.max' = '5',
                'fields.sepal_width.min' = '2',
                'fields.petal_length.max' = '7',
                'fields.petal_length.min' = '1',
                'fields.petal_width.max' = '3',
                'fields.petal_width.min' = '0'
            )
        """)

        self.tableEnv.execute_sql("""
            CREATE TABLE IF NOT EXISTS `sink` (
                `classifier` INT
            ) WITH (
                'connector' = 'print'
            )
        """)

        self.tableEnv.execute_sql("""
            CREATE VIEW `source` AS
            SELECT
                ARRAY[`sepal_length`, `sepal_width`, `petal_length`, `petal_width`]
            FROM
                `origin`
        """)



class MyProcessFunction(ProcessFunction):

    def open(self, runtime_context: RuntimeContext):

        script = runtime_context.get_job_parameter("python.script", "./src/main/resources/script/demo_script.py")
        arg = runtime_context.get_job_parameter("python.arg", "./src/main/resources/model/demo.joblib")

        with open(script, "r") as f:
            code = f.read()
        exec(code, globals())

        self.model = user_open(arg)


    def process_element(self, value: Row, ctx: 'ProcessFunction.Context'):
        # ! Python 返回一个可迭代对象
        return [Row(user_eval(self.model, value))]