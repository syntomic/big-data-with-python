from pyflink.table.udf import FunctionContext, ScalarFunction, udf
from pyflink.table.types import DataTypes


class DemoUDF(ScalarFunction):

    def __init__(self, arg:str=None, script:str=None):
        self.arg = arg
        self.script = script

    def open(self, function_context: FunctionContext):
        # ! Python UDF中不能获得全局参数，只能初始化参数
        with open(self.script, "r") as f:
            code = f.read()
        exec(code, globals())

        self.model = user_open(self.arg)

    def eval(self, value):
        return user_eval(self.model, value)


demo_udf = udf(DemoUDF(
    "./src/main/resources/model/demo.joblib",
    "./src/main/resources/script/demo_script.py"
), result_type=DataTypes.INT())