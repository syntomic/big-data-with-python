from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession

spark = SparkSession.builder \
        .appName("test") \
        .config("master","local[4]") \
        .enableHiveSupport() \
        .getOrCreate()

# 创建源表
df = spark.createDataFrame([(5.1, 3.5, 1.4, 0.2), [5.7, 3., 4.2, 1.2]], ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df.createOrReplaceTempView("source")

# 编写UDF
with open("./src/main/resources/script/demo_script.py", "r") as f:
    code = f.read()

exec(code, globals())
model = user_open("./src/main/resources/model/demo.joblib")

def demo_udf(data):
    return user_eval(model, data)

spark.udf.register("demo_udf", demo_udf, returnType=IntegerType())

# 执行SQL
result = spark.sql("select demo_udf(array(sepal_length, sepal_width, petal_length, petal_width)) from source")
result.show()