## BigData With Python
- 目的：探讨大数据场景下使用Python不同技术方案的优劣
    - 实现
    - 性能

- 需求：近线模型预测服务
    - 输入: kafka实时数据流
    - 处理：调用算法脚本实现数据处理及预测
    - 输出：预测结果

- 方案
    - Flink 1.16
        - Flink SQL + Python UDF
        - PyFlink + DataStream API
        - Flink + DataStream API：pemja
    - Spark 2.4
        - PySpark

- 执行模式
    - 流 vs 批
    - 进程 vs 线程
    - chain vs disable chain

- 性能
    - 批处理耗时
    - 流消费速率

- 资源消耗
    - CPU
    - 内存

## 实验
- 环境配置
    - Maven 3.8.6
    - Java 1.8
    - Python 3.8: `pip install -r requirements.txt`

- demo：本地运行
    - PyFlink + DataStream API: `python -m src.main.python.com.syntomic.bigdata '{"job.name":"PurePyFlink","module.name":"pure_pyflink"}'`
    - 运行参数
        - Flink SQL + Python UDF: `{"rest.port":"8082","job.name":"FlinkSQLWithPythonUDF","sql.execute-sql":"./src/main/resources/sql/demo.sql","python.executable":"${PATHOFPYTHON}","python.client.executable":"$${PATHOFPYTHON}","python.files":"./src/main/python/com/syntomic/bigdata"}`
        - Flink + Pemja: `{"rest.port":"8082","job.name":"FlinkWithPemja","sql.execute-sql":"./src/main/resources/sql/demo_connector.sql","python.executable":"${PATHOFPYTHON}"}`