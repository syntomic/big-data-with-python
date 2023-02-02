## 实验
- 目的：探讨大数据场景下使用Python不同技术方案的优劣
    - 实现
    - 性能

- 流程：真实线上数据但进行一定简化
    - 输入: g83 replay线上数据
    - 处理：相同用户python脚本
    - 输出：简化命中信息

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
    - chain vs disable

- 性能
    - 耗时
    - 消费速率

- 资源消耗
    - CPU
    - 内存