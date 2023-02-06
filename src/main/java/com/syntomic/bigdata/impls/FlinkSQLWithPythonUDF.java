package com.syntomic.bigdata.impls;

import com.syntomic.bigdata.AbstractJob;
import com.syntomic.bigdata.utils.SQLUtil;

/** 利用Flink SQL加上Python UDF处理数据 */
public class FlinkSQLWithPythonUDF extends AbstractJob {
    @Override
    public void run() throws Exception {
        SQLUtil.executeSql(tableEnv, conf);
    }
}
