package com.syntomic.bigdata;

import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

/** 作业基类 */
public abstract class AbstractJob {

    protected Configuration conf;
    protected StreamExecutionEnvironment env;
    protected StreamTableEnvironment tableEnv;

    public abstract void run() throws Exception;

    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    public void setEnv(StreamExecutionEnvironment env) {
        this.env = env;
    }

    public void setTableEnv(StreamTableEnvironment tableEnv) {
        this.tableEnv = tableEnv;
    }
}
