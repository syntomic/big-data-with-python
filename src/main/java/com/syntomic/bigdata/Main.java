package com.syntomic.bigdata;

import java.util.Map;

import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.configuration.ConfigOptions;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.core.type.TypeReference;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

/** 统一作业入口 */
public class Main {

    private static final String CLASS_PREFIX = "com.syntomic.bigdata.impls.";

    /** 作业类的名称 */
    public static final ConfigOption<String> JOB_NAME =
            ConfigOptions.key("job.name")
                    .stringType()
                    .defaultValue("WordCountJob")
                    .withDescription("The Class Name of Job");
    public static void main(String[] args) throws Exception {
        // 解析参数
        ObjectMapper mapper = new ObjectMapper();
        Configuration conf = Configuration.fromMap(
            mapper.readValue(args[0], new TypeReference<Map<String, String>>() {}));

        // 根据参数构建作业实例
        String className = CLASS_PREFIX + conf.get(JOB_NAME);
        AbstractJob job = (AbstractJob) Class.forName(className).newInstance();

        // 设置作业相关参数并运行
        job.setConf(conf);
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment(conf);
        env.getConfig().setGlobalJobParameters(conf);
        job.setEnv(env);
        job.setTableEnv(StreamTableEnvironment.create(env));
        job.run();
    }
}
