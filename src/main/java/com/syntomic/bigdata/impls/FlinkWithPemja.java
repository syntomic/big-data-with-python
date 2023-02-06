package com.syntomic.bigdata.impls;

import static org.apache.flink.python.PythonOptions.PYTHON_EXECUTABLE;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.configuration.ConfigOptions;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;

import com.syntomic.bigdata.AbstractJob;
import com.syntomic.bigdata.utils.SQLUtil;

import pemja.core.PythonInterpreter;
import pemja.core.PythonInterpreterConfig;

/** 利用Flink加上Pemja(FFI)处理数据 */
public class FlinkWithPemja extends AbstractJob {

    public static final ConfigOption<String> PYTHON_ARG =
            ConfigOptions.key("python.arg")
                    .stringType()
                    .defaultValue("./src/main/resources/model/demo.joblib")
                    .withDescription("The init python arg");

    public static final ConfigOption<String> PYTHON_SCRIPT =
            ConfigOptions.key("python.script")
                    .stringType()
                    .defaultValue("./src/main/resources/script/demo_script.py")
                    .withDescription("The user's python script");

    @Override
    public void run() throws Exception {
        // 创建相应表
        SQLUtil.executeSql(tableEnv, conf);

        // 处理数据
        DataStream<Row> source = tableEnv.toDataStream(tableEnv.from("source"));
        DataStream<Row> transform = source.process(new MyProcessFunction())
                .returns(SQLUtil.getRowTypeInfo(tableEnv.from("sink")));

        // 执行
        tableEnv.fromDataStream(transform).executeInsert("sink");
    }


    private static class MyProcessFunction extends ProcessFunction<Row, Row> {

        private transient PythonInterpreter interpreter = null;

        @Override
        public void open(Configuration parameters) throws Exception {
            Configuration conf = (Configuration) getRuntimeContext().getExecutionConfig().getGlobalJobParameters();

            PythonInterpreterConfig config =
                    PythonInterpreterConfig.newBuilder()
                            .setPythonExec(conf.getString(PYTHON_EXECUTABLE))
                            .build();

            interpreter = new PythonInterpreter(config);

            // 执行用户脚本
            interpreter.exec(new String(Files.readAllBytes(Paths.get(conf.getString(PYTHON_SCRIPT))), StandardCharsets.UTF_8));

            // 执行初始化方法
            // ! invoke方法有bug: https://github.com/alibaba/pemja/issues/26
            interpreter.set("arg", conf.getString(PYTHON_ARG));
            interpreter.exec("model=user_open(arg)");

        }

        @Override
        public void processElement(Row value, ProcessFunction<Row, Row>.Context ctx, Collector<Row> out)
                throws Exception {

            interpreter.set("data", value.getField(0));
            interpreter.exec("predict=user_eval(model, data)");
            out.collect(Row.of(interpreter.get("predict", Integer.class)));
        }
    }
}