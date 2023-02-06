package com.syntomic.bigdata.utils;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

import org.apache.commons.lang3.text.StrSubstitutor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.configuration.ConfigOptions;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.types.utils.TypeConversions;
import org.apache.flink.types.Row;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** SQL工具类 */
public class SQLUtil {

    public static final Logger LOG = LoggerFactory.getLogger(SQLUtil.class);

    public static final ConfigOption<String> EXECUTE_SQL =
            ConfigOptions.key("sql.execute-sql")
                    .stringType()
                    .noDefaultValue()
                    .withDescription("SQL file name");

    /**
     * 读取资源文件夹下的SQL文件
     * @param conf sql中变量配置
     * @return 可执行的sql语句列表
     */
    public static String[] readSqls(Configuration conf) throws IOException {
        // 读取sql文件
        String sqlFile = new String(Files.readAllBytes(Paths.get(conf.getString(EXECUTE_SQL))), StandardCharsets.UTF_8);

        // 替换参数与注释
        sqlFile = new StrSubstitutor(conf.toMap()).replace((sqlFile));
        sqlFile = Pattern.compile("--.*$", Pattern.MULTILINE).matcher(sqlFile).replaceAll("").trim();

        // 得到具体执行的sql语句
        return Pattern.compile("; *$", Pattern.MULTILINE).split(sqlFile);
    }

    /**
     * 脱敏显示sql语句
     * @param sql sql语句
     * @return 脱敏后sql语句
     */
    public static String maskSql(String sql) {
        // 密码隐藏显示
        sql = sql.replaceAll("(password.*?=[ '\"]+)\\w+('|\")", "$1*****$2");

        // 显示行号
        String[] lines = sql.split("\r\n|\r|\n");
        String[] linesWithNum =
                IntStream.range(1, lines.length + 1)
                        .mapToObj(i -> i + "\t" + lines[i - 1])
                        .toArray(String[]::new);


        return System.getProperty("line.separator")
            + String.join(System.getProperty("line.separator"), linesWithNum);
    }


    /**
     * 执行多条sql语句文件
     * @param tableEnv sql执行环境
     * @param conf 执行配置
     */
    public static void executeSql(TableEnvironment tableEnv, Configuration conf) throws Exception {
        String[] sqls = readSqls(conf);

        for (String sql: sqls) {
            sql = sql.trim();
            LOG.info(maskSql(sql));
            executeSql(tableEnv, sql);
        }
    }


    /**
     * 执行单条sql语句
     * @param tableEnv sql执行环境
     * @param sql sql语句
     */
    public static void executeSql(TableEnvironment tableEnv, String sql) {
        // 配置设置
        if (sql.startsWith("SET")) {
            String[] setConfig = sql.replaceAll("SET +", "").split("=");
            tableEnv.getConfig().getConfiguration().setString(
                setConfig[0].replaceAll("['\"]", "").trim(),
                setConfig[1].replaceAll("['\"]", "").trim());
        } else {
            tableEnv.executeSql(sql);
        }
    }

    /**
     * 得到表的RowTypeInformation
     * @param table 表
     * @return A {@link TypeInformation}
     */
    @SuppressWarnings({"deprecation", "unchecked"})
    public static TypeInformation<Row> getRowTypeInfo(Table table) {
        return (TypeInformation<Row>)
                TypeConversions.fromDataTypeToLegacyInfo(
                        table.getResolvedSchema().toPhysicalRowDataType());
    }

    private SQLUtil() {}
}
