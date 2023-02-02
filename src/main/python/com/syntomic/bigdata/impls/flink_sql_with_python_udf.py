
def run():

    sql = """
        INSERT INTO `sink`
        SELECT
            `time`,
            `role_id`,
            `replay`,
            `prob`
        FROM
            `source`
        WHERE
            JSON_VALUE(`relation_data`, '$.server') = '10005'
    """