import numpy as np
import pandas as pd
import json
import joblib

"""Replay预测脚本格式"""
from typing import Dict, Any


def explode_frame_data(frame_data, data, role_id):
    if len(data) == 0:
        data = []
        shoot_no_hit_list = []
    else:
        shoot_no_hit_list = [i[0] for i in data if i[2] == 0 and i[1] == role_id and len(i[3]) == 0]  # 没有命中任何数据
        shoot_hit_list = [i[0] for i in data if i[2] == 0 and i[1] == role_id and len(i[3]) == 1]  # 命中敌人数据

        data = [i for i in data if i[2] == 0 and len(i[3]) > 0 and i[1] == role_id]  # 将所有开枪的时间帧找到,同时是开枪命中的

    # 去除开枪命中的数据
    if len(shoot_no_hit_list) > 0:
        del_index = np.array(shoot_no_hit_list).reshape(1, -1).repeat([len(shoot_hit_list)], axis=0) - np.array(
            shoot_hit_list).reshape(-1, 1)
        index_two = np.unique(np.where((del_index <= 3) & (del_index >= -3))[1])
        mask = np.zeros(len(shoot_no_hit_list), np.bool_)
        mask[index_two] = True
        out_shoot_index = np.array(shoot_no_hit_list)[mask].tolist()

        shoot_no_hit_list = list(set(shoot_no_hit_list).difference(out_shoot_index))
        shoot_no_hit_list.sort()

    if len(frame_data[0]) == 5:
        new_frame = pd.DataFrame(frame_data, columns=['f_index', 'position', 'is_aim', 'aim_info', 'enemy_list'])
    else:
        new_frame = pd.DataFrame(frame_data, columns=['f_index', 'position', 'is_aim', 'enemy_list'])

    new_frame = new_frame.explode('enemy_list')
    new_frame = new_frame.dropna()
    if len(new_frame) == 0:
        return new_frame, {}, shoot_no_hit_list
    try:
        new_frame[['name', 'is_3', 'is_4']] = pd.DataFrame(new_frame['enemy_list'].values.tolist(),
                                                           columns=['0', '1', '2', '3', '4']).loc[:,
                                              ['0', '3', '4']].values

    except ValueError:
        new_frame[['name']] = pd.DataFrame(new_frame['enemy_list'].values.tolist(), columns=['0', '1', '2']).loc[:,
                              ['0']].values

        new_frame['is_3'] = None
        new_frame['is_4'] = None

    new_frame['screen_dis'] = new_frame['is_4'].map(lambda x: np.linalg.norm(x) if x else 2)
    new_frame['position_enemy'] = new_frame['enemy_list'].map(lambda x: x[1])
    dis_3d = np.linalg.norm(np.stack(new_frame['position'].values) - np.stack(new_frame['position_enemy'].values),
                            axis=-1)
    new_frame['dis_3d'] = dis_3d

    new_frame['is_3'] = new_frame['is_3'].fillna(0)  # 不可见的全部填充为0
    new_frame = new_frame.sort_values(['name', 'f_index'])  # 排序

    try:
        enemy_shoot = pd.DataFrame(data).explode(3).sort_values([3, 0]).groupby(3)[0].unique().to_dict()
    except:
        enemy_shoot = {}

    return new_frame, enemy_shoot, shoot_no_hit_list


def check_list_again(frame, shoot_no_hit_list):
    drop_list = []
    new_frame = frame.copy()
    for index_ in shoot_no_hit_list:
        enemy_name, f_start, f_end = index_[0], index_[1], index_[2]+10
        new_frame_sub = new_frame.query('f_index>=@f_start and f_index <=@f_end and screen_dis <0.1 ')
        if new_frame_sub.query('is_3==1')['name'].nunique() < 1:
            continue
        drop_list.append(enemy_name)  # 存在可见的敌人，直接进行过滤
    drop_enemy = [i for i in shoot_no_hit_list if i[0] not in drop_list]
    return drop_enemy


def judge_no_hit(frame, shoot_list, hit_enemy_name):
    new_frame = frame.copy()
    if 'aim_info' not in new_frame.columns:
        new_frame['use_vzoom_dis'] = new_frame['dis_3d']
    else:
        new_frame['use_vzoom'] = (new_frame['is_aim'] * new_frame['aim_info']).map(lambda x: 1 if x == 0 else x)  # 倍镜信息加入
        new_frame['use_vzoom_dis'] = new_frame['dis_3d']/new_frame['use_vzoom']
    new_frame = new_frame.sort_values(['f_index', 'screen_dis'])
    min_screen = new_frame.query('screen_dis<0.03')['name'].unique().tolist()  # 将不满足条件的 敌人 剔除掉

    if len(new_frame) < 1:
        return []
    group_frame = new_frame.groupby('name')['is_3'].value_counts().unstack().fillna(0)
    if 1 not in group_frame.columns:
        group_frame[1] = 0

    all_time_invisibe = group_frame.loc[group_frame[1] == 0, :]
    all_time_invisibe = all_time_invisibe.loc[~all_time_invisibe.index.isin(hit_enemy_name)]  # 排除掉有命中，但无可见的情况
    all_time_invisibe = all_time_invisibe.loc[all_time_invisibe.index.isin(min_screen)]  # 排除掉 从来没有瞄准过的敌人

    cheat_enemy_list = []
    for index_ in shoot_list:
        frame_sub_one = new_frame.query('f_index <@index_')
        if len(frame_sub_one) < 1:
            continue
        for enemy_name in all_time_invisibe.index.tolist():
            frame_sub = frame_sub_one.query('name== @enemy_name').copy()
            if (len(frame_sub) < 1) or (frame_sub['screen_dis'].min() > 0.08):
                continue
            dis_3d = frame_sub['dis_3d'].values[-1]
            f_count = int(dis_3d / 600 / 0.0355)
            file_shoot_f = index_ - f_count
            f_start, f_end = file_shoot_f - 5, file_shoot_f + 5
            if file_shoot_f < 0:
                continue
            screen_dis = frame_sub.query('f_index>=@f_start and f_index<=@f_end and use_vzoom_dis<400')['screen_dis']
            if len(screen_dis) < 3:
                continue
            describe_array = screen_dis.describe()[['mean', 'std', 'min']].to_list()
            if describe_array[0] < 0.01 and describe_array[2] < 0.007:
                cheat_enemy_list.append([enemy_name, f_start, f_end, *describe_array])
                break
        if len(cheat_enemy_list) > 0:
            break

    if len(cheat_enemy_list) > 0:
        cheat_enemy_list = check_list_again(new_frame, cheat_enemy_list)

    return cheat_enemy_list


def get_enemy_time_data(frame):
    aim_sub_sub = []
    if len(frame[0]) == 5:
        for ii in frame:
            if ii[2] == 1:
                aim_sub = [ii[0] for j in ii[4] if len(j[3:]) < 1]  # 4位置是敌人list
                if len(aim_sub) == len(ii[4]):
                    aim_sub_sub.append(1)
                else:
                    aim_sub_sub.append(0)
            else:
                aim_sub_sub.append(0)
    else:
        for ii in frame:
            if ii[2] == 1:
                aim_sub = [ii[0] for j in ii[3] if len(j[3:]) < 1]  # 3位置是敌人list
                if len(aim_sub) == len(ii[3]):  # 3位置是敌人list
                    aim_sub_sub.append(1)
                else:
                    aim_sub_sub.append(0)
            else:
                aim_sub_sub.append(0)

    aim_op_no = sum(aim_sub_sub)
    return aim_op_no


def get_invisible_time(frame, strict=False):
    import itertools
    invisibel_threshold = 60 if strict else 30
    dis_change_threshold = 5 if strict else 3

    new_frame = frame.copy()

    visible_frame = new_frame.copy().sort_values('f_index')

    new_frame = new_frame.sort_values(['f_index', 'screen_dis']).drop_duplicates('f_index',
                                                                                 keep='first')  # 一帧中找距离最小的一个当代表
    new_frame = new_frame.query('is_3==0 and screen_dis <2 ').copy().sort_values('f_index')

    open_vis = []
    open_dis = []
    open_dis_mean = []

    seq_frame = new_frame['f_index'].unique()
    red_fr = seq_frame[1:] - seq_frame[:-1]
    for_fr = np.split(seq_frame, np.where(red_fr > 1)[0] + 1)  # 超过 2 帧算第二次瞄准

    invisible_enemy = {}
    for list_fr in for_fr:
        list_fr = list_fr.tolist()
        if len(list_fr) <= 60:  # 对小于 60帧的数据进行丢弃
            continue

        frame_sub = new_frame.query('screen_dis<=0.2 and f_index in@list_fr').copy()

        if len(frame_sub) <= 60:
            continue

        group_sub = frame_sub.groupby('name')

        f_index = group_sub['f_index'].count().sort_values()

        first_enemy_name_for = f_index.loc[f_index >= invisibel_threshold ].index.tolist()

        all_sel_name = frame_sub.query('screen_dis <=0.03').groupby('name')['screen_dis'].count()

        enemy_name_for = all_sel_name[all_sel_name >= 3].index.tolist()
        enemy_name_for = list(set(enemy_name_for).intersection(first_enemy_name_for))

        for enemy_name in enemy_name_for:
            group_farme = frame_sub.query('name==@enemy_name').copy()

            seq_frame_sel = group_farme['f_index'].unique()
            red_fr_sel = seq_frame_sel[1:] - seq_frame_sel[:-1]
            for_fr_sel = np.split(seq_frame_sel, np.where(red_fr_sel > 3)[0] + 1)  # 间隔超过 3 帧算作不连续

            for for_f_index in for_fr_sel:
                for_f_index = for_f_index.tolist()
                if len(for_f_index) < invisibel_threshold:
                    continue
                else:
                    between_dis = group_farme.query('f_index in @for_f_index')[['dis_3d', 'is_aim']].values
                    min_dis = between_dis[:, 0].min()
                    if min_dis >= 160 and len(between_dis[between_dis[:, 1] == 1]) <= 10:
                        continue
                    if enemy_name not in invisible_enemy.keys():
                        invisible_enemy[enemy_name] = []
                        invisible_enemy[enemy_name].append(for_f_index)
                    else:
                        invisible_enemy[enemy_name].append(for_f_index)

    true_for_fr = {}
    for enemy_name, value in invisible_enemy.items():  # 剔除掉 移动距离不超过 3m 的 帧
        for list_fr in value:

            min_f = min(list_fr)
            visible_sub = visible_frame.query('f_index<@min_f and is_3 ==1 and name==@enemy_name').copy()
            less_3 = len(
                new_frame.query('f_index in @ list_fr and name==@enemy_name and screen_dis<=0.03'))  # 小于 0.03的帧数

            if len(visible_sub) == 0:

                enemy_pos_start = np.stack(visible_frame.query('f_index<=@min_f and name==@enemy_name')
                                           ['position_enemy'].values)[0]
                enemy_pos_end_all = np.stack(new_frame.query('f_index in @list_fr and name==@enemy_name')
                                             ['position_enemy'].values)[0]
                dis_change = np.linalg.norm(enemy_pos_end_all - enemy_pos_start)
                if dis_change >= 5:
                    if enemy_name not in true_for_fr.keys():
                        true_for_fr[enemy_name] = []
                        true_for_fr[enemy_name].append([list_fr, less_3])
                    else:
                        true_for_fr[enemy_name].append([list_fr, less_3])

            elif len(visible_sub) <= 2:
                enemy_pos_start = np.stack(visible_frame.query('f_index<@min_f and is_3 ==1 and name==@enemy_name')
                                           ['position_enemy'].values)[-1]
                enemy_pos_end_all = np.stack(new_frame.query('f_index in @list_fr and name==@enemy_name')
                                             ['position_enemy'].values)[-1]
                dis_change = np.linalg.norm(enemy_pos_end_all - enemy_pos_start)
                if dis_change >= 5:
                    if enemy_name not in true_for_fr.keys():
                        true_for_fr[enemy_name] = []
                        true_for_fr[enemy_name].append([list_fr, less_3])
                    else:
                        true_for_fr[enemy_name].append([list_fr, less_3])
            else:
                seq_frame_enemy = visible_sub['f_index'].unique()
                red_fr_enemy = seq_frame_enemy[1:] - seq_frame_enemy[:-1]
                for_fr_enemy = np.split(seq_frame_enemy, np.where(red_fr_enemy > 3)[0] + 1)  # 找到联系的可见3帧作为看见的依据
                index_for_fr = [enemy_for for enemy_for in for_fr_enemy if len(enemy_for) > 0][-1]  # 取最后一次可见作为依据

                if len(index_for_fr) < 1:
                    enemy_pos_start = np.stack(visible_frame.query('f_index<@min_f and is_3 ==1 and name==@enemy_name')
                                               ['position_enemy'].values)[-1]
                    enemy_pos_end_all = np.stack(new_frame.query('f_index in @list_fr and name==@enemy_name')
                                                 ['position_enemy'].values)[-1]
                    dis_change = np.linalg.norm(enemy_pos_end_all - enemy_pos_start)
                    if dis_change >= 5:
                        if enemy_name not in true_for_fr.keys():
                            true_for_fr[enemy_name] = []
                            true_for_fr[enemy_name].append([list_fr, less_3])
                        else:
                            true_for_fr[enemy_name].append([list_fr, less_3])
                else:
                    pos_data = visible_sub.query('f_index in @index_for_fr')
                    in_list_pos = new_frame.query('f_index in @list_fr and name==@enemy_name')

                    enemy_pos_end_array = np.stack(in_list_pos['position_enemy'].values)  # 获取敌人被透视时的位置
                    self_pos_end_array = np.stack(in_list_pos['position'].values)  # 玩家自己的位置

                    enemy_pos_start = np.stack(pos_data['position_enemy'].values)[-1]

                    start_array = enemy_pos_end_array - enemy_pos_start

                    self_enemy_array = self_pos_end_array - enemy_pos_end_array

                    len_array = np.linalg.norm(self_enemy_array, axis=-1)

                    dis_change_all = np.linalg.norm((enemy_pos_end_array - enemy_pos_start), axis=-1)

                    if dis_change_all.mean() >= dis_change_threshold:
                        dis_change_all[dis_change_all == 0] = 1
                        cos_rate = np.abs(np.sum(self_enemy_array * start_array, axis=-1) / len_array) / dis_change_all
                        cos_rate[np.isnan(cos_rate)] = 0
                        if strict and cos_rate.mean() < 0.90:
                            if enemy_name not in true_for_fr.keys():
                                true_for_fr[enemy_name] = []
                                true_for_fr[enemy_name].append([list_fr, less_3])
                            else:
                                true_for_fr[enemy_name].append([list_fr, less_3])
                        else:
                            if cos_rate.mean() < 0.90 or dis_change_all.mean() / len(list_fr) >= 0.08:
                                if enemy_name not in true_for_fr.keys():
                                    true_for_fr[enemy_name] = []
                                    true_for_fr[enemy_name].append([list_fr, less_3])
                                else:
                                    true_for_fr[enemy_name].append([list_fr, less_3])

    realy_sel = {}
    del_invisible_list = []
    if len(true_for_fr.keys()) <= 1:  # 只有一个敌人
        for key, value in true_for_fr.items():
            realy_sel[key] = []
            for value_value in value:
                realy_sel[key].append(value_value[0])
    else:
        for two_name in list(itertools.combinations(list(true_for_fr.keys()), 2)):  # 去重同时间段的 数据 存在多个敌人
            name_1, name_2 = two_name[0], two_name[1]
            for value_1 in true_for_fr[name_1]:
                for value_2 in true_for_fr[name_2]:
                    union_list = list(set(value_1[0]).union(set(value_2[0])))  # 并集
                    if len(set(value_1[0]).intersection(value_2[0])) / len(union_list) > 0.6:
                        if value_1[1] > value_2[1]:
                            del_invisible_list.append(value_2[0])
                        else:
                            del_invisible_list.append(value_1[0])
        for key, value in true_for_fr.items():
            realy_sel[key] = []
            for value_sub in value:
                if value_sub[0] not in del_invisible_list:
                    realy_sel[key].append(value_sub[0])

    screen_std = []
    for enemy_name, value in realy_sel.items():
        for for_f_index in value:
            use_frame_sub = new_frame.query('f_index in @for_f_index and name==@enemy_name').copy()

            open_dis.append(use_frame_sub['dis_3d'].std())
            screen_std.append(use_frame_sub['screen_dis'].std())

            open_dis_mean.extend(use_frame_sub['dis_3d'].to_list())

            open_vis.append(use_frame_sub['f_index'].nunique())

    if len(open_vis) == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0]  # count,mean,min,max, screen_std, open_dis_mean(min,max), open_dis(mean)
    open_vis = pd.DataFrame(open_vis)[0].describe()
    open_vis = open_vis.fillna(0)

    open_dis_mean = pd.DataFrame(open_dis_mean)[0].describe()
    open_dis_mean = open_dis_mean.fillna(0)

    return [*open_vis.values[[0, 1, 3, 7]].tolist(), np.mean(screen_std), *open_dis_mean.values[[3, 7]],
            np.mean(open_dis)]  # count mean, min,max  [0, 1, 4, 5, 7]


def get_shoot_speed(frame, enemy_shoot, strict=False):
    new_frame = frame.copy()

    if len(enemy_shoot.keys()) == 0:
        return 'no_enemy', [len(new_frame['f_index'].unique()), new_frame['dis_3d'].max()]  # 没有开枪数据

    select_data = new_frame.query('screen_dis<=0.1').groupby('name')['f_index'].nunique()

    select_name_all = list(set(select_data.index.tolist()).intersection(list(enemy_shoot.keys())))

    if len(select_name_all) <= 0:
        return 'no_enemy', [len(new_frame['f_index'].unique()), new_frame['dis_3d'].max()]
    else:
        select_name = select_data.loc[select_name_all].sort_values().index[-1]

    for ro_id in [select_name]:  # 平滑数据
        value = enemy_shoot[ro_id]
        max_value = max(value) + 20

        sub_sub_frame = new_frame.query('name==@ro_id and f_index<=@max_value').copy()

        if len(sub_sub_frame) <= 0:
            continue

        is_see = sub_sub_frame['is_3'].values
        is_see[max_value - 20:] = 1  # 开枪的以及开枪后的数据都是可见
        origon_is_see = is_see

        see_sub_list = is_see[:10].tolist()  # 平滑算法

        if strict:
            see_sub_list.extend([1 if sum(is_see[i - 10:i + 11]) >= 11 or is_see[i] == 1 else 0 for i in
                                 range(10, len(is_see) - 10)])  # 将可见扩大
            if len(is_see) > 10:
                see_sub_list.extend(is_see[-10:])
            is_see = np.array(see_sub_list)[:len(sub_sub_frame)]

            while sum(np.array(see_sub_list)[:len(sub_sub_frame)] - origon_is_see) != 0:
                origon_is_see = np.array(see_sub_list)[:len(sub_sub_frame)]
                see_sub_list = is_see[:10].tolist()  # 平滑算法
                see_sub_list.extend([1 if sum(is_see[i - 10:i + 11]) >= 11 or is_see[i] == 1 else 0 for i in
                                     range(10, len(is_see) - 10)])
                if len(is_see) > 10:
                    see_sub_list.extend(is_see[-10:])
                is_see = np.array(see_sub_list)
            is_see = np.array(see_sub_list)[:len(sub_sub_frame)]
        else:
            see_sub_list.extend([1 if sum(is_see[i - 10:i + 11]) >= 11 else 0 for i in range(10, len(is_see) - 10)])
            if len(is_see) > 10:
                see_sub_list.extend(is_see[-10:])
            is_see = np.array(see_sub_list)

        if len(is_see) <= 0:
            continue
        new_frame.loc[(new_frame['name'] == ro_id) & (new_frame['f_index'] <= max_value), 'is_3'] = is_see[
                                                                                                    :len(sub_sub_frame)]

    for key in [select_name]:
        i = enemy_shoot[key][0]
        new_sub = new_frame.query('f_index <@i and name==@key and is_3==0').copy().sort_values('f_index')
        shoot_3d_dis = -1

        if len(new_sub) > 0:
            shoot_3d_dis = new_sub['dis_3d'].values[-1]
            sel_f = new_sub.iloc[-1, 0]
            sub_list = [key, i - sel_f]

        elif sum(new_frame.query('f_index <@i and name==@key and is_3 in [0,1]')['is_3'] == 1) == 0:
            sub_list = [key, 0]  # 如果全部都是不可见，直接用 0 帧代表

        else:
            sub_list = [key, i]  # 如果全部都是可见，直接用 i 帧代表

        if shoot_3d_dis == -1:
            sel_shoot_f = set(enemy_shoot[key]).intersection(new_frame.query('name==@key')['f_index'].unique())

            if len(sel_shoot_f) == 0:
                shoot_3d_dis = new_frame.query('name==@key')['dis_3d'].values[0]
            else:
                min_shoot_f = min(sel_shoot_f)
                shoot_3d_dis = new_frame.query('f_index ==@min_shoot_f and name==@key')['dis_3d'].values[0]

    return sub_list[0], [sub_list[1], shoot_3d_dis]


def get_model_array_900017(json_frame, new_frame, enemy_shoot):
    aim_op_no = get_enemy_time_data(json_frame)

    enemy_name, f_shoot = get_shoot_speed(new_frame, enemy_shoot, strict=True)
    invisible_data = get_invisible_time(new_frame, strict=True)

    embding_arr = [aim_op_no, *f_shoot, *(np.array(invisible_data)[[0, 1, 4, 5, 7]].tolist())]

    embding_arr = np.array(embding_arr, dtype=np.float32)

    return embding_arr.reshape(1, -1)


def get_rule_array_900020(new_frame, enemy_shoot, shoot_no_hit_list):
    all_hit_enemy_names = list(enemy_shoot.keys())
    out_two_shoot_index = []
    if len(shoot_no_hit_list) > 20:  # 去除掉对命中人的开枪数据
        all_not_hit = np.array(shoot_no_hit_list)
        red_shoot_index = all_not_hit[1:] - all_not_hit[:-1]
        for_fr = np.split(all_not_hit, np.where(red_shoot_index > 20)[0] + 1)
        for seq_shoot in for_fr:
            seq_shoot = seq_shoot.tolist()
            if len(new_frame.query(
                    'f_index in @seq_shoot and screen_dis <0.05 and name not in @all_hit_enemy_names')) <= 0:
                out_two_shoot_index.extend(seq_shoot)
    shoot_no_hit_list = list(set(shoot_no_hit_list).difference(out_two_shoot_index))
    shoot_no_hit_list.sort()
    shoot_no_hit_list = [i for i in shoot_no_hit_list if i > 100]  # 过滤前100帧的开枪
    embding_rule = judge_no_hit(new_frame, shoot_no_hit_list, all_hit_enemy_names)

    return embding_rule


def load_model(file) -> Any:
    """导入模型

    Args:
        file (str): 模型路径, 生产环境从s3加载

    Returns:
        Any: 模型对象
    """
    model_all = joblib.load(file)
    return model_all


def preprocess(info) -> Any:
    """数据预处理, 转换为模型输入特征

    Args:
        info (Any): 输入原始数据

    Returns:
        Any: 模型输入特征
    """
    json_frame = json.loads(info['frame_data'])  # 加载 txt 字符文件
    logic_data = json.loads(info['logic_data'])
    role_id = json.loads(info['relation_data'])['roleid']

    # json_frame = info['frame_data']  # 加载 json 文件
    # logic_data = info['logic_data']
    # role_id = info['relation_data']['roleid']

    new_frame, enemy_shoot, shoot_no_hit_list = explode_frame_data(json_frame, logic_data['shoot_data'], role_id)

    if len(new_frame) == 0:
        embding_arr_900017 = np.zeros((1, 8))
    else:
        embding_arr_900017 = get_model_array_900017(json_frame, new_frame, enemy_shoot)

    if len(shoot_no_hit_list) == 0 or len(new_frame) == 0:
        embding_rule_900020 = []
    else:
        embding_rule_900020 = get_rule_array_900020(new_frame, enemy_shoot, shoot_no_hit_list)

    return [embding_arr_900017, embding_rule_900020]


def predict(common_feature, model_all) -> Dict[str, str]:
    """模型预测

    Args:
        feature (Any): 模型输入特征
        model (Any): 模型对象


    Returns:
        dict: 有命中就返回，key:策略代号，value:开挂概率
    """
    out_dict = {}
    feature_900017,  rule_feature_900020 = common_feature
    model_all_result_900017 = np.mean([model.predict(feature_900017, num_iteration=model.best_iteration) for model in model_all])

    if model_all_result_900017 > 0.5:
        out_dict['900017'] = model_all_result_900017
        if model_all_result_900017 >= 0.81:
            out_dict['900018'] = model_all_result_900017
    if len(rule_feature_900020) > 0:
        out_dict['900020'] = 0.85


    return out_dict
