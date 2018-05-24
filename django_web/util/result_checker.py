import re
id_num_reg = r'[X0-9]+'
idcard_reg = r'^[1-9]\d{5}(18|19|([23]\d))\d{2}((0[1-9])|(10|11|12))(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]$'
name_reg = r'^[\u4e00-\u9fa5]{2,6}$'

sex_reg = r'^(\u7537|\u5973){1}$'
chi_sim_reg = r'^[\u4e00-\u9fa5]+$'


def card_number_filter(content):
    """剔除身份证号中所有非法字符"""
    result = re.findall(id_num_reg, content)
    result = "".join(result)
    return result


def check_idcard(idnum):
    """
    check id number through regular expression
    :param idnum:
    :return:
    """
    if idnum is None:
        return False
    if isinstance(idnum, int):
        idnum = str(idnum)
    if not isinstance(idnum, str):
        return False
    match = re.match(idcard_reg, idnum)
    return match is not None


def check_name(name):
    """
    check name through regular expression
    :param name:
    :return:
    """
    if name is None:
        return False
    if not isinstance(name, str):
        return False
    match = re.match(name_reg, name)
    return match is not None


def check_sex(sex):
    """
    check sex through regular expression
    :param sex:
    :return:
    """
    if sex is None:
        return False
    if not isinstance(sex, str):
        return False
    match = re.match(sex_reg, sex)
    return match is not None


def check_chi_sim(chi_str):
    """
    check chinese sentence by regular expression
    :param chi_str:
    :return:
    """
    if chi_str is None:
        return False
    if not isinstance(chi_str, str):
        return False
    match = re.match(chi_sim_reg, chi_str)
    return match is not None


if __name__ == '__main__':
    # start = time.time()
    # for i in range(100*1000):
    #     idnum = int(time.time())
    #     ret = check_idcard(idnum)
    # end = time.time()
    # print("time spend = %d ms"%(int((end-start)*1000)))
    # idnum = 310108199101173817
    # idstr = "310108199101173817"
    # ret = check_idcard(idstr)
    # print(ret)

    # name = "张三丰__"
    # print(check_name(name))

    sex = "女"
    print(check_sex(sex))
