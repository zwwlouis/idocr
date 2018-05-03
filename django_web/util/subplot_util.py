from matplotlib import pyplot as plt


def auto_subplt(imgs, col=2, funcs=[plt.imshow], params=[], title=[""]):
    """

    :param imgs: 图像数组
    :param col: 图像阵列的列数，默认为2
    :param funcs: 作图函数
    :param params: 作图参数
    :param title: 子图标题
    :return:
    """
    img_num = len(imgs)
    func_num = len(funcs)
    param_num = len(params)
    title_num = len(title)
    row = int(img_num/col) + 1
    for i in range(row):
        for j in range(col):
            num = i * col + j
            if num < img_num:
                plt.subplot(row, col, num + 1)
                param = params[j % param_num]
                img = imgs[num]
                img_mat = img.get()
                funcs[j % func_num](img_mat, **param)
                plt.title(title[num % title_num])

    plt.show()


def img_auto_subplt(imgs, row, col):
    if len(imgs) != row * col:
        return
    for i in range(row):
        for j in range(col):
            num = i * row + j
            plt.subplot(row * 100 + col * 10 + num + 1)
            plt.imshow(imgs[num])
    plt.show()


if __name__ == '__main__':
    pass
