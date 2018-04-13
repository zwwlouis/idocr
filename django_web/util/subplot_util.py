from matplotlib import pyplot as plt


def img_subplt(imgs, row, col):
    if len(imgs) != row*col:
        return
    for i in range(row):
        for j in range(col):
            num = i*row+j
            plt.subplot(row*100+col*10+num+1)
            plt.imshow(imgs[num])
    plt.show()


def subplt(imgs, row, col, funcs, params=[],title=[]):
    img_num = len(imgs)
    func_num = len(funcs)
    for i in range(row):
        for j in range(col):
            num = i*col+j
            if num < img_num:
                plt.subplot(row, col, num+1)
                param = {}
                if j < len(params):
                    param = params[j]
                funcs[j%func_num](imgs[num],**param)
                if num < len(title):
                    plt.title(title[num])

    plt.show()