import multiprocessing


def func(num, i):
    # num[2] = 9999  # 子进程改变数组，主进程跟着改变
    for it in range(i*100000000, (i+1)*100000000):
        num[it] = it


if __name__ == "__main__":
    num = multiprocessing.Array("i", [0]*800000000)  # 主进程与子进程共享这个数组
    print(num[:])

    for i in range(8):
        p = multiprocessing.Process(target=func, args=(num, i))
        p.start()

    

    print(num[:])

