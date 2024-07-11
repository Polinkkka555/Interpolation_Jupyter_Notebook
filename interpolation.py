import subprocess
import numpy as np
import tabulate as tab
import matplotlib.pyplot as plt

# import interpolation_exe


# Вывод матрицы на экран
def print_arr(string, namevec, a):
    if (type(a) == int) or (type(a) == float):
        print(a)
    else:
        print(string)
        for i in range(len(a)):
            print("{}[{}] = {:8.4f}".format(namevec, i, a[i]))


def cubic_spline_1(x, y, t, gamma, path):
    print("_______Интерполяция кубическими сплайнами (1 вариант)_______\n")
    with open("data.txt", "w") as file:
        file.write(str("cubic_spline_1") + '\n')
        file.write(str(len(x)) + '\n')
        for i in range(len(x)):
            file.write(str(x[i]) + ' ' + str(y[i]) + '\n')
        file.write(str(len(t)) + '\n')
        for i in range(len(t)):
            file.write(str(t[i]) + ' ')
        file.write('\n')
        file.write(str(gamma[0]) + ' ' + str(gamma[1]))

    subprocess.run([path])
    #interpolation_exe.main()

    with open("answer.txt", "r") as file:
        count = 0
        for line in file:
            count += 1
            if count == 1:
                p = list(map(float, line.split()))
            elif count == 2:
                q = list(map(float, line.split()))
            elif count == 3:
                A = list(map(float, line.split()))
            elif count == 4:
                B = list(map(float, line.split()))
            elif count == 5:
                C = list(map(float, line.split()))
            elif count == 6:
                D = list(map(float, line.split()))
            elif count == 7:
                t = list(map(float, line.split()))
            elif count == 8:
                S = list(map(float, line.split()))

    n = len(x) - 1
    index = [i for i in range(0, n + 1)]
    table = {"Прогоночные\nкоэффициенты": index,
             "p": p,
             "q": q
             }
    print(tab.tabulate(table, headers="keys", tablefmt='psql', floatfmt=".4f", stralign="center"))

    print_arr('Решение СЛАУ методом прогонки: ', 'c', C)

    k = 100
    x1 = np.linspace(x[0], x[n], k)
    y1 = [0 for i in range(0, k)]

    for j in range(0, k):
        for i in range(1, n + 1):
            if (x[i - 1] <= x1[j]) and (x1[j] <= x[i]):
                y1[j] = A[i] + B[i] * (x1[j] - x[i]) + C[i] * pow((x1[j] - x[i]), 2) + D[i] * pow((x1[j] - x[i]), 3)
    del A[0]
    del B[0]
    del C[0]
    del D[0]
    index = [i for i in range(1, n + 1)]
    table = {"Коэффициенты\nсплайна": index,
             "a": A,
             "b": B,
             "c": C,
             "d": D
             }
    print(tab.tabulate(table, headers="keys", tablefmt='psql', floatfmt=".4f", stralign="center"))

    print("Значения сплайна в заданных точках:")
    for tt, SS in zip(t, S):
        print("S({}) = {}".format(tt, SS))

    plt.title('Интерполяция кубическими сплайнами (1 вариант)')  # заголовок
    plt.xlabel('x')  # ось абсцисс
    plt.ylabel('y')  # ось ординат
    plt.grid()
    plt.plot(x1, y1, color='#e484f5', label='Сплайн')
    plt.plot(t, S, 'o', color='#0702f7', label='Найденные точки')
    plt.plot(x, y, 'o', color='#96248b', label='Экспериментальные точки')
    plt.legend(fontsize=8, bbox_to_anchor=(1, 1))
    plt.show()


def cubic_spline_2(x, y, t, gamma, betta, alpha, path):
    print("_______Интерполяция кубическими сплайнами (2 вариант)_______\n")
    with open("data.txt", "w") as file:
        file.write(str("cubic_spline_2") + '\n')
        file.write(str(len(x)) + '\n')
        for i in range(len(x)):
            file.write(str(x[i]) + ' ' + str(y[i]) + '\n')
        file.write(str(len(t)) + '\n')
        for i in range(len(t)):
            file.write(str(t[i]) + ' ')
        file.write('\n')
        file.write(str(gamma[0]) + ' ' + str(gamma[1]) + '\n')
        file.write(str(betta[0]) + ' ' + str(betta[1]) + '\n')
        file.write(str(alpha[0]) + ' ' + str(alpha[1]))

    subprocess.run([path])
    #interpolation_exe.main()

    with open("answer.txt", "r") as file:
        count = 0
        for line in file:
            count += 1
            if count == 1:
                p = list(map(float, line.split()))
            elif count == 2:
                q = list(map(float, line.split()))
            elif count == 3:
                m = list(map(float, line.split()))
            elif count == 4:
                t = list(map(float, line.split()))
            elif count == 5:
                S = list(map(float, line.split()))
    n = len(x) - 1
    index = [i for i in range(0, n + 1)]
    table = {"Прогоночные\nкоэффициенты": index,
             "p": p,
             "q": q
             }
    print(tab.tabulate(table, headers="keys", tablefmt='psql', floatfmt=".4f", stralign="center"))

    k = 100
    x2 = np.linspace(x[0], x[n], k)
    y2 = [0 for i in range(0, k)]

    h = [0 for i in range(0, n + 1)]
    for i in range(1, n + 1):
        h[i] = abs(x[i] - x[i - 1])

    for j in range(0, k):
        for i in range(1, n + 1):
            if (x[i - 1] <= x2[j]) and (x2[j] <= x[i]):
                y2[j] = m[i - 1] * pow((x[i] - x2[j]), 3) / (6 * h[i]) + m[i] * pow((x2[j] - x[i - 1]), 3) / (
                        6 * h[i]) + \
                        + (y[i - 1] - m[i - 1] * h[i] * h[i] / 6) * (x[i] - x2[j]) / h[i] + (
                                y[i] - m[i] * h[i] * h[i] / 6) * (x2[j] - x[i - 1]) / h[i]

    print_arr('Коэффициенты сплайна (решение СЛАУ): ', 'm', m)
    print()
    print("Значения сплайна в заданных точках:")
    for tt, SS in zip(t, S):
        print("S({}) = {}".format(tt, SS))

    plt.title('Интерполяция кубическими сплайнами (2 вариант)')  # заголовок
    plt.xlabel('x')  # ось абсцисс
    plt.ylabel('y')  # ось ординат
    plt.grid()
    plt.plot(x2, y2, color='#e484f5', label='Сплайн')
    plt.plot(t, S, 'o', color='#0702f7', label='Найденные точки')
    plt.plot(x, y, 'o', color='#96248b', label='Экспериментальные точки')
    plt.legend(fontsize=8, bbox_to_anchor=(1, 1))
    plt.show()


def bilinear_interpolation(x, y, z, tx, ty, path):
    print("_______________Билинейная интерполяция_______________\n")
    with open("data.txt", "w") as file:
        file.write(str("bilinear_interpolation") + '\n')
        file.write(str(len(x)) + '\n')
        for i in range(len(x)):
            file.write(str(x[i]) + ' ')
        file.write('\n')
        file.write(str(len(y)) + '\n')
        for i in range(len(y)):
            file.write(str(y[i]) + ' ')
        file.write('\n')
        for j in range(len(y)):
            for i in range(len(x)):
                file.write(str(z[j][i]) + ' ')
            file.write('\n')
        file.write(str(len(tx)) + '\n')
        for i in range(len(tx)):
            file.write(str(tx[i]) + ' ')
        file.write('\n')
        for i in range(len(ty)):
            file.write(str(ty[i]) + ' ')

    subprocess.run([path])
    # interpolation_exe.main()

    with open("answer.txt", "r") as file:
        count = 0
        b = []
        answer_str = []
        tx = []
        ty = []
        tz = []
        for line in file:
            count += 1
            if (1 <= count <= (len(x) - 1) * (len(y) - 1)):
                b.append(list(map(float, line.split())))
            elif (count == 1 + (len(x) - 1) * (len(y) - 1)):
                tx = list(map(float, line.split()))
            elif (count == 2 + (len(x) - 1) * (len(y) - 1)):
                ty = list(map(float, line.split()))
            elif (count == 3 + (len(x) - 1) * (len(y) - 1)):
                tz = list(map(float, line.split()))
            else:
                answer_str.append(line.strip())

    answer_tab = [["№", "Сегмент", "b1", "b2", "b3", "b4"]]
    for j in range(len(y) - 1):
        for i in range(len(x) - 1):
            buf = [len(answer_tab)]
            buf.append(f'[{x[i]};{x[i + 1]}]×[{y[j]};{y[j + 1]}]')
            ind = j * (len(x) - 1) + i
            buf.append(b[ind][0])
            buf.append(b[ind][1])
            buf.append(b[ind][2])
            buf.append(b[ind][3])
            answer_tab.append(buf)
    print("Коэффициенты интерполирующей функции F(x,y):")
    print(tab.tabulate(answer_tab, tablefmt="pretty", headers="firstrow", stralign="left"))

    print()
    for i in range(len(answer_str)):
        print(answer_str[i].replace(']*[', ']×['))
        if (i % 3 == 2):
            print()

    xx = np.arange(x[0], x[len(x) - 1], (x[1] - x[0]) / 10)
    yy = np.arange(y[0], y[len(y) - 1], (y[1] - y[0]) / 10)
    zz = []
    for jj in range(len(yy)):
        buf = []
        for ii in range(len(xx)):
            # print(xx[ii], "+", yy[jj])
            for j in range(len(y) - 1):
                if (y[j] <= yy[jj]) and (yy[jj] <= y[j + 1]):
                    for i in range(len(x) - 1):
                        if (x[i] <= xx[ii]) and (xx[ii] <= x[i + 1]):
                            # print(xx[ii], "+", yy[jj],"   - ", j, ", ", i)
                            ind = j * (len(x) - 1) + i
                            buf.append(
                                b[ind][0] + b[ind][1] * xx[ii] + b[ind][2] * yy[jj] + b[ind][3] * xx[ii] * yy[jj])
                            break
                    break
        zz.append(buf)
        # print(zz[jj])

    xx, yy = np.meshgrid(xx, yy)
    zz = np.array(zz)
    # print(xx)
    # print(zz)

    x_dot, y_dot = np.meshgrid(x, y)
    z_dot = []
    for j in range(len(y)):
        z_dot.append([])
        for i in range(len(x)):
            z_dot[j].append(z[y.index(y_dot[j][i])][x.index(x_dot[j][i])])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.plot_surface(xx, yy, zz, cmap=plt.cm.YlGnBu_r, alpha=0.9)
    ax.scatter3D(x_dot, y_dot, z_dot, s=5, c='blue', alpha=1, label='Экспериментальные точки')
    ax.scatter3D(tx, ty, tz, c='black', s=20, alpha=1, label='Найденные точки')
    plt.legend(loc='upper left')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.azim = -160
    ax.elev = 30
    plt.show()
    plt.savefig("mygraph.png")


def bicubic_interpolation(x, y, z, tx, ty, path):
    print("_______________Бикубическая интерполяция_______________\n")
    with open("data.txt", "w") as file:
        file.write(str("bicubic_interpolation") + '\n')
        file.write(str(len(x)) + '\n')
        for i in range(len(x)):
            file.write(str(x[i]) + ' ')
        file.write('\n')
        file.write(str(len(y)) + '\n')
        for i in range(len(y)):
            file.write(str(y[i]) + ' ')
        file.write('\n')
        for j in range(len(y)):
            for i in range(len(x)):
                file.write(str(z[j][i]) + ' ')
            file.write('\n')
        file.write(str(len(tx)) + '\n')
        for i in range(len(tx)):
            file.write(str(tx[i]) + ' ')
        file.write('\n')
        for i in range(len(ty)):
            file.write(str(ty[i]) + ' ')

    subprocess.run([path])
    #interpolation_exe.main()

    answer_str = []
    txx = []
    tyy = []
    tzz = []
    with open("answer.txt", "r") as file:
        count = 0
        b = []
        for line in file:
            count += 1
            if (1 <= count <= (len(x) - 3) * (len(y) - 3)):
                b.append(list(map(float, line.split())))
            elif (count == 1 + (len(x) - 3) * (len(y) - 3)):
                txx = list(map(float, line.split()))
            elif (count == 2 + (len(x) - 3) * (len(y) - 3)):
                tyy = list(map(float, line.split()))
            elif (count == 3 + (len(x) - 3) * (len(y) - 3)):
                tzz = list(map(float, line.split()))
            else:
                answer_str.append(line.strip())

    for i in range(len(answer_str)):
        if (i%6<3):
            print(answer_str[i].replace(']*[', ']×['))
        else:
            print("       " + answer_str[i].replace(']*[', ']×['))
        if (i % 6 == 5):
            print()

    xx = np.arange(x[1], x[len(x) - 2] + 0.5, 0.5)
    yy = np.arange(y[1], y[len(y) - 2] + 0.5, 0.5)
    zz = []

    # print("xx:", xx)
    # print("yy:", yy)

    for jj in range(len(yy)):
        buf2 = []
        for ii in range(len(xx)):
            # print(xx[ii], ",", yy[jj])
            for j in range(1, len(y) - 1):
                if (y[j] <= yy[jj]) and (yy[jj] <= y[j + 1]):
                    for i in range(1, len(x) - 1):
                        if (x[i] <= xx[ii]) and (xx[ii] <= x[i + 1]):
                            buf1 = 0
                            ind = (j - 1) * (len(x) - 3) + i - 1
                            #print(ind)
                            for jjj in range(0, 4):
                                for iii in range(0, 4):
                                    buf1 += b[ind][jjj * 4 + iii] * (pow(xx[ii], iii) * pow(yy[jj], jjj))
                            buf2.append(buf1)
                            break
                    break
        # print(buf2)
        zz.append(buf2)

    xx, yy = np.meshgrid(xx, yy)
    zz = np.array(zz)

    x_dot, y_dot = np.meshgrid(x, y)
    z_dot = []
    for j in range(len(y)):
        z_dot.append([])
        for i in range(len(x)):
            z_dot[j].append(z[y.index(y_dot[j][i])][x.index(x_dot[j][i])])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.plot_surface(xx, yy, zz, cmap=plt.cm.YlGnBu_r, linewidth=10, antialiased=False, alpha=0.9)
    ax.scatter3D(x_dot, y_dot, z_dot, c='blue', alpha=1, s=5, label='Экспериментальные точки')
    ax.scatter3D(txx, tyy, tzz, c='black', s=20, alpha=1, label='Найденные точки')
    plt.legend(loc='upper left')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.azim = -160
    ax.elev = 30
    plt.show()
#_____________________________________________________________________________________________________________
'''
path = "./interpolation_exe"
var = 3

if var == 1:
    x = [-3, 0, 1, 2, 4]
    y = [-27, 0, 1, 8, 64]
    gamma = [-18, 24]
    t = [-1, -2, 3.5]

    cubic_spline_1(x, y, t, gamma, path)

# var = 2
if var == 2:
    x = [-3, 0, 1, 2, 4]
    y = [-27, 0, 1, 8, 64]
    alpha = [1, 2]
    betta = [0, 0.5]
    gamma = [27, 108]
    t = [-1, -2, 3.5]

    cubic_spline_2(x, y, t, gamma, betta, alpha, path)

if var == 3:
    x = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16]
    y = [-11, -9, -7, -5, -3, -1, 1, 3, 5, 7]
    z = []
    tx = [10.5, 5.5]
    ty = [4.5, 3.5]

    for j in range(0, len(y)):
        z.append([])
        for i in range(0, len(x)):
            z[j].append(x[i] * x[i] - y[j] * y[j])
    bilinear_interpolation(x, y, z, tx, ty, path)

if var == 4:
    x = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16]
    y = [-11, -9, -7, -5, -3, -1, 1, 3, 5, 7]
    z = []
    tx = [10.5, 5.5]
    ty = [4.5, 3.5]

    for j in range(0, len(y)):
        z.append([])
        for i in range(0, len(x)):
            z[j].append(x[i] * x[i] - y[j] * y[j])
    bicubic_interpolation(x, y, z, tx, ty, path)
'''