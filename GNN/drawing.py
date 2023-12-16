import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os


def draw_loss():
    data1_loss = np.loadtxt("./logs/GCN/GCN_loss_records.txt")
    # data2_loss = np.loadtxt("./logs/GAT/GAT_loss_records.txt")
    data2_loss = np.loadtxt("./logs/GCN_SGC/GCN_SGC_loss_records.txt")
    #第一列是训练步数，第二列的loss,所以取出相应列的数据作为绘图的x和y
    x1 = data1_loss[:,0]
    y1 = data1_loss[:,1]
    x2 = data2_loss[:,0]
    y2 = data2_loss[:,1]

    #先创建一幅图，再在这幅图上添加一个小图，小图用来显示部分放大的曲线
    fig = plt.figure(figsize = (7,5))       #figsize是图片的大小`
    ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`

    #先画出整体的loss曲线
    # pl.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')
    # ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
    p2 = pl.plot(x1, y1,'r-', label = u'GCN')
    pl.legend()
    #显示图例
    p3 = pl.plot(x2,y2, 'b-', label = u'GAT')
    pl.legend()
    pl.xlabel(u'epochs')
    pl.ylabel(u'loss')
    plt.title('Compare loss for different models in training')

    # #显示放大的部分曲线
    # # plot the box
    # tx0 = 0
    # tx1 = 10000
    # #设置想放大区域的横坐标范围
    # ty0 = 0.000
    # ty1 = 0.12
    # #设置想放大区域的纵坐标范围
    # sx = [tx0,tx1,tx1,tx0,tx0]
    # sy = [ty0,ty0,ty1,ty1,ty0]
    # pl.plot(sx,sy,"purple")
    # axins = inset_axes(ax1, width=1.5, height=1.5, loc='right')
    # #loc是设置小图的放置位置，可以有"lower left,lower right,upper right,upper left,upper #,center,center left,right,center right,lower center,center"
    # axins.plot(x1,y1 , color='red', ls='-')
    # axins.plot(x2,y2 , color='blue', ls='-')
    # axins.axis([0,20000,0.000,0.12])
    if os.path.exists("./logs/results/train_results_loss.png"):
        os.remove("./logs/results/train_results_loss.png")
    plt.savefig("./logs/results/train_results_loss.png")
    pl.show
    #pl.show()也可以

def draw_acc():
    data1_acc = np.loadtxt("./logs/GCN/GCN_acc_records.txt")
    # data2_acc = np.loadtxt("./logs/GAT/GAT_acc_records.txt")
    data2_acc = np.loadtxt("./logs/GCN_SGC/GCN_SGC_acc_records.txt")
    #第一列是训练步数，第二列的loss,所以取出相应列的数据作为绘图的x和y
    x1 = data1_acc[:,0]
    y1 = data1_acc[:,1]
    x2 = data2_acc[:,0]
    y2 = data2_acc[:,1]

    #先创建一幅图，再在这幅图上添加一个小图，小图用来显示部分放大的曲线
    fig = plt.figure(figsize = (7,5))       #figsize是图片的大小`
    ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`

    #先画出整体的loss曲线
    # pl.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')
    # ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
    p2 = pl.plot(x1, y1,'r-', label = u'GCN')
    pl.legend()
    #显示图例
    p3 = pl.plot(x2,y2, 'b-', label = u'GAT')
    pl.legend()
    pl.xlabel(u'epochs')
    pl.ylabel(u'acc')
    plt.title('Compare acc for different models in training')

    # #显示放大的部分曲线
    # # plot the box
    # tx0 = 0
    # tx1 = 10000
    # #设置想放大区域的横坐标范围
    # ty0 = 0.000
    # ty1 = 0.12
    # #设置想放大区域的纵坐标范围
    # sx = [tx0,tx1,tx1,tx0,tx0]
    # sy = [ty0,ty0,ty1,ty1,ty0]
    # pl.plot(sx,sy,"purple")
    # axins = inset_axes(ax1, width=1.5, height=1.5, loc='right')
    # #loc是设置小图的放置位置，可以有"lower left,lower right,upper right,upper left,upper #,center,center left,right,center right,lower center,center"
    # axins.plot(x1,y1 , color='red', ls='-')
    # axins.plot(x2,y2 , color='blue', ls='-')
    # axins.axis([0,20000,0.000,0.12])
    if os.path.exists("./logs/results/train_results_acc.png"):
        os.remove("./logs/results/train_results_acc.png")
    plt.savefig("./logs/results/train_results_acc.png")
    pl.show
    #pl.show()也可以


if __name__ == "__main__":
    draw_loss()
    draw_acc()