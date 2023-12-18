import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torchvision
import linpdeconfig
from torch.autograd import Variable
import matplotlib.pyplot as plt
from pltutils import *
# import seaborn as sns
import re

# sns.set_theme()

# def plot_real_and_predict_data(real_data, predict_data, example_index, name=''):
#     fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
#     g1 = sns.heatmap(real_data[:, example_index, :].T, cmap="YlGnBu", cbar=True, ax=axes[0])
#     g1.set_ylabel('cell')
#     g1.set_xlabel('time_step')
#     g2 = sns.heatmap(predict_data[:, example_index, :].T, cmap="YlGnBu", cbar=True, ax=axes[1])
#     g2.set_ylabel('cell')
#     g2.set_xlabel('time_step')
#     g3 = sns.heatmap(real_data[:, example_index, :].T - predict_data[:, example_index, :].T
#                      , cmap="YlGnBu", cbar=True, ax=axes[2])
#     g3.set_ylabel('cell')
#     g3.set_xlabel('time_step')
#     plt.show()
#
#     save_dir = 'figures/' + name
#     if not os.path.isdir(save_dir):
#         os.mkdir(save_dir)
#     file_name = '/real_predict_loss_example' + '_' + str(example_index) + '_' + '.pdf'
#     fig.savefig(save_dir + file_name)


# def plot_real_and_predict_data(real_data, predict_data, example_index, name=''):
#     fig, axes = plt.subplots(1, 2, figsize=(50, 16), sharey=True)
#     linewith = 10
#     linewith_frame = 4
#     title_fontsize = 60
#     label_fontsize = 50
#     ticks_fontsize = 50
#     cbar_size = 50
#     g1 = sns.heatmap(np.flip(real_data[:, example_index, :], 0), cmap="YlGnBu", cbar=True, annot_kws={"size":30}, ax=axes[0])
#     g1.set_ylabel('T', fontsize=label_fontsize)
#     g1.set_xlabel('X', fontsize=label_fontsize)
#     g1.set_xticklabels([])
#     g1.set_yticklabels([])
#     # plt.xticks(np.arange(0, 2, step=0.2),list('abcdefghigk'),rotation=45)
#     g1.set_title("exact u(x, t)", fontsize=title_fontsize)
#     cax1 = plt.gcf().axes[-1]
#     cax1.tick_params(labelsize=cbar_size)
#     cax1.spines['top'].set_linewidth(linewith_frame)
#     cax1.spines['bottom'].set_linewidth(linewith_frame)
#     cax1.spines['right'].set_linewidth(linewith_frame)
#     cax1.spines['left'].set_linewidth(linewith_frame)
#
#     g2 = sns.heatmap(np.flip(predict_data[:, example_index, :], 0), cmap="YlGnBu", cbar=True, ax=axes[1])
#     g2.set_ylabel('T', fontsize=label_fontsize)
#     g2.set_xlabel('X', fontsize=label_fontsize)
#     g2.set_xticklabels([])
#     g2.set_yticklabels([])
#     g2.set_title("prediction u(x, t)", fontsize=title_fontsize)
#     cax2 = plt.gcf().axes[-1]
#     cax2.tick_params(labelsize=cbar_size)
#     cax2.spines['top'].set_linewidth(linewith_frame)
#     cax2.spines['bottom'].set_linewidth(linewith_frame)
#     cax2.spines['right'].set_linewidth(linewith_frame)
#     cax2.spines['left'].set_linewidth(linewith_frame)
#
#
#
#     # g3 = sns.heatmap(np.flip(real_data[:, example_index, :], 0) - np.flip(predict_data[:, example_index, :], 0)
#     #                  , cmap="YlGnBu", cbar=True, ax=axes[2])
#     # g3.set_ylabel('T', fontsize=label_fontsize)
#     # g3.set_xlabel('X', fontsize=label_fontsize)
#     # g3.set_xticklabels([])
#     # g3.set_yticklabels([])
#     # g3.set_title("error u(x, t)", fontsize=title_fontsize)
#     # cax = plt.gcf().axes[-1]
#     # cax.tick_params(labelsize=cbar_size)
#
#     plt.show()
#
#     # save_dir = 'figures/' + name
#     # if not os.path.isdir(save_dir):
#     #     os.mkdir(save_dir)
#     # file_name = '/real_predict_data_example' + '_' + str(example_index) + '_all_time' + '.pdf'
#     # fig.savefig(save_dir + file_name)


def plot_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time, N, example_index, name):

    fix_timestep_real_data = real_data[time_steps_real, example_index, :]
    fix_timestep_predict_data = predict_data[time_steps_predict, example_index, :]

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    x_label = []
    dx = 10/N   # 0.05
    for i in range(len(fix_timestep_real_data)):
        x_label.append(i * dx)
    plot_1, = plt.plot(x_label, fix_timestep_real_data, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2, = plt.plot(x_label, fix_timestep_predict_data, label='prediction', color='blue', linestyle='--', linewidth=linewith)
    # plt.title(label='X', fontsize=title_fontsize)
    plt.xlabel('x', fontsize=label_fontsize)
    plt.ylabel('u(x)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_2],
               labels=['exact', 'prediction'],
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.show()

    save_dir = 'figures/' + name + '_' + 'time_' + str(time)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    # file_name = '/' + name + '_' + 'example' + '_' + str(example_index) + '_time_' + str(time) + '.pdf'
    file_name = '/' + name + '_' + str(example_index) + '_' + str(time) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def f_real_function(u):
    f = 0.5 * u * (3 - np.power(u, 2)) + (1/12) * 300 * np.power(u, 2) * ((3/4) - 2*u + (3/2)*np.power(u, 2) - (1/4)*np.power(u, 4))
    return f

def f_predict_function(u):
    # f = (1160.7860940528415)*u**19+(-1146.3283524425976)*u**18+(-964.2497088446811)*u**20+(912.4158921708027)*u**17+(617.4320037674581)*u**21+(-511.57388283635424)*u**16+(-482.672947845133)*u**25+(479.38525832177476)*u**26+(-409.30430839415124)*u**27+(376.77066438228417)*u**24+(-363.9240396385784)*u**13+(312.33560798338596)*u**28+(270.91910284631984)*u**14+(221.98080712057128)*u**12+(-216.64429227199187)*u**22+(-216.3550345450006)*u**29+(-164.97840563383133)*u**10+(160.98527301869387)*u**9+(-137.63677883276713)*u**23+(136.56883988236254)*u**30+(-78.38425496005947)*u**31+(-68.93525304257774)*u**8+(62.769826966319805)*u**15+(40.723107013683254)*u**32+(-26.28640758674795)*u**3+(-19.063653580108816)*u**33+(12.80313841742341)*u**11+(12.082696296737042)*u**6+(11.60547358491062)*u**4+(9.797320306240469)*u**2+(8.012951277858875)*u**34+(-3.017456061611659)*u**35+(-2.8265573137662177)*u**7+(2.6279146139931178)*u+(2.5262534317071)*u**5+(1.0169623946434123)*u**36+(-0.3067049613002629)*u**37+(0.11422365409417473)*1+(0.08280684575828465)*u**38+(-0.020026946076134717)*u**39+(0.004340734550075075)*u**40+(-0.0008429304733424418)*u**41+(0.00014639617449131606)*u**42+(-2.2643736462722105e-05)*u**43+(3.0954240400316426e-06)*u**44
    # f = (2054.6243391580924)*u**21+(-1999.4032119691499)*u**20+(1921.685583455475)*u**28+(-1838.4814322041732)*u**27+(-1787.9413448409118)*u**29+(-1734.3299404994095)*u**22+(1684.7454595612799)*u**19+(1511.8875154815278)*u**30+(1469.8202910101209)*u**26+(-1235.4406003778897)*u**18+(-1165.182312280942)*u**31+(1033.387741486473)*u**23+(815.080605939652)*u**32+(-793.302437871538)*u**25+(746.3018826476373)*u**17+(-514.9142734852825)*u**33+(-308.0186813563284)*u**16+(292.68715105226096)*u**34+(211.26520407400022)*u**11+(-198.88462932713566)*u**10+(-149.41884857982524)*u**35+(121.46739605551905)*u**9+(-117.14963690818259)*u**12+(-108.77013001389952)*u**24+(90.0920225858315)*u**14+(68.48104538759573)*u**36+(-57.11179631769827)*u**8+(-30.65634576571113)*u**3+(29.00198770811694)*u**4+(-28.19257891229919)*u**37+(26.51422314757712)*u**7+(-22.921174162651226)*u**13+(12.048168617232449)*u**15+(10.436412086869774)*u**38+(-10.07242241616574)*u**5+(9.062658508688912)*u**2+(-7.515806347339293)*u**6+(-3.4781970225780863)*u**39+(2.813447598987843)*u+(1.0448912230641352)*u**40+(-0.28324459320000667)*u**41+(-0.10190809095321407)*1+(0.06933737971874253)*u**42+(-0.01533381238019277)*u**43+(0.0030626171150243287)*u**44+(-0.0005517441773596066)*u**45+(8.940183840130248e-05)*u**46+(-1.2962068519992337e-05)*u**47+(1.6674710289362012e-06)*u**48
    # f = (-108.77699868189868)*u**18+(105.45879227452282)*u**19+(87.17151442504938)*u**17+(-82.335958458562)*u**20+(-50.26190916625862)*u**13+(48.39756507013979)*u**21+(47.17253663091515)*u**14+(-45.42935464212146)*u**10+(-43.1679890818231)*u**16+(-36.68578120117861)*u**25+(35.85131621749778)*u**9+(34.279370622733474)*u**26+(31.063433557309494)*u**24+(-27.68791542613079)*u**27+(23.438149711764616)*u**11+(-21.231357721165203)*u**3+(20.036386009585012)*u**28+(18.823935363946518)*u**12+(-14.382259168463394)*u**23+(-13.487593161100541)*u**22+(-13.197097891198377)*u**29+(12.50792584257381)*u**4+(-12.445584914612668)*u**8+(-9.397182425751883)*u**15+(7.951172340006837)*u**30+(6.443791956719194)*u**2+(-4.376172789672932)*u**31+(3.04716176339821)*u+(2.315810898006485)*u**5+(2.190535559897082)*u**32+(-1.4576138702329315)*u**7+(-0.9921556315470744)*u**33+(0.40483116044377543)*u**34+(0.24646909787356228)*1+(-0.1483459913708785)*u**35+(-0.11974508946261586)*u**6+(0.04872884337458996)*u**36+(-0.014336918240490463)*u**37+(0.0037778265315697667)*u**38+(-0.0008917658920059561)*u**39+(0.00018860451614904016)*u**40+(-3.572135997052878e-05)*u**41+(6.0468741816045375e-06)*u**42
    # f = (1140.6468638660972)*u**19+(-1060.6597006981647)*u**18+(-1020.7877968177103)*u**20+(785.2155860993576)*u**17+(743.7007700432555)*u**21+(427.74602792287925)*u**26+(-393.9619839865327)*u**27+(-385.86792044858487)*u**22+(-384.22512846376804)*u**25+(-377.776123457081)*u**16+(-326.40444288057654)*u**13+(319.90938366309814)*u**28+(301.58834872204307)*u**14+(-235.17827760528797)*u**29+(231.68401929889671)*u**24+(-190.27097954487633)*u**10+(158.1496431816656)*u**30+(156.7766384249286)*u**9+(149.98787643035521)*u**12+(-97.4602840386209)*u**31+(74.08149290872839)*u**11+(-54.98820196165385)*u**8+(54.903220110199854)*u**32+(36.441237597487614)*u**23+(-35.56500957454099)*u**15+(-28.161140605005976)*u**33+(-26.670373234244344)*u**3+(18.67582702169883)*u**6+(15.520602267403309)*u**4+(-13.93405020067454)*u**7+(13.101436992216232)*u**34+(9.25057870675284)*u**2+(-5.512108234727928)*u**35+(-3.2004048188078578)*u**5+(2.719721244956145)*u+(2.0931749032992446)*u**36+(-0.7166821131841308)*u**37+(0.22116096905428723)*u**38+(-0.061511633383306716)*u**39+(0.01927562811406923)*1+(0.01542264599657528)*u**40+(-0.0034863897969654918)*u**41+(0.0007104151462131264)*u**42+(-0.0001303507056856052)*u**43+(2.1482443488911565e-05)*u**44+(-3.164640935431262e-06)*u**45
    f = (1977.2126294449358)*u**17+(-1741.0625908701109)*u**18+(-1556.288314943953)*u**16+(-1459.2587702691008)*u**13+(1151.7860139286006)*u**12+(1134.742838728226)*u**19+(770.1813433767503)*u**14+(587.641146145812)*u**9+(-530.6197704678854)*u**10+(483.9313302232636)*u**15+(-483.5919157920183)*u**20+(-348.0217957546191)*u**23+(314.81454933124627)*u**24+(269.41855847955685)*u**22+(-236.86026697775043)*u**25+(-211.8890939616365)*u**8+(-207.5518152750119)*u**11+(156.85044383258023)*u**26+(117.92481866485281)*u**6+(-93.87530820196191)*u**27+(-89.72485042925268)*u**7+(51.54068298636804)*u**28+(-39.90957387435231)*u**5+(-37.912361708295)*u**3+(27.356146847071678)*u**4+(-26.20094487838345)*u**29+(14.750915324056038)*u**2+(12.40928168228315)*u**30+(-5.49938715116297)*u**31+(-4.839262613328451)*u**21+(2.287442533306094)*u**32+(1.9084885835784648)*u+(-0.8949406112502072)*u**33+(0.32982557751174424)*u**34+(-0.11460589608135148)*u**35+(0.03755971834942448)*u**36+(-0.025839343299501445)*1+(-0.01160830950124202)*u**37+(0.003381044708888333)*u**38+(-0.0009268531081299575)*u**39+(0.00023866383909264125)*u**40+(-5.756274046647313e-05)*u**41+(1.2952979452796207e-05)*u**42+(-2.7050265951207157e-06)*u**43
    return f

def a_real_function(u):
    if u < 0.5:
        f = 0.0
    else:
        # f = 1.0
        f = (u - 0.5) * u
    return f
# def a_predict_function(u):
#     if u <= 0.5:
#         y = (-0.656464743609309)*1+(-0.19078714955844395)*u
#     else:
#         y = (1.1412332456160317)*1+(-0.19258733840113776)*u
#     if y <= 0.0:
#         return 0.0
#     else:
#         return y
def a_predict_function(u):
    if u <= 0.25:
        y = (0.10131260368113672)*u+(-0.01019366137761922)*1
    elif u > 0.25 and u < 0.5:
        y = (-0.05844199267150074)*u+(0.05122971899458784)*1
    elif u >= 0.5 and u < 0.75:
        y = (0.6929562125370112)*u+(-0.3562683837245876)*1
    else:
        y = (-0.37134620917119565)*u+(0.033752880145923335)*1
    return np.abs(y)



def A_real_function():
    N_a = 400
    u_a_np = np.zeros(N_a + 1, dtype=float)
    u_a_0 = 0.0
    u_a_np[0] = u_a_0
    for i in range(1, N_a + 1):
        u_a_np[i] = u_a_0 + (1.0 / N_a) * i
    a_real = np.array([a_real_function(ele) for ele in u_a_np])
    A_a_real = np.zeros(N_a + 1, dtype=float)
    for index in range(1, N_a + 2):
        A_a_real[index - 1] = np.trapz(a_real[0:index], u_a_np[0:index])
    return A_a_real

def A_predict_function():
    N_a = 400
    u_a_np = np.zeros(N_a + 1, dtype=float)
    u_a_0 = 0.0
    u_a_np[0] = u_a_0
    for i in range(1, N_a + 1):
        u_a_np[i] = u_a_0 + (1.0 / N_a) * i
    a_predict = np.array([a_predict_function(ele) for ele in u_a_np])
    A_a_predict = np.zeros(N_a + 1, dtype=float)
    for index in range(1, N_a + 2):
        A_a_predict[index - 1] = np.trapz(a_predict[0:index], u_a_np[0:index])
    return A_a_predict

def plot_real_and_predict_function(name='name'):
    x = []
    N = 100
    dx = 1.0/N
    for i in range(N + 1):
        x.append(i*dx)
    f_real = np.array([f_real_function(ele) for ele in x])
    f_predict = np.array([f_predict_function(ele) for ele in x])

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    # plt.plot()
    plot_1, = plt.plot(x, f_real, label='observation', color='red', linestyle='-', linewidth=linewith)
    # plot_2, = plt.plot(x, f_predict, label='prediction', color='orange', linestyle='-', linewidth=linewith)
    plot_3, = plt.plot(x, f_predict - (f_predict[0] - f_real[0]), label='fix', color='orange', linestyle='--', linewidth=linewith)

    plt.xlabel('u', fontsize=label_fontsize)
    plt.ylabel('f(u)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_3],  # ,  plot_2
               labels=['exact', 'revised prediction'],  # 'prediction',
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    # ax1.set_facecolor('w')
    plt.show()
    #  保存
    save_dir = 'figures/' + name + '_function'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + name + '_f' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_real_and_predict_a_function(name='name'):
    x = []
    N = 100
    dx = 1.0/N
    for i in range(N + 1):
        x.append(i*dx)
    f_real = np.array([a_real_function(ele) for ele in x])
    f_predict = np.array([a_predict_function(ele) for ele in x])

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    # plt.plot()
    plot_1, = plt.plot(x, f_real, label='observation', color='red', linestyle='-', linewidth=linewith)
    # plot_2, = plt.plot(x, f_predict, label='prediction', color='orange', linestyle='-', linewidth=linewith)
    plot_3, = plt.plot(x, f_predict, label='fix', color='orange', linestyle='--', linewidth=linewith)

    plt.xlabel('u', fontsize=label_fontsize)
    plt.ylabel('f(u)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_3],  # ,  plot_2
               labels=['exact', 'prediction'],  # 'prediction',
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    # ax1.set_facecolor('w')
    plt.show()
    # #  保存
    save_dir = 'figures/' + name + '_function'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + name + '_a' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_real_and_predict_A_function(name='name'):
    A_real = A_real_function()
    A_predict = A_predict_function()
    x = []
    N = 400
    dx = 1.0/N
    for i in range(N + 1):
        x.append(i*dx)
    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    # plt.plot()
    plot_1, = plt.plot(x, A_real, label='observation', color='red', linestyle='-', linewidth=linewith)
    # plot_2, = plt.plot(x, f_predict, label='prediction', color='orange', linestyle='-', linewidth=linewith)
    plot_3, = plt.plot(x, A_predict, label='fix', color='orange', linestyle='--', linewidth=linewith)

    plt.xlabel('u', fontsize=label_fontsize)
    plt.ylabel('f(u)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_3],  # ,  plot_2
               labels=['exact', 'prediction'],  # 'prediction',
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    # ax1.set_facecolor('w')
    plt.show()
    #  保存
    save_dir = 'figures/' + name + '_function'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + name + '_A_u' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


def plot_observation_distribution(time_points, name):
    all_data = np.load('data/' + name + '_U.npy')
    time_points = time_points
    data = all_data[time_points, :, :].flatten()
    # new_data = []
    # for ele in data:
    #     if ele != 0.0 and ele != 1.0 and ele != 0.8 and ele != 0.6 and ele != 0.4 and ele != 0.3 and ele != 0.7:
    #         new_data.append(ele)
    new_data = data
    weights = [1./len(new_data)] * len(new_data)
    label_fontsize = 50
    ticks_fontsize = 50
    linewith_frame = 4
    fig = plt.figure(figsize=(40, 20))
    # plt.hist(new_data, weights=weights, bins=50)
    plt.hist(new_data, bins=50)
    plt.ylim(0, 1000)
    plt.xlabel('u', fontsize=label_fontsize)
    plt.ylabel('proportion', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    # major_ticks_top = np.linspace(0, 1, 21)
    # ax1.set_yticks(major_ticks_top)
    plt.grid(linewidth=0.2)
    plt.show()
    # #  保存
    # save_dir = 'figures/' + name + '_observation_distribution'
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    # file_name = '/' + 'observation_distribution_all' + '.pdf'
    # fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


def plot_fixed_time_step_add_noise(real_data, real_no_noise_data, predict_data, time_steps_real, time_steps_predict, time, example_index,cell_numbers, name):

    fix_timestep_real_data = real_data[time_steps_real, example_index, :]
    fix_timestep_real_no_noise_data = real_no_noise_data[time_steps_real, example_index, :]
    fix_timestep_predict_data = predict_data[time_steps_predict, example_index, :]

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    x_label = []
    dx = 10/200   # 0.05
    for i in range(len(fix_timestep_real_data)):
        x_label.append(i * dx)
    plot_1, = plt.plot(x_label, fix_timestep_real_no_noise_data, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2, = plt.plot(x_label, fix_timestep_predict_data, label='prediction', color='blue', linestyle='--', linewidth=linewith)
    plot_3 = plt.scatter(x_label, fix_timestep_real_data, c="black")
    # plt.title(label='X', fontsize=title_fontsize)
    plt.xlabel('x', fontsize=label_fontsize)
    plt.ylabel('u(x)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_2, plot_3],
               labels=['exact', 'prediction', 'noise'],
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.show()

    save_dir = 'figures/' + name + '_' +'time_' + str(time)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'fixed_time_step_real_predict_data' + '_' + 'example' + '_' + str(example_index) + '_time_' + str(time) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


def plot_init_states(data, example_index, name):
    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    init_states = data[0]
    init_state = init_states[example_index]
    x_label = []
    dx = 10/400   # 0.05
    for i in range(400):
        x_label.append(i * dx)
    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    plot_1, = plt.plot(x_label, init_state, label='observation', color='blue', linestyle='-', linewidth=linewith)
    plt.xlabel('x', fontsize=label_fontsize)
    plt.ylabel('u(x)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.ylim(-0.1, 1.1)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    plt.show()
    # #  保存
    save_dir = 'figures/' + name + '_initial_states'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'initial_state_data' + '_' + 'example' + '_' + str(example_index) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


# 画出真实和噪声数据
def plot_fixed_time_step_real_noise_data(real_data, predict_data, time_steps_real, time_steps_predict, time, example_index, name):

    fix_timestep_real_data = real_data[time_steps_real, example_index, :]
    fix_timestep_noise_data = predict_data[time_steps_predict, example_index, :]

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    x_label = []
    dx = 10/200   # 0.05
    for i in range(len(fix_timestep_real_data)):
        x_label.append(i * dx)
    plot_1, = plt.plot(x_label, fix_timestep_real_data, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2 = plt.scatter(x_label, fix_timestep_noise_data, c="black")
    plt.xlabel('x', fontsize=label_fontsize)
    plt.ylabel('u(x)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_2],
               labels=['clean', 'noise'],
               loc="upper right", fontsize=50, frameon=True, edgecolor='green')
    # plt.show()

    save_dir = 'figures/' + name + '_' + 'exact_noise'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'fixed_time_step_real_noise_data' + '_' + 'example' + '_' + str(example_index) + '_time_' + str(time) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


def plot_loss_function_all(name):

    f = open("checkpoint/" + name + "/" + "loss.txt", encoding="utf8")
    line = f.readline()
    value_list = []
    while line:
        if "data loss0" in line and "nan" not in line:
            # print(line)
            number = re.findall(r'\d+(?:\.\d+)?', line)
            # print(number)
            value_list.append(float(number[1]))
        line = f.readline()
    value_list.remove(0.422315)
    x = np.arange(0, len(value_list))
    linewith = 5
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    plot_1, = plt.plot(x, value_list, label='loss', color='red', linestyle='-', linewidth=linewith)
    plt.xlabel('Iterations', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)

    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    plt.show()
    # #  保存
    save_dir = 'figures/' + name + '_loss'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'loss_function_all' + '.png'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_loss_function_sub(name, index, limit):

    f = open("checkpoint/" + name + "/" + "loss.txt", encoding="utf8")
    line = f.readline()
    value_list = []
    while line:
        if "data loss0" in line and "nan" not in line:
            # print(line)
            number = re.findall(r'\d+(?:\.\d+)?', line)
            # print(number)
            value_list.append(float(number[1]))
        line = f.readline()
    linewith = 2
    linewith_frame = 1
    label_fontsize = 10
    ticks_fontsize = 10

    new_value_list = []
    for i in range(len(value_list)):
        if i >= index:
            new_value_list.append(value_list[i])
    x = np.arange(0, len(new_value_list))
    fig = plt.figure(figsize=(8, 5))
    plot_1, = plt.plot(x, new_value_list, label='loss', color='green', linestyle='-', linewidth=linewith)
    # plt.xlabel('Iterations', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlim(0, len(new_value_list))
    plt.ylim(0.0, limit)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)

    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    plt.show()
    # #  保存
    save_dir = 'figures/' + name + '_loss'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'loss_function_sub' + '.png'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)







if __name__ == "__main__":
    N = 200
    test_data_name = 'beta_300_case_2_N_200_0.5_obs_3'
    real_data_file = 'data/' + test_data_name + '_U' + '.npy'
    predict_data_file = 'data/' + 'predict_' + test_data_name + '_U' + '.npy'

    real_time_step = 910
    predict_time_step = 614
    time = 2
    example_index = 0
    real_data = np.load(real_data_file)
    predict_data = np.load(predict_data_file)

    # # 真实和预测以及误差情况
    # plot_real_and_predict_data(real_data, predict_data, example_index, name=test_data_name)

    # 固定时间下真实和预测情况
    # (不含有噪声)
    plot_fixed_time_step(real_data, predict_data, real_time_step, predict_time_step, time, N, example_index, name=test_data_name)
    #（含有噪声）
    # cell_numbers = N
    # real_no_noise_data = np.load('data/N_400_example_2_dt_0.1_layer_10_beta_0.5_no_alpha_U' + '.npy')
    # plot_fixed_time_step_add_noise(real_data, real_no_noise_data, predict_data, real_time_step, predict_time_step, time, example_index, cell_numbers, name=test_data_name)

    # f(u)函数的差别
    # plot_real_and_predict_function(name=test_data_name)
    # plot_real_and_predict_a_function(name=test_data_name)
    # plot_real_and_predict_A_function(name=test_data_name)

    # 画出观测数据得分布图
    # time_point_list = [0, 44, 88, 132, 176, 220, 264, 308, 352, 396]
    # plot_observation_distribution(time_points=time_point_list, name=test_data_name)

    # 画出噪音数据下的初始状态
    # example_index = 0
    # plot_init_states(real_data, example_index, test_data_name)



    # 比较真实数据和加了误差的数据
    # time_steps_real = 101   # 34(03), 67(06), 101(09)
    # time_steps_predict = time_steps_real
    # time = 0.9
    # example_index = 0
    # plot_fixed_time_step_real_noise_data(real_no_noise_data, real_data, time_steps_real, time_steps_predict, time, example_index, name=test_data_name)



    # 画出损失函数
    # plot_loss_function_all(name=test_data_name)
    # index = 168
    # limit = 0.0001
    # plot_loss_function_sub(name=test_data_name, index=index, limit=limit)







