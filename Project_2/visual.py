import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""

Type 1: plot mean of dtv across iterations 1-100 for different methods

"""
df0 = pd.read_csv("dataframe_param3_A_known_method0.csv", header = None)
df1 = pd.read_csv("dataframe_param3_A_known_method1.csv", header = None)
df2 = pd.read_csv("dataframe_param3_A_known_method2.csv", header = None)

#df_dtv0 = pd.read_excel("dtv_method2_iter.xlsx",header=None)
dtv_means0 = [np.nanmean(df0.iloc[:,indeks]) for indeks in range(df0.shape[1])]
dtv_means1 = [np.nanmean(df1.iloc[:,indeks]) for indeks in range(df1.shape[1])]
dtv_means2 = [np.nanmean(df2.iloc[:,indeks]) for indeks in range(df2.shape[1])]


plt.figure(1)
plt.scatter([x for x in range(1,101)], dtv_means0, color="blue",s=20, label='method 0',alpha=0.3)
plt.scatter([x for x in range(1,101)], dtv_means1, color='red',s=20,label='method 1', alpha = 0.3)
plt.scatter([x for x in range(1,101)], dtv_means2, color='green',s=20,label='method 2', alpha = 0.3)

plt.title(r"DTV between two iterations")
plt.xlabel(r"iteration")
plt.ylabel("DTV")
plt.xticks(np.arange(1, 101, step=10))
plt.legend()
plt.show()


"""

Type 2: plot boxplot of final dtv for different method
        with respect to "large sample" and "small sample"

"""

general0 = pd.read_csv("alpha_known_method0.csv")
general0 = general0.drop(columns=['Unnamed: 0'], axis=1)

# set values to differentiate small and large sample
w_cons_small = [3,4,5]
k_cons_small = [10,100]

w_cons_large = [25,50,70,100]
k_cons_large = [1000]

general0_small = general0.loc[general0['w'].isin(w_cons_small)]
general0_small = general0_small.loc[general0_small['k'].isin(k_cons_small)]

general0_large = general0.loc[general0['w'].isin(w_cons_large)]
general0_large = general0_large.loc[general0_large['k'].isin(k_cons_large)]

# box plot for method 0 and two sample types
fig = plt.figure()
data = [general0_small.loc[:,"final_dtv"],general0_large.loc[:,"final_dtv"]]
ax = fig.add_axes([0,0,1,1])
plt.title(r"final DTV method 0")
plt.xlabel(r"sample size")
plt.ylabel("final DTV")
bp=ax.boxplot(data)
plt.xticks([1, 2], ['small', 'large'])
plt.show()

# change method number for title

def final_dtv_boxplot(general0):
    # set values to differentiate small and large sample
    w_cons_small = [3,4,5]
    k_cons_small = [10,100]
    
    w_cons_large = [25,50,70,100]
    k_cons_large = [1000]
    
    general0_small = general0.loc[general0['w'].isin(w_cons_small)]
    general0_small = general0_small.loc[general0_small['k'].isin(k_cons_small)]
    
    general0_large = general0.loc[general0['w'].isin(w_cons_large)]
    general0_large = general0_large.loc[general0_large['k'].isin(k_cons_large)]
    
    # box plot for method 0 and two sample types
    fig = plt.figure()
    data = [general0_small.loc[:,"final_dtv"][~np.isnan(general0_small.loc[:,"final_dtv"])],general0_large.loc[:,"final_dtv"][~np.isnan(general0_large.loc[:,"final_dtv"])]]
    ax = fig.add_axes([0,0,1,1])
    plt.title(r"final DTV method 2")
    plt.xlabel(r"sample size")
    plt.ylabel("final DTV")
    bp=ax.boxplot(data)
    plt.xticks([1, 2], ['small', 'large'])
    plt.show()

general1 = pd.read_csv("alpha_known_method1.csv")
general1 = general1.drop(columns=['Unnamed: 0'], axis=1)
final_dtv_boxplot(general1)

general2 = pd.read_csv("alpha_known_method2.csv")
general2 = general2.drop(columns=['Unnamed: 0'], axis=1)
final_dtv_boxplot(general2)


"""

Type 3: plot dtv across iterations 1-20 for specific sample

"""
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

large0 = pd.read_csv("#large_param3_alpha_yes_method_0.csv")
large0 = large0.drop(columns=['Unnamed: 0'], axis=1)
large0 = large0.T
ax = large0.plot(legend=False)
plt.title('DTV between two consecutive iterations, method 0')
plt.text(20,0.06,r'$w=6, k=2000, \alpha=0.6$', fontdict=font)
plt.xlabel('iterations')
plt.ylabel('dtv')
plt.show()

# for method 1:
large1 = pd.read_csv("#large_param3_alpha_yes_method_1.csv")
large1 = large1.drop(columns=['Unnamed: 0'], axis=1)
large1 = large1.T
ax = large1.plot(legend=False)
plt.title('DTV between two consecutive iterations, method 1')
plt.text(20,0.04,r'$w=6, k=2000, \alpha=0.6$', fontdict=font)
plt.xlabel('iterations')
plt.ylabel('dtv')
plt.show()

# for method 2:
large2 = pd.read_csv("#large_param3_alpha_yes_method_2.csv")
large2 = large2.drop(columns=['Unnamed: 0'], axis=1)
large2 = large2.T
ax = large2.plot(legend=False)
plt.title('DTV between two consecutive iterations, method 2')
plt.text(20,0.06,r'$w=6, k=2000, \alpha=0.6$', fontdict=font)
plt.xlabel('iterations')
plt.ylabel('dtv')
plt.show()

"""

Type 4: box plot of dtv for different alpha

"""

df_alpha = pd.read_excel('dtv_small_alpha_known.xlsx')
df_alpha.boxplot(grid=False)

# small sample
df_alpha = pd.read_excel('alphas.xlsx')
df_alpha.iloc[:,0] = abs(df_alpha.iloc[:,0]-0.1)
df_alpha.iloc[:,1] = abs(df_alpha.iloc[:,1]-0.3)
df_alpha.iloc[:,2] = abs(df_alpha.iloc[:,2]-0.5)
df_alpha.iloc[:,3] = abs(df_alpha.iloc[:,3]-0.7)
df_alpha.iloc[:,4] = abs(df_alpha.iloc[:,4]-0.9)

df_alpha.boxplot(grid=False)


df_dtv = pd.read_excel('dtvs.xlsx')
df_dtv.boxplot(grid=False)
