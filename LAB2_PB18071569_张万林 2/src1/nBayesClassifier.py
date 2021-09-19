# import numpy as np
# import math
# from collections import Counter
# from process_data import load_and_process_data
# from evaluation import get_micro_F1,get_macro_F1,get_acc
#
# class NaiveBayes:
#     '''参数初始化
#     Pc: P(c) 每个类别c的概率分布
#     Pxc: P(c|x) 每个特征的条件概率
#     '''
#     def __init__(self):
#         self.Pc={}
#         self.Pxc={}
#
#     '''
#     通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
#     建议全部取log，避免相乘为0
#     '''
#     def fit(self,traindata,trainlabel,featuretype):
#         '''
#         需要你实现的部分
#         '''
#         ## 计算先验概率
#         row_num = len(traindata)
#         col_num = len(traindata[0])
#         for key in trainlabel:    ## trainlabel的 key
#             if key[0] in self.Pc.keys():
#                 self.Pc[key[0]] += 1
#             else:
#                 self.Pc[key[0]] = 1
#
#         ## 构建 每种先验的条件概率
#         for key in self.Pc.keys():
#             # 计算每种类别在数据集中出现的概率
#             self.Pc[key] /= row_num
#             # 构建self.condition_prob中的key
#             self.Pxc[key] = {}
#             for i in range(col_num):
#                 #### 每个属性一个字典
#                 self.Pxc[key][i] = {}
#                 if featuretype[i] == 0:  ## 如果是离散的 ，则每个
#                     for k in np.unique(traindata[:, i], axis=0):
#                         ### 统计数字
#                         self.Pxc[key][i][k] = 0
#                 if featuretype[i] == 1:  ## 如果是连续的 ，则每个求sigma ，
#                     ## 求值为 key 的集合
#                     traindata_key = []
#                     for ith in range (0,trainlabel.shape[0]):
#                         if trainlabel[ith][0] == key:
#                             traindata_key.append(traindata[ith][i])
#                     ## 求mu,sigma
#                     mu = np.mean(traindata_key)
#                     sigma = np.std(traindata_key)
#                     self.Pxc[key][i][0] = mu
#                     self.Pxc[key][i][1] = sigma
#
#         for i in range(0,len(traindata)):
#             for j in range(0,len(traindata[i])):
#                 if featuretype[j] == 0:
#                     if traindata[i][j] in self.Pxc[trainlabel[i][0]]:
#                         self.Pxc[trainlabel[i][0]][j][traindata[i][j]] += 1
#                     else:
#                         self.Pxc[trainlabel[i][0]][j][traindata[i][j]] = 1
#
#         for label_key in self.Pxc.keys():
#             for atti_k in self.Pxc[label_key].keys():
#                 total = 0
#                 for v in self.Pxc[label_key][atti_k].values():  ## 该属性值的统计和
#                     total += v
#                 for key_atti_k in self.Pxc[label_key][atti_k].keys():
#                     # 计算每种类别确定的条件下各个特征出现的概率
#                     self.Pxc[label_key][atti_k][key_atti_k] /= total
#
#         for key in self.Pc.keys():
#             print(key)
#             for atti in range(len(traindata[0])):
#                 if featuretype[atti] == 0:
#                     print(self.Pxc[key][atti])
#                 if featuretype[atti] == 1:
#                     print(self.Pxc[key][atti])
#         # ********* End *********#
#
#     '''
#     根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
#     返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
#     feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
#     '''
#
#     @staticmethod
#     def gauss(x, mu, sigma):  ## 计算高斯概率
#         coef = 1.0 / (np.sqrt(2 * np.pi)*sigma )
#         component = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
#         return coef * component
#
#     def predict(self,features,featuretype):
#         '''
#         需要你实现的部分
#         '''
#         result = []
#         #对每条测试数据都进行预测
#         for ith,f in enumerate(features):
#             #可能的类别的概率
#             prob = np.zeros(len(self.Pc.keys()))
#             ii = 0
#             for label,label_prob in self.Pc.items():
#                 #计算概率
#                 prob[ii] = math.log(label_prob)
#                 for j in range(0,len(features[0])): ## j个属性
#                     if featuretype[j] == 0:
#                         last_prob = self.Pxc[label][j][f[j]]
#                     else:
#                         last_prob = NaiveBayes.gauss(f[j],self.Pxc[label][j][0],self.Pxc[label][j][1])
#                     prob[ii] += math.log(last_prob)
#                 ii += 1
#             #取概率最大的类别作为结果
#             result.append(list(self.Pc.keys())[np.argmax(prob)])
#         print(np.array(result).astype(int).reshape(-1,1))
#         return np.array(result).astype(int).reshape(-1,1)
#         #********* End *********#s
#
#
#
# def main():
#     # 加载训练集和测试集
#     train_data,train_label,test_data,test_label=load_and_process_data()
#     feature_type=[0,1,1,1,1,1,1,1] #表示特征的数据类型，0表示离散型，1表示连续型
#
#     Nayes=NaiveBayes()
#     Nayes.fit(train_data,train_label,feature_type) # 在训练集上计算先验概率和条件概率
#
#     pred=Nayes.predict(test_data,feature_type)  # 得到测试集上的预测结果
#     # 计算准确率Acc及多分类的F1-score
#     print("Acc: "+str(get_acc(test_label,pred)))
#     print("macro-F1: "+str(get_macro_F1(test_label,pred)))
#     print("micro-F1: "+str(get_micro_F1(test_label,pred)))
#
# main()
import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1, get_macro_F1, get_acc, total_label


class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''

    def __init__(self):
        self.Pc = {}
        self.Pxc = {}
        self.V = []  # 均值
        self.S = []  # 方差
        self.cnum = 0  # 种类个数

    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''

    def fit(self, traindata, trainlabel, featuretype):
        '''
        需要你实现的部分
        '''
        # 计算Pc
        label_set = total_label(trainlabel)  # 统计所有的类别
        N = len(label_set)
        self.cnum = N

        # 初始化
        D = len(trainlabel)
        for c in range(N):
            self.Pc[c] = 1.0 / (D + N)

        # 计算概率
        for i in range(D):
            k = trainlabel[i][0] - 1
            # print(k)
            self.Pc[k] += 1.0 / (D + N)

        # 计算Pxc
        # 对于连续数据采用高斯分布来拟合，所以先计算出对应类的所有数据的方差和均值
        Xc = []
        for i in range(len(self.Pc)):
            Xc.append([])

        # 首先对所有数据分类
        for i in range(len(trainlabel)):
            Xc[trainlabel[i][0] - 1].append(traindata[i])

        # 求对应类别的均值和方差
        for c in range(len(Xc)):
            v = [0] * len(traindata[0])  # 均值
            s = [0] * len(traindata[0])  # 方差
            x = np.transpose(np.array(Xc[c]))
            for i in range(len(traindata[0])):
                if featuretype[i] == 1:
                    v[i] = np.mean(x[i])
                    s[i] = np.var(x[i])
            self.V.append(v)
            self.S.append(s)

        # 求概率
        for c in range(len(Xc)):
            for i in range(len(traindata[0])):
                for j in range(len(Xc[c])):
                    if featuretype[i] == 1:
                        pass  # 连续数据只需计算分布即可
                    else:
                        xi = Xc[c][j][i]
                        if (c, xi) in self.Pxc:
                            self.Pxc[(c, xi)] += 1.0 / (len(Xc[c]) + 3)
                        else:
                            self.Pxc[(c, xi)] = 1.0 / (len(Xc[c]) + 3)

    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''

    def predict(self, features, featuretype):
        '''
        需要你实现的部分
        '''
        shape = [[]]
        # print(self.S)
        for X in features:
            maxpc = 0
            maxc = 0
            for c in range(self.cnum):
                pc = self.Pc[c]
                for i in range(len(X)):
                    xi = X[i]
                    if featuretype[i] == 1:  # 连续数据需要计算出概率
                        pc *= ((1.0 / math.sqrt(2 * math.pi * self.S[c][i])) * math.exp(
                            math.pow(xi - self.V[c][i], 2) / (-2 * self.S[c][i])))
                    else:  # 离散数据直接取概率即可
                        pc *= (self.Pxc[(c, xi)])
                if pc > maxpc:
                    maxpc = pc
                    maxc = c + 1
            shape[0].append(maxc)

        # 输出合适格式的结果
        return np.transpose(np.array(shape))


def main():
    # 加载训练集和测试集
    train_data, train_label, test_data, test_label = load_and_process_data()
    feature_type = [0, 1, 1, 1, 1, 1, 1, 1]  # 表示特征的数据类型，0表示离散型，1表示连续型

    Nayes = NaiveBayes()
    Nayes.fit(train_data, train_label, feature_type)  # 在训练集上计算先验概率和条件概率

    pred = Nayes.predict(test_data, feature_type)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: " + str(get_acc(test_label, pred)))
    print("macro-F1: " + str(get_macro_F1(test_label, pred)))
    print("micro-F1: " + str(get_micro_F1(test_label, pred)))


main()
