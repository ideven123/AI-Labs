
#（隐层神经元设置为5， 4， 4， 3，激活函数设置为self.sigmoid）（1分
# 实现BP算法（1分）；
# 实现梯度下降算法（1分）
import numpy as np
import matplotlib.pyplot

class MLP4:
    def __init__(self, input_size, hidden_num1, hidden_num2, num_classes):
        # 隐层参数设置 ，初始化为随机值
        self.w = [
            np.random.rand(hidden_num1, input_size),
            np.random.rand(hidden_num2, hidden_num1),
            np.random.rand(num_classes, hidden_num2)
        ]
        self.b = [
            np.random.rand(hidden_num1, 1),
            np.random.rand(hidden_num2, 1),
            np.random.rand(num_classes, 1)
        ]
        self.forward_info = None
        self.diff_info = None

    ## 激活函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def dsigmoid(self, s):
        return s * (1 - s)

    # 正向传播 ，按步骤计算 ,并计算前向信息
    def forward(self, data):
        u1 = np.dot(self.w[0], data.T) + self.b[0]
        u2 = np.dot(self.w[1], self.sigmoid(u1)) + self.b[1]
        u3 = np.dot(self.w[2], self.sigmoid(u2)) + self.b[2]
        self.forward_info = {
            "x": data.T,
            "u1": u1,
            "u2": u2,
            "u3": u3,
            "y1": self.sigmoid(u1),
            "y2": self.sigmoid(u2),
            "y3": self.softmax(u3),
        }
        return self.forward_info["y3"]

    def predict(self, data):
        return self.forward(data).argmax(0)

    ## 静态函数 ，计算交叉熵
    def cross_entrop(self, a1, a2):
        loss = np.zeros_like(a1, dtype=float)
        for i in range(0,len(loss)):
            loss[i] = -np.log(a2[a1[i][0]][i])
        return np.mean(loss)

    def loss(self, data, label):
        return self.cross_entrop(label, self.forward(data))

    ## BP算法
    def BP(self, data, label):
        # 从最后一层开始更新参数
        ## 计算，每一层的新值
        y3 = self.forward_info["y3"].copy()
        for i in range(0,len(label)):
            y3[label[i][0]][i] = y3[label[i][0]][i] - 1
        new_w3 = y3.dot( self.forward_info["y2"].T)

        new_b2 = self.w[2].T.dot(y3) * self.dsigmoid(self.forward_info["y2"])
        new_w2 = np.dot(new_b2, self.forward_info["y1"].T)

        new_b1 = np.dot(self.w[1].T, new_b2) * self.dsigmoid(self.forward_info["y1"])
        new_w1 = np.dot(new_b1, self.forward_info["x"].T)

        ##记录后向计算结果
        row = data.shape[0]
        self.diff_info = {
            "delta_w3": np.divide(new_w3, row),
            "delta_w2": np.divide(new_w2, row),
            "delta_w1": np.divide(new_w1, row),
            "delta_b3": np.mean(y3, axis=1, keepdims=True),
            "delta_b2": np.mean(new_b2, axis=1, keepdims=True),
            "delta_b1": np.mean(new_b1, axis=1, keepdims=True),
        }

    ## 梯度下降算法
    def GradientDescent(self, eta=0.1):
        #根据计算公示计算 ， 更新w , b
        for i in range(3):
            s = "delta_b" + str(i + 1)
            self.b[i] = self.b[i] - eta * self.diff_info[s]
        for i in range(3):
            s = "delta_w" + str(i + 1)
            self.w[i] = self.w[i] - eta * self.diff_info[s]
        return


# 静态函数， 作图
def plot(loss):
    idx = list(range(len(loss)))
    matplotlib.pyplot.figure(num=4, figsize=(18, 8))
    matplotlib.pyplot.plot(idx, loss)
    matplotlib.pyplot.show()


def main():
    # 隐层神经元设置为5， 4， 4， 3
    mlp4 = MLP4(5, 4, 4, 3)
    # 数据维度为(100，5)
    traindata = np.random.rand(100, 5)
    # label的维度(100，1)
    trainlabel = np.random.randint(0, 3, (100, 1))
    loss_record = []
    epoch = 100
    for i in range(epoch):
        # 记录交叉熵损失
        loss_record.append(mlp4.loss(traindata, trainlabel))
        mlp4.BP(traindata, trainlabel)
        mlp4.GradientDescent()
    # 作loss的训练曲线
    plot(loss_record)
main()