import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pyro
from pyro.distributions import Normal
from pyro.infer.autoguide.guides import AutoDelta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import maketrain as mt
import pickle

# =============================
# ニューラルネットワークモデル
# =============================
# バイアス項なし全結合Layerを定義
class NonBiasLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(NonBiasLinear, self).__init__()
        self.weight = nn.Parameter(data=torch.randn(input_size, output_size), requires_grad=True)

    def forward(self, input_tensor):
        return torch.mm(input_tensor, self.weight)


# 2層ニューラルネットワークモデル
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = NonBiasLinear(input_size, hidden_size)
        self.fc2 = NonBiasLinear(hidden_size, output_size)

    def forward(self, x):
        output = self.fc1(x)
        output = torch.tanh(output)
        output = self.fc2(output)
        return output


# ベイズニューラルネットワークモデル
class BNN(object):
    def __init__(self, input_size, hidden_size, output_size, w_sigma, y_sigma):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.net = Net(input_size, hidden_size, output_size)
        self.w_sigma = w_sigma
        self.y_sigma = y_sigma

    def model(self, x_data, y_data):
        # 事前分布
        w1_size = (self.input_size, self.hidden_size)
        w2_size = (self.hidden_size, self.output_size)
        w1_prior = Normal(torch.zeros(size=w1_size), self.w_sigma * torch.ones(size=w1_size))
        w2_prior = Normal(torch.zeros(size=w2_size), self.w_sigma * torch.ones(size=w2_size))
        priors = {'fc1.weight': w1_prior, 'fc2.weight': w2_prior}
        # lift
        lifted_module = pyro.random_module("module", self.net, priors)
        lifted_bnn_model = lifted_module()
        # print("len(x_data):{}".format(len(x_data)))
        prediction_mean = lifted_bnn_model(x_data).squeeze()
        for i in pyro.plate("map", len(x_data)):
            print("\r i:{}/{}".format(i,len(x_data)),end="")
            pyro.sample("obs_{}".format(i), Normal(prediction_mean[i], self.y_sigma), obs=y_data[i])
        return prediction_mean 


    # MAP推定
    def MAPestimation(self, x_data, y_data, num_iterations=10000):
        guide = AutoDelta(self.model)
        svi = SVI(self.model, guide, Adam({"lr": 1e-3}), loss=Trace_ELBO())

        # train
        pyro.clear_param_store()
        for j in range(num_iterations):
            loss = svi.step(x_data, y_data)
            if j % (num_iterations // 10) == 0:
                print("[iteration %05d] loss: %.4f" % (j + 1, loss / len(x_data)))

        # MAP推定値を取得
        param_dict = {}
        for name, value in pyro.get_param_store().items():
            param_dict[name] = value.data
        w1_MAP = param_dict['auto_module$$$fc1.weight']
        w2_MAP = param_dict['auto_module$$$fc2.weight']
        self.net.fc1.weight.data = w1_MAP
        self.net.fc2.weight.data = w2_MAP
        return w1_MAP, w2_MAP

    # ヘッセ行列の計算
    def _compute_hessian(self, x_data, hessian_size):
        hessian_matrix = torch.zeros(size=(hessian_size, hessian_size))
        for x in x_data:
            x.unsqueeze_(0)
            f = self.net.forward(x)
            f.backward(retain_graph=False)
            with torch.no_grad():
                grad_w1 = self.net.fc1.weight.grad
                grad_w2 = self.net.fc2.weight.grad
                grad_f = torch.cat([grad_w1.reshape(-1, 1), grad_w2.reshape(-1, 1)], dim=0)  # 勾配(列ベクトル)の形に整形
                hessian_matrix += torch.mm(grad_f, torch.t(grad_f))
            self.net.zero_grad()  # 勾配を0に戻す
        return hessian_matrix

    # ラプラス近似分布の計算
    def LaplaceApproximation(self, x_data, y_data):
        # 平均ベクトルについて
        w1_MAP, w2_MAP = self.MAPestimation(x_data, y_data)
        W_MAP_vector = torch.cat([w1_MAP.reshape(-1, 1), w2_MAP.reshape(-1, 1)], dim=0)
        # 共分散行列について
        M = W_MAP_vector.shape[0]
        hessian_matrix = self._compute_hessian(x_data, hessian_size=M)
        lambda_matrix = (self.w_sigma ** (-2)) * torch.eye(M) + (self.y_sigma ** (-2)) * hessian_matrix
        self.lambda_mat_inv = torch.inverse(lambda_matrix)

    # 事後予測分布の計算
    def predict(self, x_pred):
        f_pred = self.net.forward(x_pred)
        f_pred.backward(retain_graph=False)
        with torch.no_grad():
            grad_w1 = self.net.fc1.weight.grad
            grad_w2 = self.net.fc2.weight.grad
            g = torch.cat([grad_w1.reshape(-1, 1), grad_w2.reshape(-1, 1)], dim=0)
            y_pred_sigma2 = self.y_sigma ** 2 + torch.mm(torch.t(g), torch.mm(self.lambda_mat_inv, g))  # 予測分散
        self.net.zero_grad()  # 勾配を0に戻す
        return f_pred, torch.sqrt(y_pred_sigma2)


# =============================
# main関数
# =============================
def main():
    data_size = 0
    K = 4
    H_0 = 6 + 1 + 1  # 入力次元(state_dim,action_dim,bias_dim)
    H_1 = 10  # 中間層のユニット数
    D = 6  # 出力次元
    
    # 訓練データセット
    
    data_l = mt.make_train_data()
    data = torch.tensor(data_l)
    data_size = len(data)
    print(data_size)
    test_n = int(data_size/K)
    train_n = data_size - test_n
    
    x_train = data[:10000, 0:7]
    x_train = torch.cat([x_train, torch.ones_like(x_train[:,0]).reshape(-1,1)], dim=1)
    y_train = data[:10000, 7:]
    
    x_test = data[10000:20000, 0:7]
    x_test = torch.cat([x_test, torch.ones_like(x_test[:,0]).reshape(-1,1)], dim=1)
    y_test = data[10000:20000:, 7:]
    
    # ハイパーパラメータ
    w_sigma = torch.tensor(0.75)
    y_sigma = torch.tensor(0.09)
    # ネットワークモデル
    bnn = BNN(H_0, H_1, D, w_sigma, y_sigma)
    # 推論
    bnn.LaplaceApproximation(x_train, y_train)
    # 予測
    x = torch.linspace(-6.0, 6.0, 1000).reshape(-1, 1)
    x_news = torch.cat([x, torch.ones_like(x)], dim=1)  # 予測入力点
    y_means = []  # yの予測平均を格納するlist
    y_sigmas = []  # yの予測標準偏差を格納するlist
    # 各点で予測分布のパラメータを求める
    for x in x_test:
        x.unsqueeze_(0)
        y_new_mu, y_new_sigma = bnn.predict(x)
        y_means.append(y_new_mu.item())
        y_sigmas.append(y_new_sigma.item())
    
    # 結果の図示
    print("save!")
    with open("mu_samples.binaryfile","wb") as f:
        pickle.dump(mu_samples, f)
    with open("y_samples.binaryfile","wb") as f:
        pickle.dump(y_samples, f)
    with open("bnn.binaryfile", "wb") as f:
        pickle.dump(bnn,f)


if __name__ == '__main__':
    main()
