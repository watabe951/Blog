import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pyro
from pyro.distributions import Normal
from pyro.infer.autoguide.guides import AutoDelta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


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
        with pyro.plate("map", len(x_data)):
            prediction_mean = lifted_bnn_model(x_data).squeeze()
            pyro.sample("obs", Normal(prediction_mean, self.y_sigma), obs=y_data)
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
    H_0 = 2  # 入力次元
    H_1 = 4  # 中間層のユニット数
    D = 1  # 出力次元
    # 訓練データセット
    data = torch.tensor([[-4.5, -0.22],
                         [-4.4, -0.10],
                         [-4.0, 0.00],
                         [-2.9, -0.11],
                         [-2.7, -0.33],
                         [-1.5, -0.20],
                         [-1.3, -0.08],
                         [-0.8, -0.21],
                         [0.1, -0.34],
                         [1.5, 0.10],
                         [2.0, 0.11],
                         [2.1, 0.14],
                         [2.6, 0.21],
                         [3.5, 0.23],
                         [3.6, 0.38]])
    x_data = data[:, 0].reshape(-1, 1)
    x_data = torch.cat([x_data, torch.ones_like(x_data)], dim=1)  # biasごと入力に含ませる
    y_data = data[:, 1]

    # ハイパーパラメータ
    w_sigma = torch.tensor(0.75)
    y_sigma = torch.tensor(0.09)
    # ネットワークモデル
    bnn = BNN(H_0, H_1, D, w_sigma, y_sigma)
    # 推論
    bnn.LaplaceApproximation(x_data, y_data)
    # 予測
    x = torch.linspace(-6.0, 6.0, 1000).reshape(-1, 1)
    x_news = torch.cat([x, torch.ones_like(x)], dim=1)  # 予測入力点
    y_means = []  # yの予測平均を格納するlist
    y_sigmas = []  # yの予測標準偏差を格納するlist
    # 各点で予測分布のパラメータを求める
    for x_new in x_news:
        x_new.unsqueeze_(0)
        y_new_mu, y_new_sigma = bnn.predict(x_new)
        y_means.append(y_new_mu.item())
        y_sigmas.append(y_new_sigma.item())
    y_means = np.array(y_means)  # ndarray化
    y_sigmas = np.array(y_sigmas)  # ndarray化

    # 結果の図示
    x = x.flatten().numpy()
    plt.scatter(x_data[:, 0].numpy(), y_data.numpy(), color="black")
    plt.plot(x, y_means)
    low_y = y_means - 2 * y_sigmas
    high_y = y_means + 2 * y_sigmas
    plt.fill_between(x, low_y, high_y, alpha=0.3)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.xlim(-6.0, 6.0)
    plt.ylim(-0.6, 0.6)
    plt.show()


if __name__ == '__main__':
    main()
