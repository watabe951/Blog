import matplotlib.pyplot as plt
import torch
import pyro
from pyro.distributions import Normal, Delta
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc.nuts import NUTS
from pyro.infer.mcmc.util import predictive


# =============================
# ニューラルネットワークモデル
# =============================
class BNN(object):
    def __init__(self, input_size, hidden_size, output_size, w_sigma, y_sigma):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w_sigma = w_sigma
        self.y_sigma = y_sigma

    def model(self, x_data, y_data):
        # パラメータの生成
        with pyro.plate("w1_plate_dim2", self.hidden_size):
            with pyro.plate("w1_plate_dim1", self.input_size):
                w1 = pyro.sample("w1", Normal(0, self.w_sigma))
        with pyro.plate("w2_plate_dim2", self.output_size):
            with pyro.plate("w2_plate_dim1", self.hidden_size):
                w2 = pyro.sample("w2", Normal(0, self.w_sigma))

        f = lambda x: torch.mm(torch.tanh(torch.mm(x, w1)), w2)
        # 観測データの生成
        with pyro.plate("map", len(x_data)):
            prediction_mean = f(x_data).squeeze()
            pyro.sample("obs", Normal(prediction_mean, self.y_sigma), obs=y_data)
            return prediction_mean

    def nuts_sampling(self, x_data, y_data, num_samples, warmup_steps):
        nuts_kernel = NUTS(self.model, target_accept_prob=0.99)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
        mcmc.run(x_data, y_data)
        self.posterior_samples = mcmc.get_samples()

    def predict(self, x_pred):
        def wrapped_model(x_data, y_data):
            pyro.sample("prediction", Delta(self.model(x_data, y_data)))

        samples = predictive(wrapped_model, self.posterior_samples, x_pred, None)
        return samples["prediction"], samples["obs"]


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
    # モデル
    bnn = BNN(H_0, H_1, D, w_sigma, y_sigma)
    # サンプリング
    bnn.nuts_sampling(x_data, y_data, num_samples=1000, warmup_steps=500)
    # 予測
    x = torch.linspace(-6.0, 6.0, 1000).reshape(-1, 1)
    x_new = torch.cat([x, torch.ones_like(x)], dim=1)  # 予測入力点
    mu_samples, y_samples = bnn.predict(x_new)

    # 結果の図示
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    for sample in mu_samples:
        ax[0].plot(x.flatten().numpy(), sample.detach().numpy(), alpha=0.05, color='g')
    ax[0].scatter(x_data[:, 0].flatten().numpy(), y_data.numpy(), color="black")
    ax[0].set(xlabel="$x$", ylabel="$y$", xlim=(-6.0, 6.0), ylim=(-0.6, 0.6))

    for sample in y_samples:
        ax[1].plot(x.flatten().numpy(), sample.detach().numpy(), alpha=0.005, color='g')
    ax[1].scatter(x_data[:, 0].flatten().numpy(), y_data.numpy(), color="black")
    ax[1].set(xlabel="$x$", ylabel="$y$", xlim=(-6.0, 6.0), ylim=(-0.6, 0.6))
    plt.show()


if __name__ == '__main__':
    main()
