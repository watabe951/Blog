import matplotlib.pyplot as plt
import torch
import pyro
from pyro.distributions import Normal, Delta
from pyro.infer.autoguide.guides import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.predictive import Predictive
import maketrain as mt
import pickle

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
        prediction_mean = f(x_data).squeeze()
        for i in pyro.plate("map", len(x_data)):
            print("\ri:{}/{}".format(i,len(x_data)),end="")
            pyro.sample("obs_{}".format(i), Normal(prediction_mean[i], self.y_sigma), obs=y_data[i])
        return prediction_mean


    def VI(self, x_data, y_data, num_samples=1000, num_iterations=30000):
        self.guide = AutoDiagonalNormal(self.model)
        optim = Adam({"lr": 1e-3})
        loss = Trace_ELBO()
        svi = SVI(self.model, self.guide, optim=optim, loss=loss)

        # train
        # print("train!")
        pyro.clear_param_store()
        # print("train2!")
        for j in range(num_iterations):
            # print("train3!")
            # print("x_data.size():{}, y_data.size():{}".format(x_data.size(), y_data.size()))
            loss = svi.step(x_data, y_data)
            # print("train4!")
            # if j % (num_iterations // 10) == 0:
            print("[iteration %05d] loss: %.4f" % (j + 1, loss / len(x_data)))
        print("sample!")
        # num_samplesだけ事後分布からサンプルを生成
        dict = {}
        for i in range(num_samples):
            sample = self.guide()  # sampling
            for name, value in sample.items():
                if not dict.keys().__contains__(name):
                    dict[name] = value.unsqueeze(0)
                else:
                    dict[name] = torch.cat([dict[name], value.unsqueeze(0)], dim=0)
        self.posterior_samples = dict

    def predict(self, x_pred):
        def wrapped_model(x_data, y_data):
            pyro.sample("prediction", Delta(self.model(x_data, y_data)))

        predictive = Predictive(wrapped_model, self.posterior_samples)
        samples =  predictive.get_samples(x_pred, None)
        return samples["prediction"], samples["obs"]


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

    # モデル
    bnn = BNN(H_0, H_1, D, w_sigma, y_sigma)

    # train
    print("train!")
    bnn.VI(x_train, y_train)
    
    # test
    print("test")
    mu_samples, y_samples = bnn.predict(x_test)
    
    print("save!")
    with open("mu_samples.binaryfile","wb") as f:
        pickle.dump(mu_samples, f)
    with open("y_samples.binaryfile","wb") as f:
        pickle.dump(y_samples, f)
    with open("bnn.binaryfile", "wb") as f:
        pickle.dump(bnn,f)
    
    print("everything is end")
    # 予測
    # print("prredict!")
    # x = torch.linspace(-6.0, 6.0, 1000).reshape(-1, 1)
    # x_new = torch.cat([x, torch.ones_like(x)], dim=1)  # 予測入力点
    # mu_samples, y_samples = bnn.predict(x_new)

    # # 結果の図示
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    # for sample in mu_samples:
    #     ax[0].plot(x.flatten().numpy(), sample.detach().numpy(), alpha=0.05, color='g')
    # ax[0].scatter(x_data[:, 0].flatten().numpy(), y_data.numpy(), color="black")
    # ax[0].set(xlabel="$x$", ylabel="$y$", xlim=(-6.0, 6.0), ylim=(-0.6, 0.6))

    # for sample in y_samples:
    #     ax[1].plot(x.flatten().numpy(), sample.detach().numpy(), alpha=0.005, color='g')
    # ax[1].scatter(x_data[:, 0].flatten().numpy(), y_data.numpy(), color="black")
    # ax[1].set(xlabel="$x$", ylabel="$y$", xlim=(-6.0, 6.0), ylim=(-0.6, 0.6))
    # plt.show()


if __name__ == '__main__':
    print("hello!!!!!!!!!!!")
    main()
    print("end")
