
import torch
import torch.nn as nn
import torch.optim as optim


# ダミーデータの生成
# 入力ベクトル(inv)と出力ベクトル(outv)は同じ次元を持つと仮定します。
batch_size = 64
input_dim = 10

# inv は0または1、d_inv と d は連続値です
prob_of_one = 0.80
random_tensor = torch.rand(batch_size, input_dim)
inv = (random_tensor < prob_of_one).float()  # True/Falseを0/1に変換
### inv = torch.randint(0, 2, (batch_size, input_dim)).float()  # 0または1の値
d_inv = torch.rand(batch_size, input_dim) * 0.1
d = torch.rand(batch_size)
print(d_inv[0])
print(d[0])

# モデルの定義（単純な線形層を使用）
# 今回は入力として inv のみをモデルに渡します。
### model = nn.Linear(input_dim, input_dim)
# モデルの定義
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.linear(x)
        x = self.softplus(x)  # Softplus活性化関数を適用して非負の値を保証
        return x
# モデルのインスタンス化
model = MyModel()  

# 新しいカスタム損失関数の定義
# 今回は d_inv_i を損失関数の計算にのみ使用し、モデルの入力には含めません。
class CustomLossV2(nn.Module):
    def forward(self, inv, d_inv, outv, d):
        # カスタム損失関数の計算 (sum_i (inv_i * d_inv_i * outv_i) - d_i)^2
        weighted_sum = torch.sum(inv * d_inv * outv, dim=1)
        loss = torch.mean((weighted_sum - d) ** 2)
        return loss

# 新しい損失関数のインスタンス化
criterion_v2 = CustomLossV2()

# オプティマイザーの定義
optimizer = optim.Adam(model.parameters(), lr=0.001)

# トレーニングループ
model.train()
for epoch in range(1000):  # 100エポックでトレーニングします。
    optimizer.zero_grad()   # 勾配を0で初期化
    output = model(inv)     # 入力データをモデルに通して出力を得る
    loss = criterion_v2(inv, d_inv, output, d)  # 損失関数を計算
    loss.backward()         # バックプロパゲーション
    optimizer.step()        # パラメータの更新

    print(f"Epoch {epoch+1}/{1000}, Loss: {loss.item()}")  # 進捗の表示

# 最終的な損失値を出力
loss.item()

# テスト用のダミーデータセットを生成します
# モデルが訓練された後、実際の入力`inv`に対して出力`outv`を生成するか確認します
# inv は0または1、d_inv と d は連続値です
prob_of_one = 0.80
random_tensor = torch.rand(batch_size, input_dim)
test_inv = (random_tensor < prob_of_one).float()  # True/Falseを0/1に変換
### test_inv = torch.randint(0, 2, (batch_size, input_dim)).float()

# モデルを評価モードに設定します
model.eval()

# 出力を計算します
with torch.no_grad():  # 勾配計算を無効化
    test_outv = model(test_inv)

print(test_inv[0])
print(test_outv[0])
