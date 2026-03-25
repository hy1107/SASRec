# SASRec 使用教學手冊

---

## 目錄

1. [環境設定](#1-環境設定)
2. [資料準備](#2-資料準備)
3. [查看資料內容](#3-查看資料內容)
4. [訓練模型](#4-訓練模型)
5. [參數說明與調整](#5-參數說明與調整)
6. [查看訓練結果](#6-查看訓練結果)
7. [在 Google Colab 使用](#7-在-google-colab-使用)
8. [常見問題](#8-常見問題)

---

## 1. 環境設定

### 本機安裝

```bash
# 建立虛擬環境（需要 Python 3.10）
python3.10 -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows

# 安裝套件
pip install -e .

# 選用：啟用 wandb 訓練紀錄
pip install -e ".[logging]"

# 選用：安裝測試工具
pip install -e ".[dev]"
```

---

## 2. 資料準備

### 2-1. 使用內建資料集

目前支援四個內建資料集：

| 名稱 | Key | 說明 |
|---|---|---|
| Amazon Beauty | `beauty` | 美妝產品評分 |
| Amazon Video Games | `video_games` | 電玩產品評分 |
| Steam | `steam` | 遊戲評論 |
| MovieLens 1M | `ml-1m` | 電影評分經典資料集 |

**查看各資料集的下載資訊：**

```bash
python scripts/show_data_info.py
```

**下載並預處理（以 ml-1m 為例）：**

```bash
# 下載
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip -P data/raw/

# 預處理（自動切割 train/valid/test）
python scripts/preprocess.py --dataset ml-1m
```

### 2-2. 使用自己的資料

準備一個 CSV 檔，格式如下（**需要 header**）：

```
user_id,item_id,timestamp
user001,item_A,1609459200
user001,item_B,1609462800
user002,item_A,1609466400
...
```

- `user_id`：使用者 ID（字串或數字皆可）
- `item_id`：物品 ID（字串或數字皆可）
- `timestamp`：互動時間（數字，用於排序）

**執行預處理：**

```bash
python scripts/preprocess.py \
    --input_path data/raw/mydata.csv \
    --fmt csv \
    --output_dir data/processed/mydata
```

**其他格式：**

```bash
# Amazon ratings CSV（無 header，格式：user,item,rating,timestamp）
python scripts/preprocess.py --input_path ratings.csv --fmt amazon_csv --output_dir data/processed/amazon

# MovieLens .dat 或 .zip
python scripts/preprocess.py --input_path ml-1m.zip --fmt movielens --output_dir data/processed/ml-1m

# Steam JSON（支援 .gz 壓縮）
python scripts/preprocess.py --input_path reviews.json.gz --fmt steam_json --output_dir data/processed/steam
```

**調整過濾門檻：**

```bash
# 預設 min_interactions=5（互動次數少於 5 的使用者/物品會被過濾）
python scripts/preprocess.py --dataset beauty --min_interactions 10
```

### 2-3. 預處理輸出檔案

預處理完成後，`data/processed/<dataset>/` 資料夾會有四個檔案：

```
data/processed/ml-1m/
├── train.txt       # 訓練序列
├── valid.txt       # 驗證集（每人最後第 2 個互動）
├── test.txt        # 測試集（每人最後 1 個互動）
└── item_count.txt  # 物品總數
```

---

## 3. 查看資料內容

### 3-1. 用 Python 讀取並檢視

```python
from sasrec.data.preprocessor import load_processed_data

train_data, valid_data, test_data, item_count = load_processed_data("data/processed/ml-1m")

print(f"使用者數量：{len(train_data)}")
print(f"物品總數：{item_count}")

# 查看某個使用者的訓練序列
uid = 1
print(f"\n使用者 {uid} 的訓練序列（前 10 個）：{train_data[uid][:10]}")
print(f"使用者 {uid} 的驗證物品：{valid_data[uid]}")
print(f"使用者 {uid} 的測試物品：{test_data[uid]}")
```

輸出範例：
```
使用者數量：6040
物品總數：3706
使用者 1 的訓練序列（前 10 個）：[1193, 661, 914, 3408, 2355, 1197, ...]
使用者 1 的驗證物品：2804
使用者 1 的測試物品：594
```

### 3-2. 統計序列長度分布

```python
import numpy as np
from sasrec.data.preprocessor import load_processed_data

train_data, valid_data, test_data, item_count = load_processed_data("data/processed/ml-1m")

lengths = [len(seq) for seq in train_data.values()]
print(f"序列長度統計：")
print(f"  最短：{min(lengths)}")
print(f"  最長：{max(lengths)}")
print(f"  平均：{np.mean(lengths):.1f}")
print(f"  中位數：{np.median(lengths):.1f}")
```

### 3-3. 直接查看原始檔案

```bash
# 看 train.txt 前 5 行（格式：user_id item1 item2 ...）
head -5 data/processed/ml-1m/train.txt

# 看 valid.txt 前 5 行（格式：user_id item）
head -5 data/processed/ml-1m/valid.txt

# 看物品總數
cat data/processed/ml-1m/item_count.txt
```

---

## 4. 訓練模型

### 4-1. 基本訓練指令

```bash
# 使用內建資料集
python scripts/run_experiment.py --dataset ml-1m

# 使用自己的資料
python scripts/run_experiment.py --data_dir data/processed/mydata
```

### 4-2. 訓練過程輸出說明

```
Epoch   1 | loss=0.6923 | NDCG@5=0.1823 | HR@5=0.2541 | NDCG@10=0.2134 | HR@10=0.3621 | NDCG@20=0.2456 | HR@20=0.4812
Epoch   2 | loss=0.6541 | NDCG@5=0.2156 | HR@5=0.2934 | NDCG@10=0.2478 | HR@10=0.4012 | ...
...
Early stopping at epoch 87 (patience=20)
Test metrics: NDCG@5=0.3012 | HR@5=0.4123 | NDCG@10=0.3456 | HR@10=0.5234 | ...
```

- **loss**：訓練損失（越低越好）
- **NDCG@K**：Normalized Discounted Cumulative Gain，考慮排名位置的準確率（越高越好）
- **HR@K**：Hit Rate，前 K 名中有沒有正確答案（越高越好）
- **Early stopping**：連續 20 個 epoch 沒有改善就自動停止

---

## 5. 參數說明與調整

所有參數的預設值在 `configs/base.yaml`，可以直接修改檔案，或在指令後面加 `key=value` 覆蓋。

### 5-1. 完整參數列表

| 參數 | 預設值 | 說明 |
|---|---|---|
| **模型架構** | | |
| `model.hidden_units` | 50 | Embedding 維度，越大模型越複雜 |
| `model.num_blocks` | 2 | Transformer 層數 |
| `model.num_heads` | 1 | Multi-head attention 的 head 數量 |
| `model.dropout_rate` | 0.2 | Dropout 機率，防止過擬合 |
| **資料** | | |
| `data.maxlen` | 50 | 輸入序列的最大長度，超過會截斷 |
| `data.num_neg_eval` | 100 | 評估時每個使用者的負樣本數量 |
| **訓練** | | |
| `train.lr` | 0.001 | Adam 學習率 |
| `train.beta1` | 0.9 | Adam β1 |
| `train.beta2` | 0.98 | Adam β2 |
| `train.weight_decay` | 0.0 | L2 正則化 |
| `train.batch_size` | 128 | 每個 batch 的使用者數量 |
| `train.num_epochs` | 200 | 最大訓練 epoch 數 |
| `train.patience` | 20 | Early stopping 等待 epoch 數 |
| `train.seed` | 42 | 隨機種子（保證可重現） |
| **評估** | | |
| `eval.k_values` | [5, 10, 20] | 計算 NDCG@K 和 HR@K 的 K 值 |
| **紀錄** | | |
| `logging.use_wandb` | false | 是否啟用 wandb |
| `logging.wandb_project` | sasrec | wandb 專案名稱 |
| `logging.checkpoint_dir` | checkpoints/ | 模型儲存位置 |
| `logging.log_dir` | runs/ | TensorBoard 紀錄位置 |

### 5-2. 參數調整範例

```bash
# 加大模型（適合資料量大的情況）
python scripts/run_experiment.py --dataset ml-1m \
    model.hidden_units=128 \
    model.num_blocks=4 \
    model.num_heads=2

# 防止過擬合（資料量小時）
python scripts/run_experiment.py --dataset beauty \
    model.dropout_rate=0.5 \
    train.weight_decay=0.01

# 調整學習率與訓練時間
python scripts/run_experiment.py --dataset ml-1m \
    train.lr=0.0005 \
    train.num_epochs=500 \
    train.patience=30

# 加長序列（適合互動歷史長的使用者）
python scripts/run_experiment.py --dataset ml-1m \
    data.maxlen=200

# 只看 NDCG@10 和 HR@10
python scripts/run_experiment.py --dataset ml-1m \
    "eval.k_values=[10]"
```

### 5-3. 各資料集建議參數

| 資料集 | hidden_units | num_blocks | maxlen | dropout_rate |
|---|---|---|---|---|
| Beauty（小） | 50 | 2 | 50 | 0.5 |
| Video Games（小） | 50 | 2 | 50 | 0.5 |
| Steam（中） | 128 | 2 | 50 | 0.2 |
| ML-1M（中大） | 128 | 2 | 200 | 0.2 |

---

## 6. 查看訓練結果

### 6-1. 模型 checkpoint

```
checkpoints/
└── ml-1m/
    ├── best.pt     ← 驗證集 NDCG@10 最高的模型
    └── latest.pt   ← 最後一個 epoch 的模型
```

### 6-2. TensorBoard（本機）

```bash
tensorboard --logdir runs/
# 開啟瀏覽器：http://localhost:6006
```

可以看到：
- `train/loss`：每個 epoch 的訓練損失曲線
- `val/NDCG@10`、`val/HR@10` 等：驗證集指標曲線

### 6-3. TensorBoard（Colab）

```python
%load_ext tensorboard
%tensorboard --logdir runs/
```

### 6-4. 載入模型做推論

```python
import torch
from sasrec.model.sasrec import SASRec
from sasrec.data.preprocessor import load_processed_data

# 載入資料（取得 item_count）
_, _, _, item_count = load_processed_data("data/processed/ml-1m")

# 建立模型（參數要與訓練時相同）
model = SASRec(
    item_num=item_count,
    hidden_units=50,
    maxlen=50,
    num_blocks=2,
    num_heads=1,
    dropout_rate=0.0,   # 推論時設 0
)

# 載入最佳 checkpoint
model.load_state_dict(torch.load("checkpoints/ml-1m/best.pt", map_location="cpu", weights_only=True))
model.eval()

# 對一個使用者做推論（輸入他的歷史序列）
seq = torch.tensor([[1193, 661, 914, 3408, 2355, 0, 0, 0, 0, 0]])  # 0 是 padding
with torch.no_grad():
    logits = model(seq)  # shape: [1, item_count]
    top10 = logits[0].topk(10).indices + 1  # +1 因為 item id 從 1 開始
    print(f"推薦前 10 名物品 ID：{top10.tolist()}")
```

### 6-5. 啟用 wandb

```bash
# 安裝
pip install -e ".[logging]"

# 訓練時啟用
python scripts/run_experiment.py --dataset ml-1m \
    logging.use_wandb=true \
    logging.wandb_project=my_sasrec_project
```

---

## 7. 在 Google Colab 使用

### 完整流程（從頭到尾）

```python
# Cell 1：安裝
!git clone https://github.com/hy1107/SASRec.git
%cd SASRec
!pip install -e .
```

```python
# Cell 2：下載 ml-1m 資料
!wget https://files.grouplens.org/datasets/movielens/ml-1m.zip -P data/raw/
```

```python
# Cell 3：預處理
!python scripts/preprocess.py --dataset ml-1m
```

```python
# Cell 4：查看資料統計
from sasrec.data.preprocessor import load_processed_data
import numpy as np

train_data, valid_data, test_data, item_count = load_processed_data("data/processed/ml-1m")
lengths = [len(seq) for seq in train_data.values()]
print(f"使用者數：{len(train_data)}, 物品數：{item_count}")
print(f"序列長度 — 最短：{min(lengths)}, 最長：{max(lengths)}, 平均：{np.mean(lengths):.1f}")
```

```python
# Cell 5：訓練（使用 GPU）
!python scripts/run_experiment.py --dataset ml-1m \
    model.hidden_units=128 \
    train.batch_size=256
```

```python
# Cell 6：查看 TensorBoard
%load_ext tensorboard
%tensorboard --logdir runs/
```

### 上傳自己的資料到 Colab

```python
from google.colab import files
uploaded = files.upload()  # 選擇你的 CSV 檔

!python scripts/preprocess.py \
    --input_path mydata.csv \
    --fmt csv \
    --output_dir data/processed/mydata

!python scripts/run_experiment.py --data_dir data/processed/mydata
```

---

## 8. 常見問題

**Q：訓練很慢怎麼辦？**

確認有使用 GPU：
```python
import torch
print(torch.cuda.is_available())  # 應該顯示 True
```
Colab：Runtime → Change runtime type → GPU

**Q：loss 不下降怎麼辦？**

嘗試調低學習率：
```bash
train.lr=0.0001
```

**Q：想要更快看到結果？**

縮短訓練時間：
```bash
train.num_epochs=50 train.patience=10
```

**Q：怎麼知道模型有沒有過擬合？**

看 TensorBoard：如果 `train/loss` 持續下降但 `val/NDCG@10` 停滯甚至下降，就是過擬合。解法：
```bash
model.dropout_rate=0.5
train.weight_decay=0.01
```

**Q：資料被過濾掉太多使用者？**

降低過濾門檻：
```bash
python scripts/preprocess.py --dataset beauty --min_interactions 3
```

**Q：序列很長但效果差？**

增加 maxlen：
```bash
data.maxlen=200
```
