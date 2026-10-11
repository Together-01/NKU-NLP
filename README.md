# NLP Homework 1

本项目完成 Task 1 要求的两种 Bag-of-Words 文本分类实验：

1. **Binary Bag of Words**：单词出现时特征值为 1，否则为 0。
2. **Word Frequency**：特征值为单词在当前文档中的出现次数。

两种表示使用完全相同的训练集词表和 Logistic Regression 分类器，并在 NYT
validation/test set 上计算 Accuracy 和 Macro-F1。最终应在实验报告中填写 test set
结果。

## 环境安装

推荐使用 Conda 创建独立环境：

```bash
conda env create -f environment.yml
conda activate nlp-hw1
```

当前 Conda 环境已经包含这些依赖时，可以直接跳过创建环境。也可以使用
`python3 -m pip install -r requirements.txt` 安装依赖。

## 运行

数据目录中已经包含固定的 80%/10%/10% 划分，直接运行：

```bash
python main.py
```

如需从 `data/nyt.csv` 重新进行分层随机划分：

```bash
python main.py --resplit
```

随机种子固定为 `67`。默认使用训练集中出现的全部 unigram 构建词表，不使用测试集
信息。若机器资源有限，可以通过 `--max-features` 限制词表大小，例如：

```bash
python main.py --max-features 50000
```

## 输出文件

- `outputs/task1_results.csv`：validation/test 的 Accuracy 和 Macro-F1。
- `outputs/task1_vocabulary.json`：仅根据训练集建立的词表。
- `outputs/*_predictions.csv`：每种方法在 validation/test set 上的预测结果。

报告测试集指标时，请使用 `outputs/task1_results.csv` 中 `split=test` 的两行。

## 本次运行结果

重新使用完整训练集词表（61,081 个 unigram）和 `C=1.0` 运行得到：

| Method | Test Accuracy | Test Macro-F1 |
| --- | ---: | ---: |
| Binary Bag of Words | 0.9826 | 0.9622 |
| Word Frequency | 0.9878 | 0.9704 |

## Task 2：Word Embedding

Task 2 包含三组 100 维词向量实验：

1. `glove.6B.100d` 预训练 GloVe。
2. 使用 `data/ag.csv` 训练 Word2Vec。
3. 只使用 `data/nyt_train.csv` 训练 Word2Vec，避免 validation/test 信息泄漏。

三种方法都对文档中已知单词的向量取平均，再使用 Logistic Regression 分类。运行
GloVe 实验前，从作业指定的 Stanford 地址下载并解压 100d 文件：

```bash
python download_glove.py
```

下载脚本使用 HTTP Range，只并行下载 ZIP 中的 100d 文件并执行大小和 CRC 校验，
不需要获取完整的 822 MB 压缩包。

`task2.py` 默认读取 `embeddings/glove.6B.100d.txt`，也可以通过 `--glove-path`
指定其他位置。

运行全部实验：

```bash
python task2.py
```

也可以单独运行某一组实验：

```bash
python task2.py --methods glove
python task2.py --methods ag_word2vec
python task2.py --methods nyt_word2vec
```

默认 Word2Vec 参数为 `vector_size=100`、`window=5`、`min_count=2`、`sg=1`、
`epochs=10`、`seed=67` 和 `workers=1`。增加 `--workers` 可以加快训练，但可能降低
逐次运行的完全可复现性。使用 `--retrain` 可以忽略已保存模型并重新训练。

Task 2 输出：

- `outputs/task2_results.csv`：三种方法的 validation/test 指标和词向量覆盖率。
- `outputs/task2_*_predictions.csv`：各方法的预测结果。
- `models/task2/*.model`：AG News 和 NYT 训练得到的 Word2Vec 模型。

### Task 2 本次运行结果

| Method | Test Accuracy | Test Macro-F1 | Test Token Coverage |
| --- | ---: | ---: | ---: |
| Pre-trained GloVe 6B 100d | 0.9792 | 0.9549 | 0.9977 |
| Word2Vec trained on AG News | 0.9714 | 0.9384 | 0.9730 |
| Word2Vec trained on NYT Train | 0.9757 | 0.9473 | 0.9925 |

## Task 3：BERT Fine-tuning

Task 3 使用 `google-bert/bert-base-uncased` 完成 NYT 三分类。实现严格使用
`max_length=64` 并训练 3 epochs，每轮在 validation set 上计算 Accuracy 和
Macro-F1。正式结果使用第 3 个 epoch 结束后的 final checkpoint；同时额外保存
validation Macro-F1 最优的 checkpoint 供分析，但不用于正式测试结果。

默认参数：

- Train batch size：32
- Gradient accumulation：1
- Learning rate：`2e-5`
- Weight decay：`0.01`
- Maximum sequence length：64
- Epochs：3
- Random seed：67

运行：

```bash
conda activate nlp-hw1
python task3.py
```

当前机器无 CUDA 时会自动使用 CPU。已存在 final 模型时，可以只重新评价：

```bash
python task3.py --eval-only
```

Task 3 输出：

- `outputs/task3_results.csv`：第 3 个 epoch 的 validation 和最终 test 指标。
- `outputs/task3_training_history.csv`：三个 epoch 的训练及验证指标。
- `outputs/task3_bert_test_predictions.csv`：测试集预测结果。
- `models/task3/bert_final/`：第 3 个 epoch 结束后的正式模型与 tokenizer。
- `models/task3/bert_best/`：validation Macro-F1 最优的补充模型与 tokenizer。

### Task 3 本次运行结果

| Checkpoint | Test Accuracy | Test Macro-F1 |
| --- | ---: | ---: |
| Epoch 3 final | 0.9809 | 0.9557 |

## 最终测试集汇总

所有方法使用相同的 NYT train/validation/test 划分：

| Task | Method | Test Accuracy | Test Macro-F1 |
| --- | --- | ---: | ---: |
| Task 1 | Binary Bag of Words | 0.9826 | 0.9622 |
| Task 1 | Word Frequency | 0.9878 | 0.9704 |
| Task 2 | Pre-trained GloVe 6B 100d | 0.9792 | 0.9549 |
| Task 2 | Word2Vec trained on AG News | 0.9714 | 0.9384 |
| Task 2 | Word2Vec trained on NYT Train | 0.9757 | 0.9473 |
| Task 3 | BERT-base-uncased, epoch 3 final | 0.9809 | 0.9557 |
