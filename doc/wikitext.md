## 数据集简介

WikiText 语言建模数据集是从 Wikipedia 上经过验证的优秀和精选文章集中提取的超过 1 亿条数据的集合。

与 Penn Treebank (PTB) 的预处理版本相比，WikiText-2 大 2 倍以上，WikiText-103 大 110 倍以上。WikiText 数据集具有更大的词汇量，并保留了在 PTB 中被删除的原始大小写、标点符号和数字等内容。由于它由完整的文章组成，因此该数据集适合可以使用长文本上下文特征进行训练的模型。

本数据集合来自于[huggingface的数据集](https://huggingface.co/datasets/wikitext)，迁移目的是为了方便用户使用。

## 数据集加载方式
```python
from modelscope.msdatasets import MsDataset
# Load the wikitext dataset
train_datasets = MsDataset.load('wikitext', subset_name='wikitext-2-v1', split='train')
eval_datasets = MsDataset.load('wikitext', subset_name='wikitext-2-v1', split='validation')
```

## 数据及结构

### 数据集列表

#### wikitext-103-raw-v1

- **下载数据量:** 183.09 MB
- **最终数据集大小:** 523.97 MB
- **占用硬盘容量:** 707.06 MB

`validation`数据集的样例如下（样例因长度过长而经过裁剪）：
```
{
    "text": "\" The gold dollar or gold one @-@ dollar piece was a coin struck as a regular issue by the United States Bureau of the Mint from..."
}
```

#### wikitext-103-v1

- **下载数据量:** 181.42 MB
- **最终数据集大小:** 522.66 MB
- **占用硬盘容量:** 704.07 MB

`train`数据集的样例如下（样例因长度过长而经过裁剪）：
```
{
    "text": "\" Senjō no Valkyria 3 : <unk> Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to..."
}
```

#### wikitext-2-raw-v1

- **下载数据量:** 4.50 MB
- **最终数据集大小:** 12.91 MB
- **占用硬盘容量:** 17.41 MB

`train`数据集的样例如下（样例因长度过长而经过裁剪）：
```
{
    "text": "\" The Sinclair Scientific Programmable was introduced in 1975 , with the same case as the Sinclair Oxford . It was larger than t..."
}
```

#### wikitext-2-v1

- **下载数据量:** 4.27 MB
- **最终数据集大小:** 12.72 MB
- **占用硬盘容量:** 16.99 MB

`train`数据集的样例如下：
```
{
    "text": "\" Senjō no Valkyria 3 : <unk> Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to..."
}
```

### 数据字段

**注意：** 以下的数据字段在每个数据集的各数据文件（train、validation等）都是相同的。

#### wikitext-103-raw-v1
- `text`: `string`类型的文本

#### wikitext-103-v1
- `text`: `string`类型的文本

#### wikitext-2-raw-v1
- `text`: `string`类型的文本

#### wikitext-2-v1
 `text`: `string`类型的文本

### 数据量

|name               |  train|validation| test|
|-------------------|------:|---------:|----:|
|wikitext-103-raw-v1|1801350|      3760| 4358|
|wikitext-103-v1    |1801350|      3760| 4358|
|wikitext-2-raw-v1  |  36718|      3760| 4358|
|wikitext-2-v1      |  36718|      3760| 4358|
