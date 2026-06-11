# BertClassifier(Not Work)

## 数据来源
  - [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset/data)

## 数据预处理
  通过 `python data.py -f "/path/to/News_Category_Dataset_v3.json" -t "/root/models/bert-base-uncased"` 获得训练集和测试集, 其中 `-f` 指定到 News_Category_Dataset_v3.json 位置, `-t` 指定到hugging face bert-base-uncased目录
  
  该脚本内包含逻辑如下:

  - 从 `News_Category_Dataset_v3.json` 中拆分得到训练集和验证集(为简单起见, 也作为测试集使用), 过程中记录类别与索引映射, 得到类别权重(当前类/总数, 用于FocalLoss)
  - 原文->tokens使用了多进程处理 + huffing face BertTokenizer
  - 实现了EDA, 包含回译, 移除和交换, 在 `NewsCategoryDataset` 可选择是否使用EDA生成训练集(测试集一定不使用EDA)
      - `_worker_func` 为包含EDA; `_worker_without_eda` 为不包含EDA
  - 脚本执行后将生成 `train.pt` , `test.pt` 和 `class_weights.pt`
      - `train.pt` 和 `test.pt` 使用时可以通过在外部导入 `data.LoadDataset` 读取
      - `class_weights.pt` 可直接通过 `torch.load()`读取

## Net
  主要为BERT+MLP进行微调, Net中设计了空间增强的方法 `text_mixup`
      - 实际训练中没有使用 `text_mixup` , 如果需要使用该方法需要修改 `train._FineTuningBertBase.train_epochs` 在计算得到 pred 时传入 labels

## 执行训练
  - 使用 `FocalLoss` 作为损失函数; `Adam` 作为优化器且 BERT 和 MLP 使用不同学习率; 代码支持学习率调度器但实际没有使用
  - 对 MLP 部分进行了 kaiming 初始化; 训练中加入了梯度剪裁
  - 支持 `DataParallel` 多GPU训练(但当前学习率和batchsize是基于单卡设置的)
  - 当前训练中使用的是加权重采样方式, `num_samples` 为2倍的训练数据量
  - `_CosineScheduler` 为自定义的余弦调度器
  - 训练完毕后会讲模型权重自动保存在 `fine-tuning-bert.pth`

## 其余部分
  - 由于数据集存在长尾问题, 因此通过F1判断模型质量更合适, 代码实现于 `utils.f1_report`
  - 模型可通过ONNX导出

## 后续可做TODO
    - 当前EDA实现是将所有样本都按照一定概率进行增强处理，后续可尝试只将少数类进行EDA
    - 对类别可进行优化. 方案一: 合并小类后训练, 方案二: 按照类别数拆成 多数类:中间类:少数类 训练三个模型加一个路由模型