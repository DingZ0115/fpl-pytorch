# fpl-pytorch

Q. DataSet返回的horizontal_flip，在dataloader产生batch以后，从布尔型变成了tensor
在PyTorch中，`DataLoader`的`collate_fn`参数控制如何将多个数据样本（来自`Dataset`的`__getitem__`
方法的返回值）批处理（或“collate”）成一个批次。默认情况下，`DataLoader`使用一个默认的`collate_fn`
，这个函数会尝试将列表中的数据合并成批次。对于张量数据，它会将它们堆叠起来；对于序列数据（如列表），它会将它们合并；对于数值和布尔值，它会尝试将它们转换为张量。

因此，当你的`Dataset`的`__getitem__`方法返回包含布尔值的数据时，`DataLoader`通过其默认的`collate_fn`
将这些布尔值转换为了PyTorch布尔张量，即使你在`__getitem__`方法中没有显式进行这种转换。这是为了保持数据类型的一致性，便于在PyTorch中进行批量处理和运算。

Q. batch的形状
pytorch中
batch - (11,batch_size)
`batch`实际上是一个包含两个或多个元素的元组（或者是列表），而不是直接包含数据项的单一对象。这种情况通常发生在当你的`Dataset`
对象返回的每个项是一个元组，比如`(data, label)`，而`DataLoader`将这些项集合成批次时。这时，`len(batch)`
实际上给出的是元组中元素的数量，通常对应于数据和标签（如果有其他元素，数字可能更大）。
在这种情况下，`batch[0]`可能是所有数据项的集合，而`batch[1]`是所有标签的集合。因此，`len(batch[0])`
实际上是批量大小，也就是每个批次中数据项的数量，这与你设置的`batch_size`相匹配。
举个例子，如果你的数据集返回的每个项是一个形如`(image, label)`的元组，那么经过`DataLoader`处理后，每个`batch`
将是形如`([images], [labels])`的结构，其中`[images]`和`[labels]`分别是图片和标签的列表（或张量）。在这种情况下：

- `len(batch)`将返回2（因为`batch`包含两个元素：数据和标签）。
- `len(batch[0])`将返回批次中的数据项数量，即`batch_size`。
  这解释了为什么你会观察到`len(batch[0]) = batch_size`，而不是`len(batch) = batch_size`的情况。

Chainer 中
batch - (batch_size, 11) 每个样本包括11个特征

## 待办

1. dataset - SceneDatasetForAnalysis
2. cnn- CNN_Ego，CNN_Pose



