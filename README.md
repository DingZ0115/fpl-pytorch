# fpl-pytorch

batch - (11,batch_size)
`batch`实际上是一个包含两个或多个元素的元组（或者是列表），而不是直接包含数据项的单一对象。这种情况通常发生在当你的`Dataset`对象返回的每个项是一个元组，比如`(data, label)`，而`DataLoader`将这些项集合成批次时。这时，`len(batch)`实际上给出的是元组中元素的数量，通常对应于数据和标签（如果有其他元素，数字可能更大）。
在这种情况下，`batch[0]`可能是所有数据项的集合，而`batch[1]`是所有标签的集合。因此，`len(batch[0])`实际上是批量大小，也就是每个批次中数据项的数量，这与你设置的`batch_size`相匹配。
举个例子，如果你的数据集返回的每个项是一个形如`(image, label)`的元组，那么经过`DataLoader`处理后，每个`batch`将是形如`([images], [labels])`的结构，其中`[images]`和`[labels]`分别是图片和标签的列表（或张量）。在这种情况下：
- `len(batch)`将返回2（因为`batch`包含两个元素：数据和标签）。
- `len(batch[0])`将返回批次中的数据项数量，即`batch_size`。
这解释了为什么你会观察到`len(batch[0]) = batch_size`，而不是`len(batch) = batch_size`的情况。