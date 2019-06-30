# Feature process framework

Can be viewed as a simplified version of [TFX](https://www.tensorflow.org/tfx/),
with data validation/transform capabilities.

## 特征验证

## 特征转换

## TODO

- [ ] 如何高效收集统计数据？最好只扫一遍即可。还要考虑数据无法一次性加载到内存中的情况。
- [ ] 完善测试用例。
- [ ] 如何设计代码架构，比如统计好数据之后还要把这些统计结果传到`transformer`或者
  `validator`里，能不能更OO一点，设计一个更合理的类？
- [ ] C++ implementation
