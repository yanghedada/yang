# yang
start tests
bearing_LSTM.py 是把美国西储大学轴承数据下载下来，并用EMD分解技术。通过EMD把轴承分解，提取前10的imf作为特征数据，数据格式为（2500，10）。
经过cnn卷积，再通过两个GRU1网路，之后拉平，全连接。

EMD_1.py 是介绍怎么用EMD分解技术的。具体看官方代码。
