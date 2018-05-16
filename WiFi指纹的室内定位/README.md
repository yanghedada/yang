数据在https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc.zip 

wifi_indoor 与wifi_autoencoder 是两种不同的方案。


1 。  wifi_indoor之后利用全连接nn ，进行预测结果预测。



2.wifi_autoencoder利用自动编码机进行编码，先进性无监督训练。

再把encode_x送入softmax进行判别