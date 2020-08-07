在Java代码中，为了分析PDF格式的论文文件，并调用机器学习模型对文章中的关键句子进行标签分类，我们调用了python代码打包生成的"paperAnalyze.exe"程序。本文件夹下的python代码文件为"paperAnalyze.exe"程序的源代码。

paperAnalyze.py为项目中进行论文内容分析的主程序入口，将指定路径下的PDF论文文件进行读取、分析操作
funcLib.py中定义了其它py代码文件中所用到的部分函数定义
predict.py、textrank.py中均为供 paperAnalyze.py 文件调用的模块代码文件