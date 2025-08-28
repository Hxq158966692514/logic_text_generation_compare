#-*- codeing = utf-8 -*-
#@Time : 2025-08-24 9:14
#@Author : 韩笑奇
#@File : download_dataset.py
#@Software: PyCharm


# 下载firefly数据


# from datasets import load_dataset
#
# ds = load_dataset('YeungNLP/firefly-train-1.1M')
#
# ds.save_to_disk('data')


# 下载limo最新款数据

from datasets import load_dataset

ds = load_dataset('GAIR/LIMO-v2')

ds.save_to_disk('less is more for reasoning')