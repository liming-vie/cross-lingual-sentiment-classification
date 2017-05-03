# Cross-Lingual Sentiment Classification
Implementation of [Co-Training for Cross-Lingual Sentiment Classification](http://dl.acm.org/citation.cfm?id=1687913)

## Prerequisite
* Python 2.7
* tqdm, prograss bar
* mtranslate, Python API for Google translate
* pyltp, Python extension for LTP
* scikit-learn, Python tool for mining and data analysis
* scipy, Python-based ecosystem of open-source software for mathematics, science, and engineering

## Result
Use accuracy to measure the performance of different domains.


|p, n, iter| DVD | Music |Book | Total |
| :--: | :--: | :--: | :--: | :--: |
|5, 5, 80|0.8055|0.74825|0.816|0.78992|
|10, 10, 80|0.817|0.76775|0.81575|0.80017|
|10, 10, 120|0.8205|0.77725|0.819|0.80558|
|15, 15, 120|**0.8235**|**0.7845**|**0.82825**|**0.81208**|
