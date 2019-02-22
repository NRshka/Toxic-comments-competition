Multilabel классификация интернет комментариев по классам токсичности.
В pdf файле - обзорн хода работы. Упор делался на нейросетевые подходы.
Для работы необходимо следующее:
1. Python 3.*, ниже 3.7 - для поддержки TensorFlow
2. Tensorflow, keras
3. nltk
4. numpy, pandas, sklearn, re
5. Скачать и поместить FastText pretrained embedding с размерностью 300:
https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip

Для сравнения, прилагается ipyb notebook с решением той же задачи с помощью линейной регрессии.