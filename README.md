# FastAI-Deeplearning-AiTAProject
*******************************************************************
*The application that will finally put an end to moral philosophy.*
******************************************************************

Just kidding, it's a text classification application using FastAI, trained on a database of all reddit "Am I The ***hole" posts. This is based on the same principles that are used for some sentiment analysis applications. This is a deep learning project that trains and deploys a neural net using every "Am I the ***hole" post and its verdict from Reddit.

It starts by employing transfer leaning by taking a pre-trained language model and fine tuning it with our new body of text:
![Imgur](https://i.imgur.com/Rbb1HHr.png)

After that is complete, we then train again with our labels so we can have a classifier:
![Imgur](https://i.imgur.com/Fu3zS7F.png)

Data set from this article:
https://dvc.org/blog/a-public-reddit-dataset

And used the Fast.ai library, which is a simplified overlay of Pytorch, Pandas, Numpy, SKLearn and other packages.

The Fast.ai quickstart guide can be found here. https://docs.fast.ai/text.html
They also offer a MOOC! And it is awesome.
