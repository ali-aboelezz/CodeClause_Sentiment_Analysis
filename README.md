# Sentiment Analysis

### About Sentiment Analysis:
   Sentiment analysis is the process of detecting positive or negative sentiment in text. It’s often used by businesses to detect sentiment in social data, gauge brand reputation, and understand customers.

### We Will Use:  
- packed padded sequences
- pre-trained word embeddings
- bidirectional RNN
- multi-layer RNN
- regularization
- Adam optimizer
- LSTM

### About RNN:
We'll be using a recurrent neural network (RNN) as they are commonly used in analyzing sequences. An RNN takes in sequence of words,  𝑋={𝑥1,...,𝑥𝑇} , one at a time, and produces a hidden state,  ℎ , for each word. We use the RNN recurrently by feeding in the current word  𝑥𝑡  as well as the hidden state from the previous word,  ℎ𝑡−1 , to produce the next hidden state,  ℎ𝑡 .

ℎ𝑡=RNN(𝑥𝑡,ℎ𝑡−1)

Once we have our final hidden state,  ℎ𝑇 , (from feeding in the last word in the sequence,  𝑥𝑇 ) we feed it through a linear layer,  𝑓 , (also known as a fully connected layer), to receive our predicted sentiment,

 𝑦̂ =𝑓(ℎ𝑇) .

Below shows an example sentence, with the RNN predicting zero, which indicates a negative sentiment. The RNN is shown in orange and the linear layer shown in silver. Note that we use the same RNN for every word, i.e. it has the same parameters. The initial hidden state,  ℎ0 , is a tensor initialized to all zeros.



### Bidirectional RNN
The concept behind a bidirectional RNN is simple. As well as having an RNN processing the words in the sentence from the first to the last (a forward RNN), we have a second RNN processing the words in the sentence from the last to the first (a backward RNN). At time step  𝑡 , the forward RNN is processing word  𝑥𝑡 , and the backward RNN is processing word  𝑥𝑇−𝑡+1 .
In PyTorch, the hidden state (and cell state) tensors returned by the forward and backward RNNs are stacked on top of each other in a single tensor.
We make our sentiment prediction using a concatenation of the last hidden state from the forward RNN (obtained from final word of the sentence),  ℎ→𝑇 , and the last hidden state from the backward RNN (obtained from the first word of the sentence),  ℎ←𝑇 , i.e.  𝑦̂ =𝑓(ℎ→𝑇,ℎ←𝑇)

The image below shows a bi-directional RNN, with the forward RNN in orange, the backward RNN in green and the linear layer in silver.


## LSTM:
LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!.

## About Dataset
[LINK](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)
- Column 1 : polarity of tweet
  negative or positive
- Column 2 : id of the tweet
- Column 3 : date of the tweet
- Column 4 : query
- Column 5 : user
- Column 6 : text of the tweet



## Libraries Used

- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [scikit Learn](https://scikit-learn.org/stable/)
- [PyTorch](https://pytorch.org/)

## Conclusion
Using a twitter dataset, implemented a LSTM model, with a BRNN , and fit our model. We ended up obtaining an accuracy of **82%** in magnitude.
