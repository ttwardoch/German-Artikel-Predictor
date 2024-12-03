# German Artikel Predictor

This project attempts to recognise which noun belongs
to which german gender (der, die, das) from the characters making up the word.
The input of the LSTM is a one-hot embedding of all characters making it a 2d tensor.
Output is a three-class softmax for the three genders. I implemented it using PyTorch. Experiments were tracked using tensorboard.

Files:
- LSTM_words.ipynb - jupyter notebook containing the hyperparameters calling other scripts for training and evaluation. Contains everything that is needed for experiments.
- LSTM_model.py - contains the LSTM model.
- Transformer_model.py - contains an unused Transformer model.
- data_model.py - contains dataset definition and function returning dataloaders.
- train_eval.py - contains training and evaluation functions
- utils.py - contains some utilities
- words_big.txt - dataset from https://github.com/aakhundov/deep-german/tree/master

The model achieved 95.53% accuracy on the test set.
The model is an LSTM with 1 layer, 256 hidden dimensions, 32 embedding dimensions, batch size of 64, and learning rate of 0.0032.
Its parameters are available under models/, and experiments under runs/.