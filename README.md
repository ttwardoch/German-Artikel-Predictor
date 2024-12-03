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

Final training logs:
~~~~
Epoch 1,  Train_loss: 0.4106, Test_loss: 0.1969, Accuracy: 93.2555
Epoch 2,  Train_loss: 0.1671, Test_loss: 0.1581, Accuracy: 94.8206
Epoch 3,  Train_loss: 0.1340, Test_loss: 0.1563, Accuracy: 94.8761
Epoch 4,  Train_loss: 0.1161, Test_loss: 0.1424, Accuracy: 95.4001
Epoch 5,  Train_loss: 0.1051, Test_loss: 0.1523, Accuracy: 95.0788
Epoch 6,  Train_loss: 0.0952, Test_loss: 0.1494, Accuracy: 95.5334
Epoch 7,  Train_loss: 0.0881, Test_loss: 0.1482, Accuracy: 95.4430
Epoch 8,  Train_loss: 0.0821, Test_loss: 0.1513, Accuracy: 95.4287
Epoch 9,  Train_loss: 0.0766, Test_loss: 0.1535, Accuracy: 95.4762
Epoch 10, Train_loss: 0.0714, Test_loss: 0.1595, Accuracy: 95.4730
~~~~
Best epoch: 6th