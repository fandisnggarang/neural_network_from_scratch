import time
import numpy as np

from model import Neural_Network

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

def train(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    start_time = time.time()
    for epoch in range(epochs):
        permutation = np.random.permutation(x_train.shape[0])

        # put data train in batch 32 (for example)
        for batch_start in range(0, x_train.shape[0], batch_size):
            indices = permutation[batch_start:min(batch_start+batch_size, x_train.shape[0])]
            batch_x, batch_y = x_train[indices], y_train[indices]

            # forward
            frw_output = model.forward(batch_x)

            # backward
            _ = model.backward(frw_output, batch_y)

            # uncomment it below if clipping is used
            # model.apply_gradient_clipping(clip_value=0.005)

            # update
            model.optimize()

        # Evaluate performance
        # train
        train_out = model.forward(x_train, training=False, dropout=False)
        train_acc = model.accuracy(y_train, train_out)
        train_loss= model.cross_entropy_loss(train_out, y_train)

        # test
        test_out  = model.forward(x_test, training=False, dropout=False)
        test_acc  = model.accuracy(y_test, test_out)
        test_loss = model.cross_entropy_loss(test_out, y_test)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} - train acc: {train_acc:.4f}, train loss: {train_loss:.4f}, test acc: {test_acc:.4f}, test loss: {test_loss:.4f}, time: {epoch_time:.2f}s")
        
def main():
    # create and split data
    n_samples  = 10000 
    n_features = 10
    n_informative = 3
    n_classes  = 3
    n_cluster_per_class = 2
    random_state = 25

    x, y  = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, 
                                n_classes=n_classes, n_clusters_per_class=n_cluster_per_class, random_state = random_state)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=random_state)

    # preprocess the data
    x_train_scaled = preprocessing.normalize(x_train)
    x_test_scaled  = preprocessing.normalize(x_test)

    # apply one hot encoding to y data
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test  = encoder.transform(y_test.reshape(-1, 1))

    # initialize model
    model = Neural_Network(input_size=n_features, hidden_size=[128, 64, 32], output_size=n_classes, l_rate=0.1, activation='sigmoid', optimizer='sgd')

    # train and test the model
    train_model = train(model, x_train_scaled, y_train, x_test_scaled, y_test, epochs=150, batch_size=32)

if __name__ == '__main__': 
    main()
    
# the codes above was modified from https://github.com/lionelmessi6410/Neural-Networks-from-Scratch/blob/main/model.py