import numpy as np
from matplotlib import pyplot as plt


def build_dataset(path):
    structure = [('X_1',np.float32),
                  ('X_2',np.float32),
                  ( 'Y' ,np.float32)]

    with open(path,encoding='utf8') as data_csv:
        data_gen = (( float(line.split(',')[0]) , float(line.split(',')[1]),
                     float(line.split(',')[2]) )for i, line in enumerate(data_csv) if i !=0)
        data = np.fromiter(data_gen,structure)

    X1 = np.array((data['X_1'])).reshape(-1,1)
    X2 = np.array((data['X_2'])).reshape(-1,1)
    X = np.concatenate((X1,X2),axis=1)
    Y = np.array(data['Y']).reshape(-1,1)

    return X,Y



def Sigmoid(a):
    return 1 / ( 1 + np.exp(-a))



if __name__ == '__main__':

    X,Y = build_dataset('clase_2_train_data.csv')
    # X -> [900x2]
    # Y -> [900x1]

    color_scatter = ['green' if i==1 else 'red' for i in Y[:,0]]
    plt.scatter(X[:,0],X[:,1],color=color_scatter)
    plt.title('Dataset de Entrenamiento')
    plt.show()

    # Mini Batch
    epochs = 60
    batch_size = 20
    epoch_error = []
    epoch_error_test = []
    n = X.shape[0]
    # learning rate
    alfa = 0.01

    # RNN init
    # layer 1 [3N] / layer 2 [2N] / layer 3 (out) [1N]
    l1_input = 2
    l1_cant_neu = 3
    W1 = np.random.rand(l1_input,l1_cant_neu)
    # W1 -> [2x3]
    b1 = np.random.rand(l1_cant_neu,1)
    # b1 -> [3x1]
    l2_input = 3
    l2_cant_neu = 2
    W2 = np.random.rand(l2_input,l2_cant_neu)
    # W2 -> [3x2]
    b2 = np.random.rand(l2_cant_neu,1)
    # b2 -> [2x1]
    l3_input = 2
    l3_cant_neu = 1
    W3 = np.random.rand(l3_input,l3_cant_neu)
    # W3 -> [2x1]
    b3 = np.random.rand(l3_cant_neu,1)
    # b3 -> [1x1]


    for i in range(epochs):
        error = 0
        for j in range(0, n , batch_size):
            X_b = X[j:j+batch_size,:]
            Y_b = Y[j:j+batch_size,:]
            # X_b -> [batch x 2]
            # Y_b -> [batch x 1]


            # forward propagation
            Z1 = W1.T @ X_b.T + b1
            # Z1 -> [3 x batch] / W1.T -> [3x2] / X_b -> [2 x batch] / b1 -> [3x1]
            A1 = Sigmoid(Z1)
            # A1 -> [3 x batch]
            Z2 = W2.T @ A1 + b2
            # Z2 -> [2x batch] / W2.T -> [2x3] / A1 -> [3x batch] / b2 -> [2x1]
            A2 = Sigmoid(Z2)
            # A2 -> [2x batch]
            Z3 = W3.T @ A2 + b3
            # Z3 -> [1x batch] / W3.T -> [1x2] / A2 -> [2x batch] / b3 -> [1x1]
            y_hat = Sigmoid(Z3)
            # y_hat -> [1x batch]

            # error
            J = np.sum(np.power(Y_b.T - y_hat,2))
            error += J

            # backward propagation
            dZ3 = - 2 * (Y_b.T - y_hat) * Sigmoid(Z3) *(1 - Sigmoid(Z3))
            # dZ3 -> [1x batch]
            grad_W3 = (1/batch_size) * dZ3 @ A2.T
            # grad_W3 -> [1x2] / dZ3 -> [1x batch] / A2.T -> [batch x 2]
            grad_b3 = (1/batch_size) * np.sum(dZ3 , axis=1 , keepdims=True)
            # grad_b3 -> [1x1]
            dZ2 = ( W3 @ dZ3 ) * Sigmoid(Z2)*(1-Sigmoid(Z2))
            # dZ2 -> [2x batch] / W3 -> [2x1] / dZ3 -> [1x batch]
            grad_W2 = (1/batch_size) * dZ2 @ A1.T
            # grad_W2 -> [2x3] / dZ2 -> [2x batch] / A1.T -> [batch x 3]
            grad_b2 = (1/batch_size) * np.sum(dZ2 , axis=1 , keepdims=True)
            # grad_b2 -> [2x1]
            dZ1 = ( W2 @ dZ2 ) * Sigmoid(Z1)*(1-Sigmoid(Z1))
            # dZ1 -> [3x batch] / W2 -> [3x2] / dZ2 -> [2x batch]
            grad_W1 = (1/batch_size) * dZ1 @ X_b
            # grad_W1 -> [3x2] / dZ1 -> [3x batch] / X_b -> [batch x 2]
            grad_b1 = (1/batch_size) * np.sum(dZ1 , axis=1 , keepdims=True)
            # grad_b1 -> [3x1]

            # Updates
            W3 = W3 - alfa * grad_W3.T
            # W3 -> [2x1] / grad_w3.T ->[2x1]
            b3 = b3 - alfa * grad_b3
            # b3 -> [1x1] / grad_b3 ->[1x1]
            W2 = W2 - alfa * grad_W2.T
            # W2 -> [3x2] / grad_w2.T ->[3x2]
            b2 = b2 - alfa * grad_b2
            # b2 -> [2x1] / grad_b2 ->[2x1]
            W1 = W1 - alfa * grad_W1.T
            # W1 -> [2x3] / grad_w1.T ->[2x3]
            b1 = b1 - alfa * grad_b1
            # b1 -> [3x1] / grad_b1 ->[3x1]

        epoch_error.append(error/X.shape[0])

    mse = np.mean(epoch_error)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(epoch_error, label= "MSE/Epoch (lr:{}) = {:.2f}".format(alfa,mse))
    ax.legend()
    plt.title('MSE Train dataset')
    plt.show()

    # Dataset Test

    X_t, Y_t = build_dataset('clase_2_test_data.csv')
    # X_t -> [100x2]
    # Y_t -> [100x1]

    for i in range(epochs):
        error = 0
        for j in range(0, n , batch_size):
            X_b = X_t[j:j+batch_size,:]
            Y_b = Y_t[j:j+batch_size,:]
            # X_b -> [batch x 2]
            # Y_b -> [batch x 1]


            # forward propagation
            Z1 = W1.T @ X_b.T + b1
            # Z1 -> [3 x batch] / W1.T -> [3x2] / X_b -> [2 x batch] / b1 -> [3x1]
            A1 = Sigmoid(Z1)
            # A1 -> [3 x batch]
            Z2 = W2.T @ A1 + b2
            # Z2 -> [2x batch] / W2.T -> [2x3] / A1 -> [3x batch] / b2 -> [2x1]
            A2 = Sigmoid(Z2)
            # A2 -> [2x batch]
            Z3 = W3.T @ A2 + b3
            # Z3 -> [1x batch] / W3.T -> [1x2] / A2 -> [2x batch] / b3 -> [1x1]
            y_hat = Sigmoid(Z3)
            # y_hat -> [1x batch]

            # error
            J = np.sum(np.power(Y_b.T - y_hat,2))
            error += J
        epoch_error_test.append(error / X.shape[0])

    mse_t = np.mean(epoch_error_test)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(epoch_error_test,color='r', label="MSE/Epoch (lr:{}) = {:.2f}".format(alfa, mse_t))
    ax.legend()
    plt.title('MSE Test dataset')
    plt.show()