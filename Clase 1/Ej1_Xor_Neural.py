import numpy as np
from matplotlib import pyplot as plt

def Sigmoid(a):
    return 1 / ( 1 + np.exp(-a))


if __name__ == '__main__':

    X = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])

    Y = np.array([0,1,1,0])

    l1_input = 2
    l1_cant_neu = 2
    W1 = np.random.rand(l1_input,l1_cant_neu)
    # W1 -> [2x2]
    b1 = np.random.rand(l1_input,1)
    # b1 -> [2x1]
    l2_input = 2
    l2_cant_neu = 1
    W2 = np.random.rand(l2_input,l2_cant_neu)
    # W2 -> [2x1]
    b2 = np.random.rand(l2_cant_neu,1)
    # b2 -> [1x1]

    # SGD
    epochs = 3000
    epoch_error = []
    lr = 0.1
    y_hat = np.array([0.0,0.0,0.0,0.0])

    for i in range(epochs):
        for j in range(X.shape[0]):

            # forward propagation
            Xi = X[j,:].reshape(-1,1)
            # Xi ->[2x1]
            Z1 = W1 @ Xi + b1
            # Z1 -> [2x1] / W1 -> [2x2] / X[j] -> [2x1]  / b1 -> [2x1]
            A1 = Sigmoid(Z1)
            # A1 -> [2x1]
            Z2 = W2.T @ A1 + b2
            # Z2 -> [1x2] / W2.T -> [1x2] / A1-> [2x1]  / b1 -> [1x1]
            A2 = Sigmoid(Z2)


            y_hat[j]=A2

            error = Y[j] - A2
            #print(error)

            # backward propagation
            grad_W2 = -2 * error * A1
            # grad_W2 -> [2x1]
            grad_b2 = -2 * error

            # grad_b2 -> [1x1]
            dZ1 = (-2 * error * W2) * Sigmoid(Z1) * (1 - Sigmoid(Z1))
            # dZ1 -> [2x1]
            grad_W1 = dZ1 @ Xi.T
            # grad_w1 -> [2x2]
            grad_b1 = np.sum(dZ1 , axis=1 , keepdims=True)
            # grad_b1 ->[2x1]
            # Updates
            W2 = W2 - lr * grad_W2
            W1 = W1 - lr * grad_W1
            b2 = b2 - lr * grad_b2
            b1 = b1 - lr * grad_b1

        epoch_error.append((np.square(Y - y_hat)).mean())


    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(epoch_error, label="MSE/Epoch (lr:{}) ".format(lr))
    ax.legend()
    plt.title('Grafica de MSE de la XOR ')
    plt.show()


    # Grafica de barras de la salida real y la salida obtenida
    salida = ['Y1', 'Y2', 'Y3', 'Y4']
    # Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(salida))
    # tamaño de cada barra
    width = 0.25
    fig, ax = plt.subplots()
    # Generamos las barras para el conjunto de hombres
    rects1 = ax.bar(x - width / 2, Y, width, label='Y (Real)')
    # Generamos las barras para el conjunto de mujeres
    rects2 = ax.bar(x + width / 2, y_hat, width, label='Y (hat)')
    # Añadimos las etiquetas de identificacion de valores en el grafico
    ax.set_title('Salida Real vs Salida aprendida')
    ax.set_xticks(x)
    ax.set_xticklabels(salida)
    # Añadimos un legen() esto permite mmostrar con colores a que pertence cada valor.
    ax.legend()
    def autolabel(rects):
        """Funcion para agregar una etiqueta con el valor en cada barra"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    # Añadimos las etiquetas para cada barra
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    # Mostramos la grafica con el metodo show()
    plt.show()