import numpy as np
import plotly.graph_objects as go

def tanh(x):
    ''' Função tangente hiperbólica '''
    return np.tanh(x)

def dtanh(x):
    ''' Derivada da função tangente hiperbólica '''
    return 1./np.cosh(x)**2

def meanSquareError(logits, y_real, derivate=False):
    if derivate:
        return 2*(y_real-logits)
    else:
        eqm = ((logits-y_real)**2).mean()
        if len(y_real.shape) == 1:
            return eqm
        elif len(y_real.shape) == 2 and y_real.shape[1] != 1:
            return eqm.mean()
        return eqm

class Jordan:
    ''' Jordan network '''

    def __init__(self, *args):
        ''' Criando rnn usando *args, lista de argumentos com a quantidade de neurônios. 
            A rnn Jordan envia seu fluxo de informação nas camadas de neurônios da seguinte forma: 
            1° entrada -> 2° escondida -> 3° saída -> 4° contexto -> 5° escondida -> 6° saída.
        '''

        self.shape = args
        n = len(args)

        # Criando as camadas e os bias 
        self.layers = []
        self.bias = []
        self.epoch = 0
        self.lossFunction = meanSquareError
        self.loss = []
        self.lossVal = []
        # Input layer (+1 unit for bias
        #              +size of oputput layer)
        self.layers.append(np.ones(self.shape[0]+1+self.shape[-1]))
        
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))
            self.bias.append(np.ones(self.shape[i]))

        # Criandos os pesos sinápticos (aleatóriamente entre -0.25 a +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))
            
        # dw Variável para armazenar o ultimo valor do peso sinaptico (for momentum)
        self.dw = [0,]*len(self.weights) # Vetor de zeros do mesmo tamanho dos pesos
        
        # Reset weights (Função para zerar os pesos sinápticos)
        self.reset() 

    def reset(self):
        ''' Reset weights (Função para zerar os pesos sinápticos, normalizando entre -0.25 a +0.25) '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25

    def forward(self, data, weights, bias):
        ''' Propaga os dados da camada de entrada para a camada de saída. '''

        # Inserindo os dados na camada de entrada
        self.layers[0][0:self.shape[0]] = data
        # Por ser uma rnn os dados também são colocados na camada de contexto 
        self.layers[0][self.shape[0]:-1] = self.layers[-1]
        
        for i in range(1,len(self.shape)):

            # Somatório vetorial entre os pesos sinápticos e os dados
            soma = np.dot(self.layers[i-1], weights[i-1])
            # Inserindo o bias e usando uma função de ativação
            self.layers[i][...] = tanh(soma + bias[i-1])

        # Return output
        return self.layers[-1]


    def backward(self, target, lrate, momentum):
        ''' Backpropagation comum. '''

        deltas = []

        # Calcula o erro da saída da rede em comparação com o sinal de referência
        error = target - self.layers[-1]
        delta = error*dtanh(self.layers[-1])
        deltas.append(delta)

        # Calcula o delta para as camadas escondidas
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dtanh(self.layers[i])
            deltas.insert(0,delta)
            
        # Atualizando os valores dos pesos sinápticos e do bias
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            dw0 = np.atleast_2d(self.dw[i])
            self.weights[i] += lrate*dw + momentum*(dw - dw0)
            bias = lrate*dw + momentum*(dw - dw0) 
            self.bias[i] += bias.mean(axis=0) 
            self.dw[i] = dw

    def validation(self, xVal, yVal):

        self.outVal = np.zeros(len(yVal))

        for n in range(len(xVal)):
                self.outVal[n] = network.forward(xVal[n], self.weights, self.bias)

        self.lossVal.append(self.lossFunction(self.outVal, yVal))    

        if self.lossVal[self.epoch] == min(self.lossVal):

            self.epochMin = self.epoch
            self.weightsMin = self.weights.copy()
            self.biasMin = self.bias.copy()
        

    def learn(self, xTrain, yTrain, xVal, yVal, precision=1e-8, epochMax=2000, lrate=0.1, momentum=0.1):
        '''[Função que executa treinamento]
        
        Função dedicada para realizar o treinamento
         com a validação, para obter os melhores 
         valores de pesos sinapticos.
        '''
        self.error = 1
        self.loss.append(1)
        self.lossVal.append(1)
        self.outTrain = np.zeros(len(yTrain))

        while self.error > precision:
            self.epoch +=1

            for n in range(len(xTrain)):
                self.outTrain[n] = self.forward(xTrain[n], self.weights, self.bias)
                self.backward(yTrain[n], lrate, momentum)

            self.loss.append(self.lossFunction(self.outTrain, yTrain))
            self.error = abs(self.loss[self.epoch]-self.loss[self.epoch-1])    

            self.validation(xVal, yVal)
            
            if self.epoch > epochMax:
                break

    def predict(self, data, target):
        self.out = np.zeros(len(target))

        for n in range(len(target)):
            self.out[n] = network.forward(data[n], self.weightsMin, self.biasMin)

        loss = self.lossFunction(self.out, target)  

        print('MSE Teste:  ' + str(loss))
        return self.out

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------------------
    network = Jordan(1,5,1)

    tam = 100
    x = np.linspace(-30,30,tam)
    
    y = tanh(np.cos(x*np.pi)**2 + np.cos(x*np.pi))
    x = x/30

    x_train, x_val, x_test = x[0:round(tam * 0.7)], x[round(tam * 0.7):round(tam * 0.85)], x[round(tam * 0.85)::]
    y_train, y_val, y_test = y[0:round(tam * 0.7)], y[round(tam * 0.7):round(tam * 0.85)], y[round(tam * 0.85)::]


    network.learn(x_train, y_train, x_val, y_val)
    yy = network.predict(x_test, y_test)

    #-----------------------------------------------------
    plt.figure(figsize=(10,5))

    plt.plot(x_train, y_train, color='b', label="train")
    plt.plot(x_val, y_val, color='r', label='val')

    plt.plot(x_test, y_test, color='g', label='test')
    plt.plot(x_test, yy, color='y', marker = ".", label='predict')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

    plt.show()
    #-----------------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=network.loss,
                                mode='lines',
                                name='MSE - Train: ',
                                line=dict(color='darkviolet', width=2.5, dash='dashdot')))

    fig.add_trace(go.Scatter(y=network.lossVal,
                                mode='lines+markers',
                                name='MSE - Val: ',
                                line=dict(color='firebrick', width=1.5, dash='dashdot')))

    fig.update_layout(title='MSE x Interação',
                       xaxis_title='Interação',
                       yaxis_title='MSE',
                       showlegend=True,
                       height=800,
                       xaxis_rangeslider_visible=False)

    fig.write_html('mse.html', auto_open=True)