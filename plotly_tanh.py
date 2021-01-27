import plotly.graph_objects as go
import numpy as np


def tanh(x, b=1):
    ''' Função tangente hiperbólica '''
    return np.tanh(b*x/10)

def dtanh(x, b = 1):
    ''' Derivada da função tangente hiperbólica '''
    return 1./(np.cosh(b*x/10)**2)
	
x=np.linspace(-5,5,100)

fig = go.Figure()
for n in range(1,10):    
    fig.add_trace(go.Scatter(x=x, y=tanh(x, n),
                        mode='lines+markers',
                        name='tanh b: '+str(n)))
    fig.add_trace(go.Scatter(x=x, y=dtanh(x, n),
                        mode='lines+markers',
                        name='dtanh b: '+str(n)))
    fig.update_layout(title='MSE x Interação',
                       xaxis_title='Interação',
                       yaxis_title='MSE')

fig.write_html('tanh_dtanh.html', auto_open=True)


