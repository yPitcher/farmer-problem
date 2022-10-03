import numpy as np
import warnings as wr
from sklearn import neural_network as nn
from sklearn.exceptions import ConvergenceWarning

#Informacoes relevantes
#Alfabeto => {G,A,C}
#00 (G)
#11 (A)
#10 (C)
#Trabalhando com a sentenca: GAGGGCCGACGAG
#ordem: q0 -> q1, q2, q3, q2, q3, q4, q3, q2, q1, q6, q7, q8, q5 <= final

# taxa de aprendizado
lr = 0.03

#dataset de treino
#pos.automato q0 q1 q2 q3 q4 q5 q6 q7 q8  Simb.
train = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      #q0 --> 00 (G)
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],      #q1 --> 11 (A)
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],      #q2 --> 00 (G)
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],      #q3 --> 00 (G)
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],      #q2 --> 00 (G)
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],      #q3 --> 10 (C)
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],      #q4 --> 10 (C)
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],      #q3 --> 00 (G)
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],      #q2 --> 11 (A)
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],      #q1 --> 10 (C)
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],      #q6 --> 00 (G)
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],      #q7 --> 11 (A)
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])     #q8 --> 00 (G)

# rotulos do dataset de treino
# Para onde ele deve ir (transicao)
#pos.automato q0 q1 q2 q3 q4 q5 q6 q7 q8  Simb.
testrotule = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],      #q1
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],      #q2
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],      #q3
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],      #q2
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],      #q3
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],      #q4
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],      #q3
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],      #q2
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],      #q1
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],      #q6
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],      #q7
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],      #q8
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])     #q5 <= final

# dataset de testrotulees
#pos.automato q0 q1 q2 q3 q4 q5 q6 q7 q8  Simb.
test = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      #q0 --> 00 (G)
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],      #q1 --> 11 (A)
                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],      #q2 --> 00 (G)
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],      #q3 --> 00 (G)
                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],      #q2 --> 00 (G)
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],      #q3 --> 10 (C)
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],      #q4 --> 10 (C)
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],      #q3 --> 00 (G)
                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],      #q2 --> 11 (A)
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],      #q1 --> 10 (C)
                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],      #q6 --> 00 (G)
                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],      #q7 --> 11 (A)
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])     #q8 --> 00 (G)

mlp = nn.MLPClassifier(hidden_layer_sizes=(10,), max_iter=200, alpha=1e-4, solver='lbfgs', verbose=10, random_state=1, learning_rate_init=lr)

# treino
print('#################### EXECUCAO ####################')
print('Treinamento')
with wr.catch_warnings():
    wr.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(train, testrotule)

# testrotulee
print('Testes')
Y = mlp.predict(test)

# resultado
print('Resultado procurado')
print(testrotule)
print("Score de treino: %f" % mlp.score(train, testrotule))
print('Resultado encontrado')
print(Y)
print("Score do teste: %f" % mlp.score(test, testrotule))

sumY = [sum(Y[i]) for i in range(np.shape(Y)[0])] # saida
sumtestrotule = [sum(testrotule[i]) for i in range(np.shape(testrotule)[0])] # target