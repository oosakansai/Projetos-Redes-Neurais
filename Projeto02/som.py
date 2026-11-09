import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from time import perf_counter

# leitura da imagem e pré-processamento dos dados ----------------------------------------------------------------

# leitura da imagem. Ela é um ndarray de 3 dimensões [altura, largura, [R, G, B]]
imagem = plt.imread('imagens/ilustracao_gradiente_abstrato.jpg')
# imagem alternativa: imagem = plt.imread('imagens/foto_flores_arco_iris.jpgg')

linhas_imagem = imagem.shape[0]
colunas_imagem = imagem.shape[1]

pixels = np.reshape(imagem, (linhas_imagem*colunas_imagem, 3))

#normalização do RGB de cada pixel
scaler = MinMaxScaler()
pixels_normalizados = scaler.fit_transform(pixels)

grids = [4, 8, 16]
erros_topologicos = []
erros_quantizacao = []
tempo = []

# inicialização e treinamento do SOM ----------------------------------------------------------------

for grid in grids:
  som = MiniSom(grid, grid, 3, sigma=0.5, learning_rate=0.05, neighborhood_function='triangle', random_seed=42,
          activation_distance='euclidean',
          topology='rectangular',
          sigma_decay_function = 'linear_decay_to_one',
          decay_function = 'linear_decay_to_zero',)

  #Neuronios iniciais representam cores aleatórias da imagem
  som.random_weights_init(pixels_normalizados)

  comeco = perf_counter()
  som.train_random(pixels_normalizados, 50000,verbose=True)
  fim = perf_counter() - comeco
  tempo.append(fim)

  erros_quantizacao.append(som.quantization_error(pixels_normalizados))
  erros_topologicos.append(som.topographic_error(pixels_normalizados))

print("duracoes: ", tempo)
print("erros_quantizacao: ", erros_quantizacao)
print("erros_topologicos: ", erros_topologicos)

# graficos de desempenho ----------------------------------------------------------------

# grafico tempo de treinamento
xpoints = np.array(grids) ** 2
ypoints = np.array(tempo)
plt.plot(xpoints, ypoints, marker = 'o', color='red')
plt.title("tempo de treinamento por n de neurônios")
plt.xlabel("neurônios")
plt.ylabel("tempo (s)")
plt.grid()
plt.show()

# grafico erro de quantização
ypoints = np.array(erros_quantizacao)
plt.plot(xpoints, ypoints, marker = 'o', color='red')
plt.title("erro de quantização por n de neurônios")
plt.xlabel("neurônios")
plt.ylabel("erro de quantização")
plt.grid()
plt.show()

# grafico erro topologico
ypoints = np.array(erros_topologicos)
plt.plot(xpoints, ypoints, marker = 'o', color='red')
plt.title("erro topologico por n de neurônios")
plt.xlabel("neurônios")
plt.ylabel("erro topológico")
plt.grid()
plt.show()

# quantização e a formação da imagem final ----------------------------------------------------------------

pixels_finais = som.quantization(pixels_normalizados) #quantização para associar cada pixel a uma das cores finais
imagem_final = np.zeros(imagem.shape)

for i, array_rgb in enumerate(pixels_finais): # inserir valores quantizados
    imagem_final[np.unravel_index(i, shape=(linhas_imagem, colunas_imagem))] = array_rgb


# plotagem imagem final e cores finais ----------------------------------------------------------------

plt.figure(figsize=(6, 6))

plt.subplot(211)
plt.title('imagem final')
plt.imshow(imagem_final)
plt.xticks([])
plt.yticks([])

plt.subplot(212)
plt.title('cores finais')

cores = som.get_weights().reshape(-1, 3)
paleta = np.tile(cores[np.newaxis, :, :], (100, 1, 1))
plt.imshow(paleta, interpolation='nearest', aspect='auto')

plt.xticks(range(0, len(cores), 2))
plt.yticks([])

plt.tight_layout()
plt.show()