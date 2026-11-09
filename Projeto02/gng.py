#pip install dyconnmap

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dyconnmap.cluster import GrowingNeuralGas
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

# treinamento do GNG -------------------------------------------------------------------------------------

SEED = 72
rng = np.random.RandomState(SEED)

neuronios = [16, 64, 256]
duracoes = []
erros_quantizacao = []
erros_topologicos = []

for n in neuronios:
    gng = GrowingNeuralGas(
        n_max_protos=n,
        iterations=50000,
        l=200,
        a_max=100,
        a=0.3,
        b=0.001,
        lrate=None,
        rng=rng
    )

    comeco = perf_counter()
    gng.fit(pixels_normalizados)
    duracao = perf_counter() - comeco

    duracoes.append(duracao)

    # Erro de quantização
    pixels_quantizados, symbols = gng.encode(
        pixels_normalizados
    )
    pixels_quantizados = np.asarray(
        pixels_quantizados
    ).squeeze()

    erro_q = np.mean(
        np.linalg.norm(
            pixels_normalizados - pixels_quantizados,
            axis=1
        )
    )
    erros_quantizacao.append(erro_q)

    # Erro topológico
    grafo = gng._GrowingNeuralGas__g
    protos = np.asarray(gng.protos)

    distancias = np.linalg.norm(
        pixels_normalizados[:, None, :] - protos[None, :, :],
        axis=2
    )

    dois_mais_proximos = np.argsort(
        distancias, axis=1
    )[:, :2]

    erros = 0

    for primeiro, segundo in dois_mais_proximos:
        if not grafo.has_edge(primeiro, segundo):
            erros += 1

    erro_t = erros / len(pixels_normalizados)
    erros_topologicos.append(erro_t)

print("duracoes: ", duracoes)
print("erros_quantizacao: ", erros_quantizacao)
print("erros_topologicos: ", erros_topologicos)

# graficos de desempenho ----------------------------------------------------------------

# grafico tempo de treinamento
xpoints = np.array(neuronios)
ypoints = np.array(duracoes)
plt.plot(xpoints, ypoints, marker = 'o')
plt.title("tempo de treinamento por n de neurônios")
plt.xlabel("neurônios")
plt.ylabel("tempo (s)")
plt.grid()
plt.show()

# grafico erro de quantização
ypoints = np.array(erros_quantizacao)
plt.plot(xpoints, ypoints, marker = 'o')
plt.title("erro de quantização por n de neurônios")
plt.xlabel("neurônios")
plt.ylabel("erro de quantização")
plt.grid()
plt.show()

# grafico erro topologico (apenas se a lista não estiver vazia)
ypoints = np.array(erros_topologicos)
plt.plot(xpoints, ypoints, marker = 'o')
plt.title("erro topologico por n de neurônios")
plt.xlabel("neurônios")
plt.ylabel("erro topológico")
plt.grid()
plt.show()

# quantização e a formação da imagem final ----------------------------------------------------------------

pixels_quantizados, symbols = gng.encode(pixels_normalizados)
pixels_quantizados = np.asarray(pixels_quantizados).squeeze()
pixels_quantizados = scaler.inverse_transform(pixels_quantizados)
clustered = pixels_quantizados.reshape(imagem.shape)

cores_gng = scaler.inverse_transform(gng.protos)

# plotagem imagem final e cores finais ----------------------------------------------------------------

plt.figure(figsize=(6, 6))

plt.subplot(211)
plt.title('imagem final')
plt.imshow(clustered.astype('uint8'))
plt.xticks([])
plt.yticks([])

plt.subplot(212)
plt.title('cores finais')

cores_img = np.repeat(cores_gng.astype('uint8')[np.newaxis, :, :], 80, axis=0)
plt.imshow(cores_img, aspect='auto')
n_cores = len(cores_gng)
plt.xticks(range(0, n_cores, 2))   
plt.yticks([])
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.5, box.height]) 

plt.tight_layout()
plt.show()