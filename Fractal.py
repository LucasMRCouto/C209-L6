import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Carregar o arquivo de áudio
fs, audio = wavfile.read('beemo.wav')  # fs é a taxa de amostragem, audio é o array numpy com o áudio

# Normalizar o áudio se ele não estiver no formato de ponto flutuante
if audio.dtype != np.float32:  # Verificar se 'audio' não é do tipo float32
    audio = audio / np.max(np.abs(audio))  # Normalizar os dados de áudio para o intervalo [-1, 1]

# Aplicar um filtro passa-baixa
nyquist = 0.5 * fs  # Frequência de Nyquist
low_cutoff = 500 / nyquist  # Frequência de corte normalizada pela frequência de Nyquist
b, a = signal.butter(2, low_cutoff, btype='low')  # Coeficientes do filtro Butterworth
filtered_audio = signal.lfilter(b, a, audio)  # Aplicar o filtro ao áudio

# Equalizador paramétrico: Boost em 1kHz com largura de banda de 2 (Q)
f0 = 1000  # Frequência central do boost
Q = 2  # Largura de banda
w0 = f0 / (fs / 2)  # Frequência normalizada
bw = w0 / Q  # Largura de banda normalizada
b, a = signal.iirpeak(w0, bw)  # Coeficientes do filtro IIR com pico
equalized_audio = signal.lfilter(b, a, audio)  # Aplicar o filtro ao áudio

# Plotar o áudio original e os sinais processados
t = np.arange(audio.shape[0]) / fs  # Criar um vetor de tempo baseado na taxa de amostragem e no número de amostras
plt.figure(figsize=(12, 8))  # Definir o tamanho da figura para plotagem

plt.subplot(3, 1, 1)
plt.plot(t, audio)
plt.title('Áudio Original')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(t, filtered_audio)
plt.title('Áudio Filtrado (Passa-Baixa)')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(t, equalized_audio)
plt.title('Áudio Equalizado (Boost em 1kHz)')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()