import torch
import torch.nn as nn
import numpy as np
import copy
import sys

def model_zeros(model, device = 'cpu'):
    # Cria uma cópia profunda do modelo para que o modelo original não seja alterado
    copy_model = copy.deepcopy(model)
    for param in copy_model.parameters():
        # zera todos os parametros
        param.data.zero_()

    return copy_model

def random_param(model, device = 'cpu'):
    # Cria uma cópia profunda do modelo para que o modelo original não seja alterado
    copy_model = copy.deepcopy(model)
    for param in copy_model.parameters():
        # gera valores aleatorios para serem utilizados como parametros
        param.data = torch.rand_like(param.data)

    return copy_model

def shuffle_model(model):
    # Cria uma cópia profunda do modelo para que o modelo original não seja alterado
    copy_model = copy.deepcopy(model)
    
    # Itera sobre todos os parâmetros do modelo copiado
    for param in copy_model.parameters():
        # Achata o tensor de parâmetros para uma dimensão
        data_flatten = param.data.view(-1)
        
        # Gera uma permutação aleatória dos índices dos elementos
        index_random = torch.randperm(len(data_flatten))
        
        # Aplica a permutação ao tensor achatado
        shuffled_param = data_flatten[index_random]
        
        # Redimensiona o tensor embaralhado de volta ao formato original
        param.data = shuffled_param.view(param.data.shape)
    
    return copy_model

def model_noise(model, SNR, client):
    # Fazendo uma cópia profunda do modelo para preservar o modelo original
    client.train()
    return client.model

    for param in copy_model.parameters():
        with torch.no_grad():
            flatten_param = torch.flatten(param.data)
            num_elements = len(flatten_param)

            #print(param.data[0])
            # Calculando a potência do sinal
            power_signal = torch.mean(param.data ** 2)

            # Calculando a potência do ruído com base na SNR desejada
            power_noise =  power_signal / SNR
            
            #print(f"Potência do ruído: {power_noise.item()}")

            # Gerando o ruído com a distribuição normal com desvio padrão adequado
            noise_matrix = np.random.normal(0.0, np.sqrt(power_noise), size = param.data.shape)

            # Adicionando o ruído ao parâmetro original
            param.add_(torch.tensor(noise_matrix))

            # Verificação da potência real do ruído (opcional)
            #actual_power_noise = torch.sum(noise_matrix ** 2) / num_elements
            #actual_power_signal = torch.sum(param.data ** 2) / num_elements
            #print(f"Potência real do ruído: {actual_power_noise.item()}")
            #print(f"Potência final do sinal: {actual_power_signal.item()}")

            # Calculando a SNR real (opcional)
            snr_real = (power_signal / noise_matrix)
            #print(f"SNR real calculado: {snr_real.item()}")


    return copy_model

#if __name__ == "__main__":
    #model = nn.Linear(3, 2)

    
    #zero_model = model_zeros(model)
    #random_model = random_param(model)
    #shuffle_model_ = shuffle_model(model)
    #noise_model_ = model_noise(model, 1)

    '''print("Current Model: ")
    for param in model.parameters():
        print(param)

    print("\nShuffle Model: ")
    for param in shuffle_model_.parameters():
        print(param)

    print("\nRandom Model: ")
    for param in random_model.parameters():
        print(param)

    print("\nNoise Model: ")
    for param in noise_model_.parameters():
        print(param)

    print("\nZero Model: ")
    for param in zero_model.parameters():
        print(param)'''
