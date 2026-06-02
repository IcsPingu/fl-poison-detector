import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG
from flcore.clients.clientmaliciousavg import ClientMaliciousAVG
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
import numpy as np
#from scipy.cluster.hierarchy import dendrogram, linkage
class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break
        self.cc = args.cluster_comparation
        self.rate_client_fake = args.rate_client_fake


        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.rs_train_time = []
        self.Budget = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        self.n_client_malicious = args.n_client_malicious
        self.current_round = -1
        self.cc = args.cluster_comparation
        self.ids =[]
       
    def set_clients(self, clientObj):
        indexes = list(range(self.num_clients))
        self.client_quarantine_dict = {indexes: {'quarentena': 0, 'roundsQuarent': 0} for indexes in range(self.num_clients)}
        n_malicious = self.n_client_malicious
        self.index_malicious = np.random.choice(indexes, n_malicious, replace=False)
        print(self.index_malicious)
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            
            if i not in self.index_malicious:
                client = clientObj(self.args, id=i, train_samples=len(train_data), 
                                test_samples=len(test_data), train_slow=train_slow, 
                                send_slow=send_slow)
            elif i in self.index_malicious:
                client = ClientMaliciousAVG(self.args, id=i, train_samples=len(train_data), 
                                test_samples=len(test_data), train_slow=train_slow, 
                                send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        self.current_round += 1
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        quarantined_ids = set()
        for client_id, status in self.client_quarantine_dict.items():
        # Verifica a condição
            if status['roundsQuarent'] > 0:
                quarantined_ids.add(client_id)
        new_num_clients = [client for client in self.clients if client.id not in quarantined_ids]
        if not new_num_clients:
            print('[select_clients] todos os clientes estao em quarentena; mantendo o modelo global neste round.')
            self.current_num_join_clients = 0
            return []
        self.current_num_join_clients = max(1, int(len(new_num_clients) * self.join_ratio))
        self.current_num_join_clients = min(self.current_num_join_clients, len(new_num_clients))
        selected_clients = list(np.random.choice(new_num_clients, self.current_num_join_clients, replace=False))
        print(len(selected_clients))
        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        if len(self.selected_clients) == 0:
            self.uploaded_ids = []
            self.uploaded_weights = []
            self.uploaded_models = []
            self.ids = []
            return

        active_count = int((1 - self.client_drop_rate) * self.current_num_join_clients)
        active_count = min(max(0, active_count), len(self.selected_clients))
        if active_count == 0:
            self.uploaded_ids = []
            self.uploaded_weights = []
            self.uploaded_models = []
            self.ids = []
            return
        active_clients = random.sample(self.selected_clients, active_count)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.ids =[]
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.send_local_model(self.current_round))
                self.ids.append(client.id)
        if tot_samples <= 0:
            self.uploaded_weights = []
            self.uploaded_models = []
            self.uploaded_ids = []
            self.ids = []
            return
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def cosine_similarity(self, model1_params, model2_params):
        #Calculo do cosseno
        model1_params_flat = np.concatenate([p.detach().cpu().numpy().flatten() for p in model1_params])
        model2_params_flat = np.concatenate([p.detach().cpu().numpy().flatten() for p in model2_params])

        
        dot_product = np.dot(model1_params_flat, model2_params_flat)
        norm_model1 = np.linalg.norm(model1_params_flat)
        norm_model2 = np.linalg.norm(model2_params_flat)
        
        similarity = dot_product / (norm_model1 * norm_model2)
        return similarity

    def calculate_similarity_with_global_model(self, global_model_params):
        #Similaridade cosseno com o modelo global 
        similarities = []
        for idx, client_model in enumerate(self.uploaded_models):
            client_model_params = list(client_model.parameters())  # Parâmetros cliente
            similarity = self.cosine_similarity(client_model_params, global_model_params)
            similarities.append((self.ids[idx], similarity)) # Armazenando o ID do e a similaridade
        return similarities
    def flatten_model_params(self, model):
        #retorna parametros achatados
        return torch.cat([p.detach().flatten() for p in model.parameters()])
    
    def calcular_gradiente_l2(self, modelo, lambda_l2=0.01):
        # Sua implementação do cálculo do gradiente L2
        gradiente_l2 = 0.0
        for parametro in modelo.parameters():
            if parametro.grad is not None:
                gradiente_l2 += torch.sum(parametro ** 2).item()
        gradiente_l2 = lambda_l2 * gradiente_l2
        return gradiente_l2

    def calculate_similarity_scores(self):
        #Calcula a matriz de similaridade de cosseno entre todos os clientes e retorna a matriz + os scores médios de cada cliente.

        num_clients = len(self.uploaded_models)
        if num_clients < 2:
            return np.eye(num_clients), {self.ids[i]: 0.0 for i in range(num_clients)}
        
        #Flatten todos os modelos e empilhar em um único tensor
        all_params = torch.stack([self.flatten_model_params(model) for model in self.uploaded_models])  # shape: [num_clients, num_features]
        
        #Normalizar os vetores para calcular cosseno facilmente
        norms = torch.norm(all_params, dim=1, keepdim=True)  # [num_clients, 1]
        normalized_params = all_params / (norms + 1e-10)
        
        #Calcular a matriz de similaridade: dot product entre vetores normalizados
        similarity_matrix = torch.matmul(normalized_params, normalized_params.T)  # shape: [num_clients, num_clients]
        #for row in similarity_matrix:
        #    print("  ".join([f"{x:0.6f}" for x in row])) 
        #Converter para NumPy
        similarity_matrix_np = similarity_matrix.cpu().numpy()
        desvio_padrao = torch.std(similarity_matrix)
        grad=[]
        if desvio_padrao < 0.01:
    # Inicializar o dicionário para armazenar os gradientes dos clientes
            grad = {}

            # Itera sobre os modelos carregados
            for i, model in enumerate(self.uploaded_models):
                # Calcula o gradiente L2 para cada modelo
                gradiente_l2 = self.calcular_gradiente_l2(model, lambda_l2=0.01)
                
                # Adiciona o gradiente ao dicionário com o id do cliente
                grad[self.ids[i]] = gradiente_l2
                
                # Exemplo de adicionar esse gradiente à perda do modelo
                try:
                    print(f"Gradiente L2 calculado para o cliente {self.ids[i]}: {gradiente_l2}")
                except Exception as e:
                    print(f"Erro ao imprimir o gradiente para o cliente {self.ids[i]}: {e}")

            # Exibe a média dos gradientes
            print(np.mean(list(grad.values())))

            return similarity_matrix_np, grad

        #Calcular score médio de cada cliente (excluindo diagonal)
        #scores = (similarity_matrix_np.sum(axis=1) - 1) / (num_clients - 1)  # média da similaridade com os outros clientes
        scores= self.calculate_client_scores(similarity_matrix_np)
        return similarity_matrix_np, scores

        
    
    """
    def calculate_similarity_scores(self):
        #Similaridade entre todos os clientes
        num_clients = len(self.uploaded_models)
        if num_clients < 2:
            return {self.ids[i]: 0.0 for i in range(num_clients)}
        similarity_matrix = np.zeros((num_clients, num_clients))  # Matriz de similaridade
        
        for i in range(len(self.uploaded_models)):
            for j in range(len(self.uploaded_models)):
                if i != j:  # Evitar comparar o cliente com ele mesmo
                    client_model_i_params = list(self.uploaded_models[i].parameters())
                    client_model_j_params = list(self.uploaded_models[j].parameters())
                    
                    # Calculando a similaridade de cosseno entre os clientes
                    similarity = self.cosine_similarity(client_model_i_params, client_model_j_params)
                    
                    # Somando a similaridade na pontuação
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity  # A similaridade é simétrica
        
        # Normalizar as pontuações  
        #num_comparisons = len(self.uploaded_models) - 1  #Ta certo isso?
        #for client_id in similarity_scores:
        #   similarity_scores[client_id] /= num_comparisons
        #print(similarity_matrix)
        a= self.calculate_client_scores(similarity_matrix)
        
        return similarity_matrix, a
    """
    def calculate_client_scores(self, similarity_matrix):
        """
        Calcula um score para cada cliente baseado na matriz de similaridade.
        Score = média da similaridade do cliente com todos os outros clientes.
        """
          # Matriz de similaridade (NxN)
        num_clients = len(self.uploaded_models)
        
        # Inicializa um dicionário para armazenar os scores
        client_scores = {}
        
        for i in range(num_clients):
            # Exclui a diagonal (similaridade consigo mesmo)
            sum_similarity = np.sum(similarity_matrix[i]) - similarity_matrix[i, i]
            score = sum_similarity / (num_clients - 1)  # Média das similaridades com os outros clientes
            client_scores[self.ids[i]] = score
        
        # Opcional: normalizar scores entre 0 e 1
        scores_array = np.array(list(client_scores.values()))
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)
        
        for client_id in client_scores:
            client_scores[client_id] = (client_scores[client_id] - min_score) / (max_score - min_score + 1e-10)
        return client_scores

    def perform_clustering(self, similarity_matrix, num_clusters=2):
        # Convertendo a matriz de similaridade em uma matriz de distâncias
        distance_matrix = 1 - similarity_matrix  # Distância = 1 - Similaridade de Cosseno

        # Aplicando o KMeans
        #kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        #kmeans.fit(distance_matrix)
        
        # Atribuindo os clusters aos clientes
        #clusters = kmeans.labels_
        agglomerative = AgglomerativeClustering(n_clusters=num_clusters)
    
    # Ajustando e atribuindo os clusters
        clusters = agglomerative.fit_predict(distance_matrix)
        return clusters

    def calculate_shannon_entropy(self, model_params):
        # Modelo em lista de array 1D 
        model_params_flat = np.concatenate([p.detach().cpu().numpy().flatten() for p in model_params])

        # probabilidade de distribuiçao 
        hist, bin_edges = np.histogram(model_params_flat, bins=50, density=True)

        # Normalização suspeita 
        hist = hist / np.sum(hist)

        entropy = -np.sum(hist * np.log2(hist + 1e-10))  #log(0)
        return entropy

    def calculate_client_entropies(self):

        client_entropies = {}

        for idx, client_model in enumerate(self.uploaded_models):
            client_model_params = list(client_model.parameters())
            entropy = self.calculate_shannon_entropy(client_model_params)
            client_entropies[self.ids[idx]] = entropy
            print(f"Shannon entropy for client {self.ids[idx]}: {entropy:.4f}")

        return client_entropies

    def aggregate_parameters(self):
        if len(self.uploaded_models) == 0:
            print('[aggregate] nenhum upload valido; mantendo o modelo global anterior.')
            return

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm + "_" + str(self.cc) + "_" + str((self.rate_client_fake*100)) + "_" + str(self.n_client_malicious)
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_train_time', data=self.Budget)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
