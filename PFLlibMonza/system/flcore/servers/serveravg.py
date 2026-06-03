import time
import uuid
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
from collections import Counter
import csv
import os
from flcore.detector import fl_save
from flcore.detector.cc import ClientCheck
from flcore.detector.cc_mlp import ClientCheckMLP
class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.fpr_frr_results = []
        self.run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Open the CSV file in append mode to save results over time
        if self.cc ==3:
            self.csv_filename = 'fpr_frr_results_3.csv'
        elif self.cc ==2:
            self.csv_filename = 'fpr_frr_results_2.csv'
        elif self.cc ==6:
            self.csv_filename = 'fpr_frr_results_6.csv'
        elif self.cc ==7:
            self.csv_filename = 'fpr_frr_results_7.csv'
        else:
            self.csv_filename = 'f.csv'
        self._ensure_csv_header(self.csv_filename, ['RunID', 'Round', 'FPR', 'FRR', 'UploadFPR', 'UploadFRR'])
        self.cc_detail_filename = f'cc_detail_results_{self.cc}.csv'
        self.cc_type_filename = f'cc_type_results_{self.cc}.csv'
        if self.cc in (2, 3, 6, 7):
            detail_header = [
                'RunID', 'Round', 'CC', 'ClientID', 'AttackType', 'IsMaliciousRound',
                'MaliciousGroup', 'Removed', 'Reason', 'MLPHit', 'BERTHit',
                'MLPScore', 'BERTScore', 'MLPLabelScore', 'BERTLabelScore',
                'MLPBinaryHit', 'BERTBinaryHit', 'MLPLabelHit', 'BERTLabelHit',
                'BinaryThreshold', 'LabelThreshold', 'DecisionRule', 'BaselineScore',
            ]
            self._ensure_csv_header(self.cc_detail_filename, detail_header)
            self._ensure_csv_header(
                self.cc_type_filename,
                ['RunID', 'Round', 'CC', 'AttackType', 'Total', 'Removed', 'Rate', 'Metric'],
            )

        self.dump_dir = getattr(args, 'dump_state_dicts', '') or ''
        self.dump_start_round = int(getattr(args, 'dump_start_round', 0))
        self.client_check = None
        self.bert_client_check = None
        self.mlp_client_check = None
        if self.cc == 6:
            detector_dir = getattr(args, 'detector_dir', '') or ''
            if not detector_dir:
                raise ValueError("cc=6 requer --detector_dir apontando pro modelo treinado (ex: jpt/detector_final/).")
            print(f"[cc=6] Carregando detector DistilBERT de {detector_dir}")
            self.client_check = ClientCheck(detector_dir, threshold_key=getattr(args, 'bert_threshold_key', 'threshold_label_fpr05'))
        elif self.cc == 7:
            detector_dir = getattr(args, 'detector_dir', '') or ''
            if not detector_dir:
                raise ValueError("cc=7 requer --detector_dir apontando pro MLP artifacts dir (ex: jpt/detector_mlp_monza_cnn_mnist/).")
            print(f"[cc=7] Carregando detector MLP de {detector_dir}")
            self.client_check = ClientCheckMLP(detector_dir, threshold_key=getattr(args, 'mlp_threshold_key', 'threshold_label_fpr05'))

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()

    def _ensure_csv_header(self, filename, header):
        if os.path.exists(filename):
            with open(filename, newline='') as file:
                current_header = next(csv.reader(file), [])
            if current_header != header:
                legacy_name = f"{filename}.legacy_{self.run_id}"
                os.rename(filename, legacy_name)
                print(f"[cc={self.cc}] Arquivando CSV com header antigo: {legacy_name}")
        if not os.path.exists(filename):
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)

    def save_fpr_frr_to_csv(self, round_number, FPR, FRR, upload_fpr='', upload_frr=''):
        """
        Saves the FPR and FRR results to a CSV file for each round.
        """
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.run_id, round_number, FPR, FRR, upload_fpr, upload_frr])

    def save_cc_detail_to_csv(self, rows):
        if not rows:
            return
        with open(self.cc_detail_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            for row in rows:
                writer.writerow([
                    self.run_id, row['round'], row['cc'], row['client_id'], row['attack_type'],
                    int(row['is_malicious_round']), int(row['malicious_group']),
                    int(row['removed']), row['reason'], int(row['mlp_hit']),
                    int(row['bert_hit']), row['mlp_score'], row['bert_score'],
                    row.get('mlp_label_score', 0.0), row.get('bert_label_score', 0.0),
                    int(row.get('mlp_binary_hit', False)), int(row.get('bert_binary_hit', False)),
                    int(row.get('mlp_label_hit', False)), int(row.get('bert_label_hit', False)),
                    row.get('binary_threshold', ''), row.get('label_threshold', ''),
                    row.get('decision_rule', 'binary'),
                    row.get('baseline_score', 0.0),
                ])

    def save_cc_type_to_csv(self, round_number, rows):
        if not rows:
            return
        grouped = {}
        for row in rows:
            attack_type = row['attack_type']
            bucket = grouped.setdefault(attack_type, {'total': 0, 'removed': 0})
            bucket['total'] += 1
            bucket['removed'] += int(row['removed'])
        with open(self.cc_type_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            for attack_type in sorted(grouped):
                bucket = grouped[attack_type]
                total = bucket['total']
                removed = bucket['removed']
                rate = removed / total if total else 0.0
                metric = 'FPR' if attack_type == 'benign' else 'recall'
                writer.writerow([self.run_id, round_number, self.cc, attack_type, total, removed, rate, metric])

    def _build_cc_detail_rows(self, round_number, all_client_ids, removed_clients, reason, scores=None):
        clients_by_id = {c.id: c for c in self.clients}
        removed_set = set(int(cid) for cid in removed_clients)
        scores = scores or {}
        rows = []
        for client_id in all_client_ids:
            client = clients_by_id.get(int(client_id))
            attack_type = getattr(client, 'last_attack_type', 'unknown')
            is_malicious_round = bool(getattr(client, 'is_malicious', False))
            removed = int(client_id) in removed_set
            rows.append({
                'round': round_number,
                'cc': self.cc,
                'client_id': int(client_id),
                'attack_type': attack_type,
                'is_malicious_round': is_malicious_round,
                'malicious_group': int(client_id) in self.index_malicious,
                'removed': removed,
                'reason': reason if removed else 'none',
                'mlp_hit': False,
                'bert_hit': False,
                'mlp_score': 0.0,
                'bert_score': 0.0,
                'mlp_label_score': 0.0,
                'bert_label_score': 0.0,
                'mlp_binary_hit': False,
                'bert_binary_hit': False,
                'mlp_label_hit': False,
                'bert_label_hit': False,
                'binary_threshold': '',
                'label_threshold': '',
                'decision_rule': reason,
                'baseline_score': float(scores.get(int(client_id), 0.0)),
            })
        return rows

    def compute_upload_fpr_frr(self, rows):
        if not rows:
            return '', ''
        FP = TP = FN = TN = 0
        for row in rows:
            removed = bool(row['removed'])
            is_malicious = bool(row['is_malicious_round'])
            if removed and not is_malicious:
                FP += 1
            elif removed and is_malicious:
                TP += 1
            elif not removed and is_malicious:
                FN += 1
            else:
                TN += 1
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        frr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        return fpr, frr

    def set_client_quarantine(self, client_id):
        self.client_quarantine_dict[client_id]['quarentena'] = self.client_quarantine_dict[client_id]['quarentena'] +1
        self.client_quarantine_dict[client_id]['roundsQuarent'] = self.client_quarantine_dict[client_id]['quarentena'] *2

    def decrease_quarentine(self, client_id):
        if self.client_quarantine_dict[client_id]['roundsQuarent'] ==0:
            self.client_quarantine_dict[client_id]['roundsQuarent'] = 0
        else:
            self.client_quarantine_dict[client_id]['roundsQuarent'] = self.client_quarantine_dict[client_id]['roundsQuarent'] -1
    def compute_fpr_frr(self):
        """
        Calcula False Positive Rate (FPR) e False Rejection Rate (FRR)
        usando self.client_quarantine_dict e self.index_malicious.
        """
        FP = 0  # Falsos positivos: clientes em quarentena mas não maliciosos
        TP = 0  # Verdadeiros positivos: clientes em quarentena e maliciosos
        FN = 0  # Falsos negativos: maliciosos não detectados
        TN = 0  # Verdadeiros negativos: não maliciosos e não em quarentena

        for client_id in range(self.num_clients):
            in_quarantine = self.client_quarantine_dict[client_id]['roundsQuarent'] > 0
            is_malicious = client_id in self.index_malicious

            if in_quarantine and not is_malicious:
                FP += 1
            elif in_quarantine and is_malicious:
                TP += 1
            elif not in_quarantine and is_malicious:
                FN += 1
            elif not in_quarantine and not is_malicious:
                TN += 1

        # Evitar divisão por zero
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FRR = FN / (FN + TP) if (FN + TP) > 0 else 0

        return FPR, FRR

    def compute_fpr_frr_cluster(self, removed_clients, cluster_tuples):
        FP = 0
        TP = 0
        FN = 0
        TN = 0

        for client_id in removed_clients:
            if client_id in self.index_malicious:
                TP += 1
            else:
                FP += 1

        for client_id, _cluster in cluster_tuples:
            if client_id not in removed_clients:
                if client_id in self.index_malicious:
                    FN += 1
                else:
                    TN += 1

        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FRR = FN / (FN + TP) if (FN + TP) > 0 else 0
        return FPR, FRR

    def train(self):
        
        for i in range(self.global_rounds+1):
            s_t = time.time()
            global_state_before_round = {
                k: v.detach().clone()
                for k, v in self.global_model.state_dict().items()
            }
            quarantined_at_round_start = {
                client_id for client_id, status in self.client_quarantine_dict.items()
                if status['roundsQuarent'] > 0
            }
            self.selected_clients = self.select_clients()
            self.send_models()
            self.removed_clients = []
            self.cluster_tuples = ()
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            round_detail_rows = []
            #for client in self.selected_clients:
            #    client.train()
            for client in self.selected_clients:
                client.current_round = self.current_round

            threads = [Thread(target=client.train)
                       for client in self.selected_clients]
            [t.start() for t in threads]
            [t.join() for t in threads]

            self.receive_models()

            # Dump state_dicts pra geracao de dataset (modo --dump_state_dicts)
            if self.dump_dir and i >= self.dump_start_round:
                clients_by_id = {c.id: c for c in self.clients}
                n_saved = fl_save.save_round_dump(
                    self.uploaded_models, self.uploaded_ids,
                    clients_by_id, self.index_malicious,
                    round_idx=i, out_dir=self.dump_dir,
                    global_state_dict=global_state_before_round,
                )
                print(f'[dump] round {i}: salvos {n_saved} state_dicts em {self.dump_dir}')
            elif self.dump_dir:
                print(f'[dump] round {i}: ignorado antes de dump_start_round={self.dump_start_round}')

            if i > 0 and self.uploaded_models:
                if self.cc==2:
                    oi = time.time()
                    round_upload_ids = list(self.ids)
                    if len(self.uploaded_models) < 2:
                        print("cc=2: menos de 2 uploads validos; pulando clustering neste round.")
                    else:
                        similarity_matrix, _ = self.calculate_similarity_scores()
                        num_clusters = min(2, len(self.uploaded_models))
                        clusters = self.perform_clustering(similarity_matrix, num_clusters)

                        self.cluster_tuples = [(self.ids[idx], cluster) for idx, cluster in enumerate(clusters)]
                        for idx, cluster in enumerate(clusters):
                            print(f"Client {self.ids[idx]} is in cluster {cluster}")
                        cluster_counts = Counter([cluster for _, cluster in self.cluster_tuples])
                        min_cluster = min(cluster_counts, key=cluster_counts.get)

                        for idx in range(len(self.cluster_tuples) - 1, -1, -1):
                            client_id, cluster = self.cluster_tuples[idx]
                            if cluster == min_cluster:
                                print(f"Removing client {client_id} from cluster {cluster}")
                                self.removed_clients.append(client_id)
                                del self.uploaded_models[idx]
                                del self.ids[idx]
                                del self.uploaded_ids[idx]
                                del self.uploaded_weights[idx]
                        if self.uploaded_weights:
                            s = sum(self.uploaded_weights)
                            if s > 0:
                                self.uploaded_weights = [weight / s for weight in self.uploaded_weights]
                    detail_rows = self._build_cc_detail_rows(
                        i, round_upload_ids, self.removed_clients, reason='cluster_minority'
                    )
                    round_detail_rows = detail_rows
                    self.save_cc_detail_to_csv(detail_rows)
                    self.save_cc_type_to_csv(i, detail_rows)
                    print(f"Tempo de execução: {time.time()-oi:.4f} segundos")

                if self.cc==3:
                    oi = time.time()
                    round_upload_ids = list(self.ids)
                    client_scores = {}
                    if len(self.uploaded_models) < 2:
                        print("cc=3: menos de 2 uploads validos; pulando score neste round.")
                    else:
                        _similarity_matrix, client_scores  = self.calculate_similarity_scores()
                        scores_array = np.array(list(client_scores.values()))
                        mean_score = np.mean(scores_array)
                        std_score = np.std(scores_array)
                        print(f"Average score: {mean_score:.4f}")
                        mean_score = mean_score - std_score
                        print(f"Average score: {mean_score:.4f}")
                        client_tuples = [(self.ids[idx], client_scores[self.ids[idx]]) for idx in range(len(self.ids))]
                        total = len(self.index_malicious)
                        found = 0
                        if std_score < 0.001:
                            print("nenhum malicioso")
                        else:
                            for idx in range(len(client_tuples) - 1, -1, -1):
                                client_id, score = client_tuples[idx]
                                print(f"Esse  {client_id} with score {score:.4f} ")
                                if score < mean_score:
                                    if client_id in self.index_malicious:
                                        found += 1
                                    print(f"Removing client {client_id} with score {score:.4f} (below average)")
                                    self.set_client_quarantine(client_id)
                                    del self.uploaded_models[idx]
                                    del self.ids[idx]
                                    del self.uploaded_ids[idx]
                                    del self.uploaded_weights[idx]
                        found_pct = (found/total) * 100 if total > 0 else 0.0
                        print("porcentagem de clientes maliciosos de verdade achados: "+ str(found_pct) + "%")
                        if self.uploaded_weights:
                            s = sum(self.uploaded_weights)
                            if s > 0:
                                self.uploaded_weights = [weight / s for weight in self.uploaded_weights]
                    detail_rows = self._build_cc_detail_rows(
                        i, round_upload_ids, self.removed_clients,
                        reason='score_below_mean_minus_std',
                        scores=client_scores,
                    )
                    round_detail_rows = detail_rows
                    self.save_cc_detail_to_csv(detail_rows)
                    self.save_cc_type_to_csv(i, detail_rows)
                    print(f"Tempo de execução: {time.time()-oi:.4f} segundos")

                if self.cc==5:
                    print("vai rolar nada")
                if self.cc in (6, 7):
                    oi = time.time()
                    detector_name = 'DistilBERT' if self.cc == 6 else 'MLP'
                    clients_by_id = {c.id: c for c in self.clients}
                    true_positive_uploads = 0
                    malicious_uploads = 0
                    detail_rows = []
                    for idx in range(len(self.uploaded_models) - 1, -1, -1):
                        client_id = self.ids[idx]
                        client = clients_by_id.get(client_id)
                        attack_type = getattr(client, 'last_attack_type', 'unknown')
                        is_malicious_round = bool(getattr(client, 'is_malicious', False))
                        if is_malicious_round:
                            malicious_uploads += 1
                        sd = self.uploaded_models[idx].state_dict()
                        mlp_result = {'is_malicious': False, 'score': 0.0, 'label_score': 0.0}
                        bert_result = {'is_malicious': False, 'score': 0.0, 'label_score': 0.0}
                        if self.cc == 6:
                            bert_result = self.client_check.classify(
                                sd, global_state_dict=global_state_before_round
                            )
                            bert_hit = bool(bert_result['is_malicious'])
                            mlp_hit = False
                        else:
                            mlp_result = self.client_check.classify(
                                sd, global_state_dict=global_state_before_round
                            )
                            mlp_hit = bool(mlp_result['is_malicious'])
                            bert_hit = False
                        removed = bool((self.cc == 6 and bert_hit) or (self.cc == 7 and mlp_hit))
                        reason_parts = []
                        if mlp_hit:
                            reason_parts.append('mlp')
                        if bert_hit:
                            reason_parts.append('bert')
                        reason = '+'.join(reason_parts) if reason_parts else 'none'
                        detail_rows.append({
                            'round': i,
                            'cc': self.cc,
                            'client_id': client_id,
                            'attack_type': attack_type,
                            'is_malicious_round': is_malicious_round,
                            'malicious_group': client_id in self.index_malicious,
                            'removed': removed,
                            'reason': reason if removed else 'none',
                            'mlp_hit': mlp_hit,
                            'bert_hit': bert_hit,
                            'mlp_score': float(mlp_result.get('score', 0.0)),
                            'bert_score': float(bert_result.get('score', 0.0)),
                            'mlp_label_score': float(mlp_result.get('label_score', 0.0)),
                            'bert_label_score': float(bert_result.get('label_score', 0.0)),
                            'mlp_binary_hit': bool(mlp_result.get('binary_hit', False)),
                            'bert_binary_hit': bool(bert_result.get('binary_hit', False)),
                            'mlp_label_hit': bool(mlp_result.get('label_hit', False)),
                            'bert_label_hit': bool(bert_result.get('label_hit', False)),
                            'binary_threshold': (
                                mlp_result.get('binary_threshold')
                                if self.cc == 7 else bert_result.get('binary_threshold')
                            ),
                            'label_threshold': (
                                mlp_result.get('label_threshold')
                                if self.cc == 7 else bert_result.get('label_threshold')
                            ),
                            'decision_rule': (
                                mlp_result.get('decision_rule', 'binary')
                                if self.cc == 7 else bert_result.get('decision_rule', 'binary')
                            ),
                        })
                        if removed:
                            if is_malicious_round:
                                true_positive_uploads += 1
                            print(f'cc={self.cc}: removing client {client_id} ({detector_name} detector)')
                            self.set_client_quarantine(client_id)
                            del self.uploaded_models[idx]
                            del self.ids[idx]
                            del self.uploaded_ids[idx]
                            del self.uploaded_weights[idx]
                    self.save_cc_detail_to_csv(detail_rows)
                    self.save_cc_type_to_csv(i, detail_rows)
                    round_detail_rows = detail_rows
                    round_recall = true_positive_uploads / malicious_uploads if malicious_uploads > 0 else 0.0
                    print(
                        f'recall de uploads maliciosos no round (cc={self.cc}): '
                        f'{round_recall:.2%} ({true_positive_uploads}/{malicious_uploads})'
                    )
                    if self.uploaded_weights:
                        s = sum(self.uploaded_weights)
                        if s > 0:
                            self.uploaded_weights = [w / s for w in self.uploaded_weights]
                    print(f'Tempo de execução cc={self.cc}: {time.time()-oi:.4f}s')
            print(self.client_quarantine_dict)
            FPR=0
            FRR = 0
            if self.cc ==2:
                FPR, FRR = self.compute_fpr_frr_cluster(self.removed_clients, self.cluster_tuples)
            if self.cc ==3:
                FPR, FRR = self.compute_fpr_frr()
            if self.cc ==6:
                FPR, FRR = self.compute_fpr_frr()
            if self.cc ==7:
                FPR, FRR = self.compute_fpr_frr()
            print(f"Round {i}: False Positive Rate = {FPR:.4f}, False Rejection Rate = {FRR:.4f}")
            upload_fpr, upload_frr = self.compute_upload_fpr_frr(round_detail_rows)
            if upload_fpr != '':
                print(f"Round {i}: Upload FPR = {upload_fpr:.4f}, Upload FRR = {upload_frr:.4f}")
            self.save_fpr_frr_to_csv(i, FPR, FRR, upload_fpr, upload_frr)
            for client_id in quarantined_at_round_start:
                self.decrease_quarentine(client_id)
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
