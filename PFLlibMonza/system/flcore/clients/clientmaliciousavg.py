import numpy as np
import time
import random
from flcore.clients.clientavg import clientAVG

#from utils.privacy import *

from flcore.attack.attack import *


class ClientMaliciousAVG(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.rate_client_fake = args.rate_client_fake
        self.atack = args.atack

        self.label_flip_epochs = max(1, int(getattr(args, 'label_flip_epochs', args.local_epochs)))
        self.label_flip_lr_multiplier = float(getattr(args, 'label_flip_lr_multiplier', 1.0))
        #self.delay_atk = args.delay_atk
        self.round_init_atk = args.round_init_atk
        self.current_round = 0
        self.pending_attack_type = 'benign'

    def client_entropy(self):
        entropy_client = self.calculate_data_entropy()
        return entropy_client
    
    def _choose_attack_type(self):
        if self.current_round <= self.round_init_atk:
            return 'benign'
        is_malicious = np.random.choice(
            [False, True],
            p=[1 - self.rate_client_fake, self.rate_client_fake],
        )
        if not is_malicious:
            return 'benign'
        if self.atack == 'zero':
            return 'malicious_zeros'
        if self.atack == 'random':
            return 'malicious_random'
        if self.atack == 'shuffle':
            return 'malicious_shuffle'
        if self.atack == 'label':
            return 'malicious_label'
        if self.atack == 'all':
            return random.choice([
                'malicious_zeros',
                'malicious_random',
                'malicious_shuffle',
                'malicious_label',
            ])
        return 'benign'

    def _train_label_flip_attack(self):
        trainloader = self.load_train_data(None, is_malicious=True, drop_last=False)
        self.model.train()

        original_lrs = [group['lr'] for group in self.optimizer.param_groups]
        for group, lr in zip(self.optimizer.param_groups, original_lrs):
            group['lr'] = lr * self.label_flip_lr_multiplier

        max_local_epochs = self.label_flip_epochs
        if self.train_slow:
            slow_upper = max(2, max_local_epochs // 2 + 1)
            max_local_epochs = max(1, np.random.randint(1, slow_upper))

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        for group, lr in zip(self.optimizer.param_groups, original_lrs):
            group['lr'] = lr

    def train(self):
        self.pending_attack_type = self._choose_attack_type()
        self.last_attack_type = self.pending_attack_type
        self.is_malicious = self.pending_attack_type != 'benign'

        if not self.is_malicious:
            return super().train()

        print(f'malicioso: {self.id}')
        if self.pending_attack_type != 'malicious_label':
            return super().train()

        start_time = time.time()
        self._train_label_flip_attack()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
    
    def send_local_model(self, round):
        if self.pending_attack_type == 'malicious_zeros':
            return model_zeros(self.model, self.device)
        if self.pending_attack_type == 'malicious_random':
            return random_param(self.model, self.device)
        if self.pending_attack_type == 'malicious_shuffle':
            return shuffle_model(self.model)
        return self.model
