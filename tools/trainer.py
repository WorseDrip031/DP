import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics import Accuracy

from tools.statistics import Statistics
from tools.logging import LogCompose

class Trainer:
    def __init__(self, cfg, model):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        self.datamodule = None
        self.cfg = cfg
        self.model = model.to(self.device)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=self.cfg.learning_rate)
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy("multiclass", num_classes=model.num_classes)
        
    def setup(self, datamodule, logs=[]):
        self.log = LogCompose(list_log=logs)
        self.datamodule = datamodule
        self.datamodule.setup(self.cfg)
        
    def fit(self):
        self.log.on_training_start()

        for epoch in range(self.cfg.max_epochs):
            
            stats_train = Statistics()
            self.train_epoch(epoch, self.model, self.datamodule.dataloader_train, stats_train)
                    
            stats_val = Statistics()
            self.validate_epoch(epoch, self.model, self.datamodule.dataloader_val, stats_val)

            self.log.on_epoch_complete(epoch, Statistics.merge(stats_train, stats_val))

        self.log.on_training_stop()

    def train_epoch(self, epoch, model, dataloader, stats):
        model.train()
        with tqdm(dataloader, desc=f"Train: {epoch}") as progress:
            for x, y in progress:
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                y_hat_logits = model(x)
                l = self.loss(y_hat_logits, y)
                
                self.opt.zero_grad()
                l.backward()
                self.opt.step()

                acc = self.acc(
                    torch.softmax(y_hat_logits, dim=1).cpu(),      # Predictions
                    torch.argmax(y, dim=1).cpu()                   # Classes
                )
                
                stats.step("loss_train", l.item())
                stats.step("acc_train", acc.item())
                progress.set_postfix(stats.get())

    def validate_epoch(self, epoch, model, dataloader, stats):
        model.eval()
        with torch.no_grad():
            with tqdm(dataloader, desc=f"Val: {epoch}") as progress:
                for x, y in progress:
                    
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    y_hat_logits = model(x)
                    l = self.loss(y_hat_logits, y)
                    
                    acc = self.acc(
                        torch.softmax(y_hat_logits, dim=1).cpu(),      # Predictions
                        torch.argmax(y, dim=1).cpu()                   # Classes
                    )

                    stats.step("loss_val", l.item())
                    stats.step("acc_val", acc.item())
                    progress.set_postfix(stats.get())