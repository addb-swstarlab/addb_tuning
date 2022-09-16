import torch
import torch.nn as nn
from torch.utils.data import Dataset

class DB_Dataset(Dataset):
    def __init__(self, X, y):
        super(DB_Dataset, self).__init__()
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


class ADDB_Dataset(Dataset):
    def __init__(self, dbms1, dbms2, dbms3, y):
        super(ADDB_Dataset, self).__init__()
        self.dbms1 = dbms1
        self.dbms2 = dbms2
        self.dbms3 = dbms3
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.dbms1[idx], self.dbms2[idx], self.dbms3[idx], self.y[idx])


class SingleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SingleNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.linear1 = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    

class ADDBNet(nn.Module):
    def __init__(self, dbms1_dim, dbms2_dim, dbms3_dim, hidden_dim, output_dim, params):
        super(ADDBNet, self).__init__()
        # params : ['model_save/20220907/Redis-20220907-00.pt', 'model_save/20220907/RocksDB-20220907-00.pt', 'model_save/20220907/Spark-20220907-00.pt']
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.dbms1_dim = dbms1_dim
        self.dbms2_dim = dbms2_dim
        self.dbms3_dim = dbms3_dim

        if params[0] is not None:
            self.linear_dbms1 = list(torch.load(params[0]).children())[0]
            self.linear_dbms2= list(torch.load(params[1]).children())[0]
            self.linear_dbms3 = list(torch.load(params[2]).children())[0]
            self.input_dim = self.linear_dbms1.state_dict()['0.weight'].shape[0] * 3
        elif params[0] is None:
            self.linear_dbms1 = nn.Sequential(nn.Linear(self.dbms1_dim, self.hidden_dim), nn.ReLU())
            self.linear_dbms2= nn.Sequential(nn.Linear(self.dbms2_dim, self.hidden_dim), nn.ReLU())
            self.linear_dbms3 = nn.Sequential(nn.Linear(self.dbms3_dim, self.hidden_dim), nn.ReLU())
            self.input_dim = self.hidden_dim * 3
        
        self.hidden = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))
        
    def forward(self, dbms1, dbms2, dbms3):
        x1 = self.linear_dbms1(dbms1)
        x2 = self.linear_dbms2(dbms2)
        x3 = self.linear_dbms3(dbms3)
        
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.hidden(x)
        x = self.fc(x)
        return x
        
        