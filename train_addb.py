import logging, os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import get_filename
from network import ADDB_Dataset, ADDBNet

def train(model, train_loader, lr):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    
    total_loss = 0.
    for dbms1, dbms2, dbms3, target in train_loader:
        optimizer.zero_grad()
        output = model(dbms1, dbms2, dbms3)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(train_loader)
    return total_loss

def valid(model, valid_loader):
    model.eval()

    total_loss = 0.
    outputs = torch.Tensor().cuda()
    with torch.no_grad():
        for dbms1, dbms2, dbms3, target in valid_loader:
            output = model(dbms1, dbms2, dbms3)
            loss = F.mse_loss(output, target)
            total_loss += loss.item()
            outputs = torch.cat((outputs, output))
    total_loss /= len(valid_loader)
    return total_loss, outputs

def addb_train_model(dbms, opt):
    dataset_tr = ADDB_Dataset(dbms.scaled_redis_tr, dbms.scaled_rocksdb_tr, dbms.scaled_spark_tr, dbms.scaled_y_tr)
    dataset_te = ADDB_Dataset(dbms.scaled_redis_te, dbms.scaled_rocksdb_te, dbms.scaled_spark_te, dbms.scaled_y_te)
    loader_tr = DataLoader(dataset=dataset_tr, batch_size=opt.batch_size, shuffle=True)
    loader_te = DataLoader(dataset=dataset_te, batch_size=opt.batch_size, shuffle=True)
    
    # for not using pre-trained weight   
    model = ADDBNet(dbms1_dim=dbms.scaled_redis_tr.shape[-1], 
                    dbms2_dim=dbms.scaled_rocksdb_tr.shape[-1], 
                    dbms3_dim=dbms.scaled_spark_tr.shape[-1],
                    hidden_dim=opt.hidden_dim, 
                    output_dim=dbms.scaled_y_tr.shape[1], 
                    params=[opt.redis_param, opt.rocksdb_param, opt.spark_param]).cuda()
       
    logging.info('##########TRAINING START##########')
    best_loss = 100
    name = get_filename('model_save', opt.dbms, '.pt')
    for epoch in range(opt.epochs):
        loss_tr = train(model, loader_tr, opt.lr)
        loss_te, outputs = valid(model, loader_te)
        
        logging.info(f"[{epoch:02d}/{opt.epochs}] loss_tr: {loss_tr:.8f}\tloss_te:{loss_te:.8f}")
        
        if best_loss > loss_te and epoch>15:
            best_loss = loss_te
            best_model = model
            best_outputs = outputs
            torch.save(best_model, os.path.join('model_save', name))
    logging.info(f"loss is {best_loss:.4f}, save model to {os.path.join('model_save', name)}")
    return best_model, best_outputs