import torch
import numpy as np
import torch.nn as nn


def masked_mse(preds, labels, null_val=np.nan, params=None):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan, params=None):
    return torch.sqrt(masked_mse(preds, labels, null_val=null_val, params=params))


def masked_mae(preds, labels, null_val=np.nan, params=None):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse

def get_metrics(args, device, dataloader, engine, scaler, phase, ep=None):
    outputs = []
    realy = torch.Tensor(dataloader[f'y_{phase}']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for _, (x, _, v) in enumerate(dataloader[f'{phase}_loader'].get_iterator()):
        phasex = torch.Tensor(x[..., :-1]).to(device)
        phasex = phasex.transpose(1, 3)
        
        phasev = torch.Tensor(v[..., :-1, :]).to(device)
        phasev = phasev.transpose(1, 3)        
        with torch.no_grad():
            if args.use_neighbors:
                phasex = engine.enrich_(phasex, phasev)
            phasex = nn.functional.pad(phasex, (1, 0, 0, 0))
            preds, _ = engine.model(phasex)
            preds = preds.transpose(1, 3)           
        outputs.append(preds.squeeze())
        
        del phasex, phasev, preds

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    for i in range(args.output_horizon):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = metric(pred, real)
        if phase == "test":
            log = '{} - dataset: {}. Best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(args.model_name, args.dataset_name, i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        
    return amae, amape, armse

