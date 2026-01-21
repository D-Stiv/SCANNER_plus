# MIT License
#
# Original work:
# Copyright (c) <2019> <Zonghan Wu>
#
# Modifications:
# Copyright (c) 2026 D-Stiv
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# ----
# Modifications points: 
# - inserted spatio-temporal correlation: lines 82; 84-85

import torch
import numpy as np
import torch.nn as nn

def _masked_loss(loss, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val

    mask = mask.float()
    mask = mask / mask.mean().clamp(min=1e-8)

    loss = loss * mask
    return torch.nan_to_num(loss).mean()

def masked_mse(preds, labels, null_val=np.nan, params=None):
    return _masked_loss((preds - labels) ** 2, labels, null_val)


def masked_rmse(preds, labels, null_val=np.nan, params=None):
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mae(preds, labels, null_val=np.nan, params=None):
    return _masked_loss(torch.abs(preds - labels), labels, null_val)


def masked_mape(preds, labels, null_val=np.nan):
    return _masked_loss(torch.abs(preds - labels) / labels.clamp(min=1e-8),
                        labels, null_val)

def metric(pred, real):
    return (
        masked_mae(pred, real, 0.0).item(),
        masked_mape(pred, real, 0.0).item(),
        masked_rmse(pred, real, 0.0).item()
    )

def get_metrics(args, device, dataloader, engine, scaler, phase, ep=None):
    engine.model.eval()

    realy = torch.as_tensor(dataloader[f'y_{phase}'], device=device)
    realy = realy.transpose(1, 3)[:, 0]  # (B, N, T_out)

    outputs = []

    with torch.no_grad():
        for x, _, v in dataloader[f'{phase}_loader'].get_iterator():
            x = torch.as_tensor(x[..., :-1], device=device).transpose(1, 3)
            v = torch.as_tensor(v[..., :-1, :], device=device).transpose(1, 3)

            if args.use_neighbors:
                x = engine.enrich_(x, v)

            x = nn.functional.pad(x, (1, 0))
            preds, _ = engine.model(x)
            outputs.append(preds.transpose(1, 3).squeeze())

    yhat = torch.cat(outputs)[:realy.size(0)]

    amae, amape, armse = [], [], []

    for h in range(args.output_horizon):
        pred = scaler.inverse_transform(yhat[:, :, h])
        real = realy[:, :, h]

        mae, mape, rmse = metric(pred, real)

        if phase == "test":
            print(
                f"{args.model_name} - dataset: {args.dataset_name}. "
                f"Best model on test data for horizon {h+1}, "
                f"Test MAE: {mae:.4f}, Test MAPE: {mape:.4f}, Test RMSE: {rmse:.4f}"
            )

        amae.append(mae)
        amape.append(mape)
        armse.append(rmse)

    return amae, amape, armse
