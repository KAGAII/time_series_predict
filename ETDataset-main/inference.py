import torch.nn as nn
import torch
import utils
from new_train_transformer import single_sample

def run_encoder_decoder_inference(
    model: nn.Module, 
    src: torch.Tensor, 
    forecast_window: int,
    batch_size: int,
    device,
    batch_first: bool=False,
    scaler_model = None,
ture_y = None) -> torch.Tensor:
    # Dimension of a batched model input that contains the target sequence values
    target_seq_dim = 0 if batch_first == False else 1
    # Take the last value of thetarget variable in all batches in src and make it tgt
    # as per the Influenza paper
    tgt = src[-1, :, :] if batch_first == False else src[:, -1, :] # shape [1, batch_size, features]
    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_size == 1 and batch_first == False:
        tgt = tgt.unsqueeze(0) # change from [1] to [1, 1, 1]
    if batch_first == False and batch_size > 1:
        tgt = tgt.unsqueeze(0)
    for _ in range(forecast_window-1):
        dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]
        dim_b = src.shape[1] if batch_first == True else src.shape[0]
        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_a
            ).to(device)
        src_mask = utils.generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_b
            ).to(device)
        prediction = model(src.to(device), tgt.to(device), src_mask, tgt_mask)
        if scaler_model is not None:
            single_sample(model,src,tgt,ture_y,scaler_model,device)
        if batch_first == False:
            last_predicted_value = prediction[-1, :, :]
            last_predicted_value = last_predicted_value.unsqueeze(0)
        else:
            last_predicted_value = prediction[:, -1, :]
            last_predicted_value = last_predicted_value.unsqueeze(-1)
        tgt = torch.cat((tgt, last_predicted_value.detach()), target_seq_dim)
    dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]
    dim_b = src.shape[1] if batch_first == True else src.shape[0]
    tgt_mask = utils.generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_a
        ).to(device)
    src_mask = utils.generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_b
        ).to(device)
    final_prediction = model(src.to(device), tgt.to(device), src_mask, tgt_mask)
    return final_prediction


def my_inference(
model: nn.Module,
    src: torch.Tensor,
    forecast_window: int,
    device,
)-> torch.Tensor:


    tgt = torch.zeros((src.shape[0], 1, src.shape[2])).to(src.device)

    for _ in range(forecast_window-1):
        y_predict = model((src, tgt))
        last_predicted_value = y_predict[:, -1, :]
        last_predicted_value = last_predicted_value.reshape(src.shape[0],1,src.shape[2])
        tgt = torch.cat((tgt, last_predicted_value.detach()), 1)
    final_prediction = model((src, tgt))
    return final_prediction