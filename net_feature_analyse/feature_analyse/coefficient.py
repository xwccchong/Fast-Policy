import numpy as np
import torch
import torch.nn.functional as F

def get_weight():
    low_rate = torch.tensor([12.59602544, 0.049902365, 0.135760576])
    high_rate = torch.tensor([64.28881402, 0.029787302, 0.15705597])

    low_rate_normalized = (low_rate - low_rate.min()) / (low_rate.max() - low_rate.min())
    high_rate_normalized = (high_rate - high_rate.min()) / (high_rate.max() - high_rate.min())

    low_rate_softmax = F.softmax(low_rate_normalized, dim = 0)
    high_rate_softmax = F.softmax(high_rate_normalized, dim = 0)

    weight = (low_rate_softmax + high_rate_softmax) / 2

    print(f"weight:{weight}")
    
    return weight

def get_key_steps(Mse, key_step_num=0):
        if isinstance(Mse[0], list) and isinstance(Mse[0][0], torch.Tensor):    
            Mse_cpu = [[tensor.cpu().numpy() for tensor in sublist] for sublist in Mse]
        length = len(Mse[0])
        # weights = [0.33, 0.33, 0.33] # uniform
        # weights = get_weight()
        weights = [1.0000e+00, 1.7793e-06, 1.9388e-06] 
        coefficients_array = np.array([
            [weights[0]] * length, 
            [weights[1]] * length, 
            [weights[2]] * length  
        ])
        block_mse = np.array([Mse_cpu[1], Mse_cpu[2], Mse_cpu[4]])
        score = (coefficients_array * block_mse).sum(axis=0)
        
        top_indices = np.argpartition(-score, key_step_num-1)[:key_step_num-1] 
        top_indices = np.sort(top_indices)
        key_steps = [0] 
        key_steps.extend(top_indices+1) 
        if len(key_steps) != key_step_num:
            raise Exception("key steps number is wrong!")
        print("key steps :", key_steps)
        print("final score:", score[top_indices].sum())
        
        return None
    
def get_key_steps_edm(Mse, key_step_num=0):
        if isinstance(Mse[0], list) and isinstance(Mse[0][0], torch.Tensor):    
            Mse_cpu = [[tensor.cpu().numpy() for tensor in sublist] for sublist in Mse]
        
        score = np.array(Mse_cpu[1])
        top_indices = np.argpartition(-score, key_step_num-1)[:key_step_num-1] 
        top_indices = np.sort(top_indices)
        key_steps = [0] 
        key_steps.extend(top_indices+1) 
        if len(key_steps) != key_step_num:
            raise Exception("key steps number is wrong!")
        print("key steps :", key_steps)
        print("final score:", score[top_indices].sum())
        
        return None

#ddim compensate coefficient
def noise_compensate_ddim_c():
    low_cc = [1, 1.0065590212174718, 0.9719612672058621, 0.9404256311568802, 0.9232494001306547, 0.891076559067838, 0.8634302347314536, 0.8293282745928686, 0.7548860498040626, 0.648648997365932]
    high_cc = [1, 0.9668299197319893, 0.9490387671363641, 0.9262911523076228, 0.9010784310721689, 0.8630548585957734, 0.8153314771158023, 0.7460979841022282, 0.6232315316002098, 0.41715442797841384]
    return low_cc, high_cc