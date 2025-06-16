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

def get_key_steps(Mse, key_step_num=0, ):
        if isinstance(Mse[0], list) and isinstance(Mse[0][0], torch.Tensor):    
            Mse_cpu = [[tensor.cpu().numpy() for tensor in sublist] for sublist in Mse]
        length = len(Mse[0])
        # weights = get_weight()
        weights = [0.58, 0.21, 0.21]
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

#ddim compensate coefficient
def noise_compensate_ddim_c(task,scheduler):
    ddim_cc_dict = {
        "pusht":[1, 0.967372992358088, 0.9183999205562677, 0.8509045780354169, 0.7670308265832512, 0.6623484762121691, 0.5406070915739991, 0.40427488921458055, 0.25336995340637064, 0.10828046135112909],
        "square_ph":[1, 0.9654486609370859, 0.9085120875363274, 0.829667903287964, 0.7308038361170475, 0.6145224190821152, 0.48434015887070103, 0.3436576437426259, 0.1958615596361816, 0.046236714205944614],
        "square_mh":[1, 0.9635750154789829, 0.9062177060391342, 0.8284876407283884, 0.7329490290428146, 0.6216260200751392, 0.4986843890819672, 0.3688891124178089, 0.241124156831777, 0.16048653204137467],
        "transport_ph":[1, 0.9654486609370859, 0.9085120875363274, 0.829667903287964, 0.7308038361170475, 0.6145224190821152, 0.48434015887070103, 0.3436576437426259, 0.1958615596361816, 0.046236714205944614],
        "transport_mh":[1, 0.9662133816174774, 0.9086376176315676, 0.8293345992159408, 0.7300893849034723, 0.6135937325924685, 0.48284562994117153, 0.3411105322105926, 0.19248648817461703, 0.047255824290651],
        "tool_hang_ph":[1, 0.9661663735289296, 0.9088168131995709, 0.8303298030108494, 0.7323137127165533, 0.6177617069834799, 0.4905027741451927, 0.3536454483988956, 0.21215604335445015, 0.08386648575800568],
    }
    edm_cc_dict = {
        "pusht":[1, 0.9715441738693511, 1.0387600910417323, 1.0351575743468828, 1.0040125215441082, 0.8472682574711835, 0.4693656692898124],
        "square_ph":[1, 1.0816881207266908, 1.0764621128621068, 1.0649364249188575, 1.012975023242285, 0.7707197432120864, 0.522305132014321],
        "can_ph":[1, 1.1263191648154525, 1.1771458409459397, 1.140360501074368, 1.0592833240192867, 0.7678507546384084, 0.36412172613430205],
        "transport_ph":[1, 0.9433520141061649, 0.9383254549792279, 0.9252504525192543, 0.8768964039697822, 0.7359867265763865, 0.45700242693400633],
        "lift_ph":[1, 1.0413035840813096, 1.0818368344596785, 1.024504709369788, 0.9451862282396652, 0.6811853325624964, 0.31582870283523606],
        # "tool_hang_ph":[1, 0.9853486286947621, 0.9706902301630319, 0.9848463352997981, 0.9542190026557305, 0.7666411760142209, 0.5041022104512617],
        "tool_hang_ph":[1, 1.0273342349696013, 1.0339344227321958, 1.0323374823069216, 1.0281137691916338, 0.8406188581731552, 0.28001837472144003],
    }
    
    if scheduler == "DDIM":
        cc = ddim_cc_dict[task]
    elif scheduler == "EDM":
        cc = edm_cc_dict[task]
        
    return cc



def get_noise_compensate():
    noise_delta_low = np.array([0.013161103,
                                -0.056018997,
                                -0.110645298,
                                -0.124951876,
                                -0.148625663,
                                -0.145801227,
                                -0.133352507,
                                -0.126362385,
                                -0.096969026]
                                )

    noise_delta_high = np.array([-0.065239906,
                                -0.092845435,
                                -0.119538985,
                                -0.13584827,
                                -0.149644443,
                                -0.146458272,
                                -0.130567085,
                                -0.098873579,
                                -0.001316772]
                                )
    sign_low = np.sign(noise_delta_low)
    sign_high = np.sign(noise_delta_high)
    
    sqrt_abs_low = np.sqrt(np.abs(noise_delta_low))
    sqrt_abs_high = np.sqrt(np.abs(noise_delta_high))
    
    low_c = sign_low*sqrt_abs_low
    high_c = sign_high*sqrt_abs_high
    
    print(f"low_c:{low_c}")
    print(f"high_c:{high_c}")
    
    noise_cc = (low_c + high_c) / 2
    
    return low_c, high_c, noise_cc
