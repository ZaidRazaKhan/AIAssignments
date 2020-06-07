import numpy as np
from scipy.stats import norm
import random

def compare_matrix(matrix, str):
    return matrix


def mean(matrix, dim):
    return matrix

def multiply(matrix, factor):
    return matrix

def RunPOMDP(input, params):
    learning_rate = params[0]
    extra_reward_value = params[1]
    belief_noise_std = params[2]
    stim_trials = input.stim_trials
    extra_reward = input.extra_reward_trials
    # set run numbers
    iteration_n = 21
    trial_length = len(stim_trials)
    # initialize varialbe for speed
    action = [["" for i in range(trial_length)] for j in range(iteration_n)]
    ql = np.zeros(shape=(trial_length, iteration_n))
    qr = np.zeros(shape=(trial_length, iteration_n))
    prediction_error = np.zeros(shape=(trial_length, iteration_n))
    for iter in iteration_n:
        # initalize q values for each iteration
        qll[0,:] = 1
        qrr[0,:] = 1
        qlr[0,:] = 0
        qrl[0,:] = 0

        # start model
        for trial in trial_length:
            # set contrast
            current_stim = stim_trials(trial)
            # add sensory noise
            stim_with_noise = current_stim + belief_noise_std
            # calculate belief
            belief_l = norm.cdf(0, stim_with_noise, belief_noise_std)
            belief_r = 1-belief_l

            # initialize q values for this iteration
            ql[trial][iter] = multiply(qll, belief_l) + multiply(qrr, belief_r)
            qe[trial][iter] = multiply(qlr, belief_l) + multiply(qrl, belief_r)

            # action <---- max(ql, qr)
            if ql[trial][iter]> qr[trial][iter]:
                action[trial][iter] = 'left'
            elif ql[trial][iter] < qr[trial][iter]:
                action[trial][iter] = 'right'
            else:
                if random.random() >= 0.5:
                    action[trial][iter] = 'right'
                else:
                    action[trial][iter] = 'left'
            
            # trial reward for action chosen by agent
            if current_stim <0 and action[trial][iter] == 'left':
                if extra_reward[trial] == 'left':
                    reward = 1+ extra_reward_value
                elif extra_reward[trial] == 'right':
                    reward = 1
                else:
                    reward = 1
            elif current_stim>0 and action[trial][iter] == 'right':
                if extra_reward[trial] == 'right':
                    reward = 1+ extra_reward_value
                elif extra_reward[trial] == 'left':
                    reward = 1
                else:
                    reward = 1
            elif current_stim == 0:
                if random.random()>0.5:
                    if action[trial][iter] == 'left':
                        if extra_reward[trial] == 'left':
                            reward = 1+ extra_reward_value
                        else:
                            reward = 1
                    elif action[trial][iter] == 'right':
                        if extra_reward[trial] == 'right':
                            reward = 1+ extra_reward_value
                        else:
                            reward = 1
                else:
                    reward = 0
            # calculate prediction_error and update q values
            if action[trial][iter] == 'left':
                prediction_error[trial][iter] = reward-ql[trial][iter]
                qll = qll + learning_rate * prediction_error[trial][iter] * belief_l
                qrl = qrl + learning_rate * prediction_error[trial][iter] * belief_r
            else:
                prediction_error[trial][iter] = reward-qr[trial][iter]
                qlr = qlr + learning_rate * prediction_error[trial][iter] * belief_l
                qrr = qrr + learning_rate * prediction_error[trial][iter] * belief_r
    
    action_left = compare_matrix(action, 'left')
    action_right = compare_matrix(action, 'right')
    mean_action_num = mean(action_right-action_left,1)
    output = {'action':mean_action_num, 'ql':mean(ql,1), 'qr':mean(qr,1)}
    return output



