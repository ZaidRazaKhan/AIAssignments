import numpy as np
from scipy.stats import norm
import random
import statistics as st


def compare_matrix(matrix, val):
    rown = len(matrix)
    coln = len(matrix[0])
    ref_matrix = []
    for i in range(0, rown):
        dummy = []
        for j in range(0, coln):
            if matrix[i][j] != val:
                dummy.append(float(0))
            else:
                dummy.append(float(1))
        ref_matrix.append(dummy)
    return ref_matrix

def list_subtraction(l1, l2):
    row = len(l1)
    col = len(l1[0])
    subtracted_matrix = []
    for i in range(row):
        temp = []
        for j in range(col):
            temp.append(l1[i][j]-l2[i][j])
        subtracted_matrix.append(temp)
    return subtracted_matrix


def mean(matrix, dim):
    mean_list = []
    if(dim == 0) :
        for i in range(0, len(matrix)):
            mean_list.append(st.mean(matrix[i]))
    else :
        for j in range(0, len(matrix[0])) :
            mean_list.append(st.mean(matrix[:][j]))
    return mean_list


def multiply(matrix, factor):
    res_matrix = []
    for i in range(0, len(matrix)):
        dummy = []
        for j in range(0, len(matrix[0])):
            dummy.append(matrix[i][j] * factor)
        res_matrix.append(dummy)
    return res_matrix


def RunPOMDP(input, params):
    learning_rate = params[0]
    extra_reward_value = params[1]
    belief_noise_std = params[2]
    stim_trials = input["stimTrials"]
    extra_reward = input["extraRewardTrials"]
    # set run numbers
    iteration_n = 21
    trial_length = len(stim_trials)
    # initialize varialbe for speed
    action = [["left" for i in range(iteration_n)] for j in range(trial_length)]
    ql = [[0 for i in range(iteration_n)] for j in range(trial_length)]
    qr = [[0 for i in range(iteration_n)] for j in range(trial_length)]
    prediction_error = np.zeros(shape=(trial_length, iteration_n))
    for iter in range(iteration_n):
        # initalize q values for each iteration
        qll = 1
        qrr = 1
        qlr = 0
        qrl = 0

        # start model
        for trial in range(trial_length):
            # set contrast
            current_stim = stim_trials[trial]
            # add sensory noise
            stim_with_noise = current_stim + belief_noise_std
            # calculate belief
            belief_l = norm.cdf(0, stim_with_noise, belief_noise_std)
            belief_r = 1 - belief_l

            # initialize q values for this iteration
            ql[trial][iter] = qll * belief_l + qrr * belief_r
            qr[trial][iter] = qlr * belief_l + qrl * belief_r

            # action <---- max(ql, qr)
            if ql[trial][iter] > qr[trial][iter]:
                action[trial][iter] = 'left'
            elif ql[trial][iter] < qr[trial][iter]:
                action[trial][iter] = 'right'
            else:
                if random.random() >= 0.5:
                    action[trial][iter] = 'right'
                else:
                    action[trial][iter] = 'left'

            # trial reward for action chosen by agent
            reward = 0
            if current_stim < 0 and action[trial][iter] == 'left':
                if extra_reward[trial] == 'left':
                    reward = 1 + extra_reward_value
                elif extra_reward[trial] == 'right':
                    reward = 1
                else:
                    reward = 1
            elif current_stim > 0 and action[trial][iter] == 'right':
                if extra_reward[trial] == 'right':
                    reward = 1 + extra_reward_value
                elif extra_reward[trial] == 'left':
                    reward = 1
                else:
                    reward = 1
            elif current_stim == 0:
                if random.random() > 0.5:
                    if action[trial][iter] == 'left':
                        if extra_reward[trial] == 'left':
                            reward = 1 + extra_reward_value
                        else:
                            reward = 1
                    elif action[trial][iter] == 'right':
                        if extra_reward[trial] == 'right':
                            reward = 1 + extra_reward_value
                        else:
                            reward = 1
                else:
                    reward = 0
            # calculate prediction_error and update q values
            if action[trial][iter] == 'left':
                prediction_error[trial][iter] = reward - ql[trial][iter]
                qll = qll + learning_rate * prediction_error[trial][iter] * belief_l
                qrl = qrl + learning_rate * prediction_error[trial][iter] * belief_r
            else:
                prediction_error[trial][iter] = reward - qr[trial][iter]
                qlr = qlr + learning_rate * prediction_error[trial][iter] * belief_l
                qrr = qrr + learning_rate * prediction_error[trial][iter] * belief_r

    action_left = compare_matrix(action, 'left')
    action_right = compare_matrix(action, 'right')
    mean_action_num = mean(list_subtraction(action_right, action_left), 1)
    output = {'action': mean_action_num, 'ql': mean(ql, 1), 'qr': mean(qr, 1)}
    return output