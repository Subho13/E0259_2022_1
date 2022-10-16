import pandas as pd
import math

from scipy.optimize import minimize
import numpy as np

from matplotlib import pyplot as plt

def y_pred(l_z0, u, w):
    z0 = l_z0[w]
    return z0 * (1 - math.exp(-(l_z0[0] * u) / z0))

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2)

def total_mse_loss(z_list, mean_df):
    total_loss = 0
    count = 0

    overs_rem = list(range(1, 51))
    wickets_rem = list(range(1, 11))

    for w in wickets_rem:
        for u in overs_rem:
            y_true = mean_df[w][u]
            if y_true != y_true:
                continue
            y_predicted = y_pred(z_list, u, w)
            t_loss = mse_loss(y_true, y_predicted)
            total_loss += t_loss
            count += 1
    total_loss /= count

    # print(total_loss)
    return total_loss

# def calc_predicted(l_z0, mean_df):
#     overs_rem = list(range(1, 51))
#     wickets_rem = list(range(1, 11))
#     predicted_df = pd.DataFrame(columns=wickets_rem, index=overs_rem)

#     for wicket_rem in wickets_rem:
#         for over_rem in overs_rem:
#             predicted_y = y_pred(l_z0, over_rem, wicket_rem)
#             predicted_df[wicket_rem][over_rem] = predicted_y
#             if mean_df[wicket_rem][over_rem] != mean_df[wicket_rem][over_rem]:
#                 mean_df[wicket_rem][over_rem] = predicted_y
    
#     return predicted_df

def calc_mean(innings_df):
    unique_matches = innings_df.Match.unique()
    # print('Total matches:', len(unique_matches))
    incomplete_matches = []
    for match in unique_matches:
        if len(innings_df[innings_df['Match'] == match]['Over']) < 50 and 0 not in innings_df[innings_df['Match']==match]['Wickets.in.Hand']:
            incomplete_matches.append(match)
    # print('Incomplete matches:', len(incomplete_matches))
    overs_rem = list(range(1, 51))
    wickets_rem = list(range(1, 11))
    mean_df = pd.DataFrame(columns=wickets_rem, index=overs_rem)

    for wicket_rem in wickets_rem:
        wicket_df = innings_df[innings_df['Wickets.in.Hand'] == wicket_rem]
        for over_rem in overs_rem:
            over_df = wicket_df[wicket_df['Over'] == over_rem]
            # complete_over_df = over_df[~over_df['Match'].isin(incomplete_matches)]
            mean_df[wicket_rem][50 - over_rem] = np.float64(over_df['Runs.Remaining'].mean())
            # mean_df[wicket_rem][50 - over_rem] = np.float64(complete_over_df['Runs.Remaining'].mean())
    
    return mean_df

def guess_Z0(innings_df, initial_L):
    unique_full_matches = innings_df[innings_df['Wickets.in.Hand'] == 0].Match.unique()
    
    if len(unique_full_matches) > 10:
        cumsum = [0] * 11
        for ufm in unique_full_matches:
            full_match = innings_df[innings_df.Match == ufm]
            csum = 0
            for w in range(11):
                csum += full_match[full_match['Wickets.in.Hand'] == w]['Runs'].sum()
                cumsum[w] += csum
        
        cumsum = [cumsum[i]/len(unique_full_matches) for i in range(11)]
        cumsum[0] = initial_L
        return cumsum
    else:
        return [10] + [10 * i for i in range(1, 11)]

def print_params(loss, z_0):
    print('Final loss:', loss)
    print('Final parameter values')
    print('\tL =', z_0[0])
    for i in range(1, 11):
        print(f'\tZ0({i}) =', z_0[i])

def plot_z(params, mean_data, save=False):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(1, 11):
        y_predicted = []
        for j in range(1, 51):
            y_predicted.append(y_pred(params, j, i))
        plt.plot(mean_data[i], '.', color=colors[i - 1], alpha=0.4)
        plt.plot(y_predicted, color=colors[i - 1], label=f'{i} wickets')

    plt.xlabel('Overs')
    plt.ylabel('Runs')
    plt.legend(loc='upper left')
    if save:
        plt.savefig('plot.png')
    plt.show()

def main():
    filename = "../data/04_cricket_1999to2011.csv"
    innings = 1
    initial_L = 10
    df = pd.read_csv(filename)

    first_innings = df[df.Innings == innings]

    mean_data = calc_mean(first_innings)
    # Attempt 1:
    # z_list = [10] + [100] * 10
    # Attempt 2:
    # z_list = [20] + [20 * i for i in range(10)]
    # Attempt 3: # final attempt
    z_list = guess_Z0(first_innings, initial_L)
    
    optimization_result = minimize(total_mse_loss, z_list, mean_data, "L-BFGS-B")
    print()

    opt_param = optimization_result.x

    print_params(optimization_result.fun, opt_param)

    plot_z(opt_param, mean_data)

if __name__ == "__main__":
    main()
