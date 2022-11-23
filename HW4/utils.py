import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import hiive.mdptoolbox as mdptoolbox
from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning


#############################
########### PLOTS ###########
#############################

def plot_lake(env, policy=None, title='Frozen Lake'):
    colors = {
    b'S': 'silver',
    b'F': 'w',
    b'H': 'maroon',
    b'G': 'gold'
      }

    directions = {
                0: '←',
                1: '↓',
                2: '→',
                3: '↑'
                  }
    squares = env.nrow
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, xlim=(-.01, squares+0.01), ylim=(-.01, squares+0.01))
    plt.title(title, fontsize=16, weight='bold', y=1.01)
    for i in range(squares):
        for j in range(squares):
            y = squares - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1, linewidth=1, edgecolor='k')
            p.set_facecolor(colors[env.desc[i,j]])
            ax.add_patch(p)
            
            if policy is not None:
                text = ax.text(x+0.5, y+0.5, directions[policy[i, j]],
                               horizontalalignment='center', size=25, verticalalignment='center',
                               color='k')
            
    plt.axis('off')
    
def plot_forest(policy, title='Forest'):
    colors = {
    0: 'yellowgreen',
    1: 'saddlebrown'
}

    labels = {
        0: 'W',
        1: 'C',
    }

    rows = 25
    cols = 25
    
    # reshape policy array to be 2-D - assumes 500 states...
    policy = np.array(list(policy)).reshape(rows,cols)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, xlim=(-.01, cols+0.01), ylim = (-.01, rows+0.01))
    plt.title(title, fontsize=16, weight='bold', y=1.01)
    
    for i in range(25):
        for j in range(25):
            y = 25 - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1, linewidth=1, edgecolor='k')
            p.set_facecolor(colors[policy[i,j]])
            ax.add_patch(p)
            
            text = ax.text(x+0.5, y+0.5, labels[policy[i, j]],
                           horizontalalignment='center', size=10, verticalalignment='center', color='w')
    
    plt.axis('off')

    
def ql_plot_result(df, interest, dependent, independent, title=None, logscale=False):

    if dependent not in interest:
        print('Dependent variable not available')
        return
    if independent not in interest:
        print('Independent variable not available')
        return
    
    x = np.unique(df[dependent])
    y = []
    
    for i in x:
        y.append(df.loc[df[dependent] == i][independent].mean())
        
    fig = plt.figure(figsize=(6,4))
    plt.plot(x, y, 'o-', color = 'skyblue')
    
    if title == None:
        title = independent + ' vs. ' + dependent
    plt.title(title, fontsize=15)
    plt.xlabel(dependent)
    plt.ylabel(independent)
    plt.grid(True)
    if logscale:
        plt.xscale('log')
        

def result_plots(model, target_name, value_iter, gammas, policy_iter = None):
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(hspace= 5, wspace= 5)
    fig, axs = plt.subplots(2,2)
    fig.suptitle(target_name, y=1)
    color = ['lightblue','darksalmon', 'darkkhaki', 'thistle', 'sandybrown', 'mediumaquamarine']

    x = gammas
    if model ==  'VI':
        y = []
        for g in gammas:
            y.append(value_iter.loc[value_iter['gamma'] == g]['average_steps'].mean())


        y_2 = []
        for g in gammas:
            y_2.append(value_iter.loc[value_iter['gamma'] == g]['success_pct'].mean())
            
        y_3 = []
        for g in gammas:
            y_3.append(value_iter.loc[value_iter['gamma'] == g]['time'].mean())
            
        y_4 = []
        for g in gammas:
            y_4.append(value_iter.loc[value_iter['gamma'] == g]['iterations'].mean())

            
        sns.set(style="whitegrid")
        sns.set(rc={'figure.figsize':(8, 10)})
        
        sns.set(style="whitegrid")
        sns.barplot(x, y, ax=axs[0,0], palette = color)
        axs[0,0].set_title('Average Steps')
        axs[0,0].set_xlabel('Gamma')
        axs[0,0].set_ylabel('Average Steps')

        sns.set(style="whitegrid")
        sns.barplot(x, y_2, ax=axs[0,1], palette = color)
        axs[0,1].set_title('Success Ratio')
        axs[0,1].set_xlabel('Gamma')
        axs[0,1].set_ylabel('Success Pct (%)')

        sns.set(style="whitegrid")
        sns.barplot(x, y_3, ax=axs[1,0], palette = color)
        axs[1,0].set_title('Average Time')
        axs[1,0].set_xlabel('Gamma')
        axs[1,0].set_ylabel('Average Time')

        sns.set(style="whitegrid")
        sns.barplot(x, y_4, ax=axs[1,1], palette = color)
        axs[1,1].set_title('Average Iterations')
        axs[1,1].set_xlabel('Gamma')
        axs[1,1].set_ylabel('Average Iterations')
        fig.tight_layout()
        
        print('-'*50)
        print('Avg_Steps \n',  y)
        print('-'*50)
        print('Avg_Success \n', y_2)
        print('-'*50)
        print('Avg_time \n', y_3)
        print('-'*50)
        print('Avg_iter \n', y_4)
        print('-'*50)
        
        
    else:
        y = []
        for g in gammas:
            y.append(policy_iter.loc[policy_iter['gamma'] == g]['average_steps'].mean())


        y_2 = []
        for g in gammas:
            y_2.append(policy_iter.loc[policy_iter['gamma'] == g]['success_pct'].mean())
            
        y_3 = []
        for g in gammas:
            y_3.append(policy_iter.loc[policy_iter['gamma'] == g]['time'].mean())
            
        y_4 = []
        for g in gammas:
            y_4.append(policy_iter.loc[policy_iter['gamma'] == g]['iterations'].mean())

            
        sns.set(rc={'figure.figsize':(8, 10)})
        
        sns.lineplot(x, y, ax=axs[0,0], color = 'skyblue', marker="o")
        axs[0,0].set_title('Average Steps')
        axs[0,0].set_xlabel('Gamma')
        axs[0,0].set_ylabel('Average Steps')

        sns.lineplot(x, y_2, ax=axs[0,1], color = 'skyblue', marker="o")
        axs[0,1].set_title('Success Ratio')
        axs[0,1].set_xlabel('Gamma')
        axs[0,1].set_ylabel('Success Pct (%)')

        sns.lineplot(x, y_3, ax=axs[1,0], color = 'skyblue', marker="o")
        axs[1,0].set_title('Average Time')
        axs[1,0].set_xlabel('Gamma')
        axs[1,0].set_ylabel('Average Time')

        sns.lineplot(x, y_4, ax=axs[1,1], color = 'skyblue' , marker="o")
        axs[1,1].set_title('Average Iterations')
        axs[1,1].set_xlabel('Gamma')
        axs[1,1].set_ylabel('Average Iterations')
        fig.tight_layout()
        
        print('-'*50)
        print('Avg_Steps \n', y)
        print('-'*50)
        print('Avg_Success \n', y_2)
        print('-'*50)
        print('Avg_time \n', y_3)
        print('-'*50)
        print('Avg_iter \n', y_4)
        print('-'*50)


def result_plots_2(model, target_name, value_iter, gammas, policy_iter = None):
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(hspace= 5, wspace= 5)
    fig, axs = plt.subplots(2,2)
    fig.suptitle(target_name, y=1)
    color = ['lightblue','darksalmon', 'darkkhaki', 'thistle', 'sandybrown', 'mediumaquamarine']

    if model ==  'VI':
        y_2 = []
        for g in gammas:
            y_2.append(value_iter.loc[value_iter['gamma'] == g]['reward'].mean())
            
        y_3 = []
        for g in gammas:
            y_3.append(value_iter.loc[value_iter['gamma'] == g]['time'].mean())
            
        y_4 = []
        for g in gammas:
            y_4.append(value_iter.loc[value_iter['gamma'] == g]['iterations'].mean())

            
        sns.set(style="whitegrid")
        sns.set(rc={'figure.figsize':(8, 10)})
        
        sns.set(style="whitegrid")
        sns.lineplot(x=value_iter['iterations'], y=value_iter['reward'], color = 'darkseagreen', marker = 'o', ax=axs[0,0])
        axs[0,0].set_title('Rewards vs. Iterations')
        axs[0,0].set_xlabel('Iterations')
        axs[0,0].set_ylabel('Rewards')

        x = gammas
        sns.set(style="whitegrid")
        sns.barplot(x, y_2, ax=axs[0,1], palette = color)
        axs[0,1].set_title('Rewards vs. Gamma')
        axs[0,1].set_xlabel('Gamma')
        axs[0,1].set_ylabel('Average Rewards')

        sns.set(style="whitegrid")
        sns.barplot(x, y_3, ax=axs[1,0], palette = color)
        axs[1,0].set_title('Average Time')
        axs[1,0].set_xlabel('Gamma')
        axs[1,0].set_ylabel('Average Time')

        sns.set(style="whitegrid")
        sns.barplot(x, y_4, ax=axs[1,1], palette = color)
        axs[1,1].set_title('Average Iterations')
        axs[1,1].set_xlabel('Gamma')
        axs[1,1].set_ylabel('Average Iterations')
        fig.tight_layout()
        
        # print('-'*50)
        # print('Avg_Steps \n',  y)
        print('-'*50)
        print('Avg_Rewards_vs_Gamma \n', y_2)
        print('-'*50)
        print('Avg_time \n', y_3)
        print('-'*50)
        print('Avg_iter \n', y_4)
        print('-'*50)
        
        
    else:
        y_2 = []
        for g in gammas:
            y_2.append(policy_iter.loc[policy_iter['gamma'] == g]['reward'].mean())
            
        y_3 = []
        for g in gammas:
            y_3.append(policy_iter.loc[policy_iter['gamma'] == g]['time'].mean())
            
        y_4 = []
        for g in gammas:
            y_4.append(policy_iter.loc[policy_iter['gamma'] == g]['iterations'].mean())

            
        sns.set(style="whitegrid")
        sns.set(rc={'figure.figsize':(8, 10)})
        
        sns.set(style="whitegrid")
        sns.lineplot(x=policy_iter['iterations'], y=policy_iter['reward'], color = 'darkseagreen', marker = 'o', ax=axs[0,0])
        axs[0,0].set_title('Rewards vs. Iterations')
        axs[0,0].set_xlabel('Iterations')
        axs[0,0].set_ylabel('Rewards')

        x = gammas
        sns.set(style="whitegrid")
        sns.barplot(x, y_2, ax=axs[0,1], palette = color)
        axs[0,1].set_title('Rewards vs. Gamma')
        axs[0,1].set_xlabel('Gamma')
        axs[0,1].set_ylabel('Average Rewards')

        sns.set(style="whitegrid")
        sns.barplot(x, y_3, ax=axs[1,0], palette = color)
        axs[1,0].set_title('Average Time')
        axs[1,0].set_xlabel('Gamma')
        axs[1,0].set_ylabel('Average Time')

        sns.set(style="whitegrid")
        sns.barplot(x, y_4, ax=axs[1,1], palette = color)
        axs[1,1].set_title('Average Iterations')
        axs[1,1].set_xlabel('Gamma')
        axs[1,1].set_ylabel('Average Iterations')
        fig.tight_layout()
        
        # print('-'*50)
        # print('Avg_Steps \n',  y)
        print('-'*50)
        print('Avg_Rewards_vs_Gamma \n', y_2)
        print('-'*50)
        print('Avg_time \n', y_3)
        print('-'*50)
        print('Avg_iter \n', y_4)
        print('-'*50)

#############################
######### Compute  ##########
#############################
def get_score(env, policy, printInfo=False, episodes=1000):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()
        steps=0
        while True:
            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            steps+=1
            if done and reward == 1:
                # print('You have got the Frisbee after {} steps'.format(steps))
                steps_list.append(steps)
                break
            elif done and reward == 0:
                # print("You fell in a hole!")
                misses += 1
                break
    ave_steps = np.mean(steps_list)
    std_steps = np.std(steps_list)
    pct_fail  = (misses/episodes)* 100
    
    if (printInfo):
        print('----------------------------------------------')
        print('You took an average of {:.0f} steps to get the frisbee'.format(ave_steps))
        print('And you fell in the hole {:.2f} % of the times'.format(pct_fail))
        print('----------------------------------------------')
  
    return ave_steps, std_steps, pct_fail

def get_policy(env,stateValue, lmbda=0.9):
    policy = [0 for i in range(env.nS)]
    for state in range(env.nS):
        action_values = []
        for action in range(env.nA):
            action_value = 0
            for i in range(len(env.P[state][action])):
                prob, next_state, r, _ = env.P[state][action][i]
                action_value += prob * (r + lmbda * stateValue[next_state])
            action_values.append(action_value)
        best_action = np.argmax(np.asarray(action_values))
        policy[state] = best_action
    return policy 




#############################
######## Algorithms #########
#############################


def valueIteration(env, rows, cols, t, r, gammas, epsilons, showResults=False, max_iterations=100000):
    t0 = time.time()
    # create data structure to save off
    columns = ['gamma', 'epsilon', 'time', 'iterations', 'reward', 'average_steps', 'steps_stddev', 'success_pct', 'policy', 'mean_rewards', 'max_rewards', 'error']
    data = pd.DataFrame(0.0, index=np.arange(len(gammas)*len(epsilons)), columns=columns)
    
    print('Gamma,\tEps,\tTime,\tIter,\tReward')
    print(80*'_')
    
    testNum = 0
    for g in gammas:
        for e in epsilons:
            test = ValueIteration(t, r, gamma=g, epsilon=e, max_iter=max_iterations)
            
            runs  = test.run()
            Time  = runs[-1]['Time']
            iters = runs[-1]['Iteration']
            maxR  = runs[-1]['Max V']
            
            max_rewards, mean_rewards, errors = [], [], []
            for run in runs:
                max_rewards.append(run['Max V'])
                mean_rewards.append(run['Mean V'])
                errors.append(run['Error'])
            
            policy = np.array(test.policy)
            policy = policy.reshape(4,4)
            
            data['gamma'][testNum]        = g
            data['epsilon'][testNum]      = e
            data['time'][testNum]         = Time
            data['iterations'][testNum]   = iters
            data['reward'][testNum]       = maxR
            data['mean_rewards'][testNum] = {tuple(mean_rewards)}
            data['max_rewards'][testNum]  = {tuple(max_rewards)}
            data['error'][testNum]        = {tuple(errors)}
            data['policy'][testNum]       = {test.policy}
            
            print('%.2f,\t%.0E,\t%.2f,\t%d,\t%f' % (g, e, Time, iters, maxR))
            
            if showResults:
                title = 'FrozenLake_VI_' + str(rows) + 'x' + str(cols) + '_g' + str(g) + '_e' + str(e)
                plot_lake(env, policy, title)
            
            testNum = testNum + 1
                
    endTime = time.time() - t0
    print("Time taken: %.2f" %endTime)
    
    # See differences in policy
    policies = data['policy']
    
    for i,p in enumerate(policies):
        pol = list(p)[0]
        steps, steps_stddev, failures = get_score(env, pol, showResults)
        data['average_steps'][i] = steps
        data['steps_stddev'][i]  = steps_stddev
        data['success_pct'][i]   = 100-failures      
        
    # replace all NaN's
    data.fillna(0, inplace=True)
    data.head()
        
    return data

def valueIteration_v2(t, r, gammas, epsilons, showResults=False, max_iterations=100000):
    t0 = time.time()
    
    # create data structure to save off
    columns = ['gamma', 'epsilon', 'time', 'iterations', 'reward', 'average_steps', 'steps_stddev', 'success_pct', 'policy', 'mean_rewards', 'max_rewards', 'error']
    data = pd.DataFrame(0.0, index=np.arange(len(gammas)*len(epsilons)), columns=columns)
    
    print('Gamma,\tEps,\tTime,\tIter,\tReward')
    print(80*'_')
    
    testNum = 0
    for g in gammas:
        for e in epsilons:
            test = ValueIteration(t, r, gamma=g, epsilon=e, max_iter=max_iterations)
            
            runs  = test.run()
            Time  = runs[-1]['Time']
            iters = runs[-1]['Iteration']
            maxR  = runs[-1]['Max V']
            
            max_rewards, mean_rewards, errors = [], [], []
            for run in runs:
                max_rewards.append(run['Max V'])
                mean_rewards.append(run['Mean V'])
                errors.append(run['Error'])
            
            policy = np.array(test.policy)
            
            data['gamma'][testNum]        = g
            data['epsilon'][testNum]      = e
            data['time'][testNum]         = Time
            data['iterations'][testNum]   = iters
            data['reward'][testNum]       = maxR
            data['mean_rewards'][testNum] = {tuple(mean_rewards)}
            data['max_rewards'][testNum]  = {tuple(max_rewards)}
            data['error'][testNum]        = {tuple(errors)}
            data['policy'][testNum]       = {test.policy}
            
            print('%.2f,\t%.0E,\t%.2f,\t%d,\t%f' % (g, e, Time, iters, maxR))
            
            testNum = testNum + 1
        
    endTime = time.time() - t0
    print('Time taken: %.2f' % endTime)
    
    # See differences in policy
    policies = data['policy']
    
    # replace all NaN's
    data.fillna(0, inplace=True)
    data.head()
        
    return data


def policyIteration(env, rows, cols, t, r, gammas, showResults=False, max_iterations=100000):
    t0 = time.time()
    
    # create data structure to save off
    columns = ['gamma', 'epsilon', 'time', 'iterations', 'reward', 'average_steps', 'steps_stddev', 'success_pct', 'policy', 'mean_rewards', 'max_rewards', 'error']
    data = pd.DataFrame(0.0, index=np.arange(len(gammas)), columns=columns)
    
    print('gamma,\ttime,\titer,\treward')
    print(80*'_')
    
    testnum = 0
    for g in gammas:
        test = PolicyIteration(t, r, gamma=g, max_iter=max_iterations, eval_type="matrix") # eval_type="iterative"
        
        runs  = test.run()
        Time  = test.time
        iters = test.iter
        maxr  = runs[-1]['Max V']
                
        max_rewards, mean_rewards, errors = [], [], []
        for run in runs:
            max_rewards.append(run['Max V'])
            mean_rewards.append(run['Mean V'])
            errors.append(run['Error'])
        
        policy = np.array(test.policy)
        policy = policy.reshape(4,4)
        
        data['gamma'][testnum]        = g
        data['time'][testnum]         = Time
        data['iterations'][testnum]   = iters
        data['reward'][testnum]       = maxr
        data['mean_rewards'][testnum] = {tuple(mean_rewards)}
        data['max_rewards'][testnum]  = {tuple(max_rewards)}
        data['error'][testnum]        = {tuple(errors)}
        data['policy'][testnum]       = {test.policy}
        
        print('%.2f,\t%.2f,\t%d,\t%f' % (g, Time, iters, maxr))
        
        if showResults:
            title = 'frozenlake_pi_' + str(rows) + 'x' + str(cols) + '_g' + str(g)
            plot_lake(env, policy, title)
        
        testnum = testnum + 1
            
    endTime = time.time() - t0
    print('Time taken: %.2f' % endTime)
    
    # see differences in policy
    policies = data['policy']
    
    for i,p in enumerate(policies):
        pol = list(p)[0]
        steps, steps_stddev, failures = get_score(env, pol, showResults)
        data['average_steps'][i] = steps
        data['steps_stddev'][i]  = steps_stddev
        data['success_pct'][i]   = 100-failures      
        
    # replace all nan's
    data.fillna(0, inplace=True)
    data.head()
    
    return data


def policyIteration_v2(t, r, gammas, max_iterations=100000):
    t0 = time.time()
    
    # create data structure to save off
    columns = ['gamma', 'epsilon', 'time', 'iterations', 'reward', 'average_steps', 'steps_stddev', 'success_pct', 'policy', 'mean_rewards', 'max_rewards', 'error']
    data = pd.DataFrame(0.0, index=np.arange(len(gammas)), columns=columns)
    
    print('gamma,\ttime,\titer,\treward')
    print(80*'_')
    
    testnum = 0
    for g in gammas:
        test = PolicyIteration(t, r, gamma=g, max_iter=max_iterations, eval_type="matrix") # eval_type="iterative"
        
        runs  = test.run()
        Time  = test.time
        iters = test.iter
        maxr  = runs[-1]['Max V']
                
        max_rewards, mean_rewards, errors = [], [], []
        for run in runs:
            max_rewards.append(run['Max V'])
            mean_rewards.append(run['Mean V'])
            errors.append(run['Error'])
        
        data['gamma'][testnum]        = g
        data['time'][testnum]         = Time
        data['iterations'][testnum]   = iters
        data['reward'][testnum]       = maxr
        data['mean_rewards'][testnum] = {tuple(mean_rewards)}
        data['max_rewards'][testnum]  = {tuple(max_rewards)}
        data['error'][testnum]        = {tuple(errors)}
        data['policy'][testnum]       = {test.policy}
        
        print('%.2f,\t%.2f,\t%d,\t%f' % (g, Time, iters, maxr))
        
        testnum = testnum + 1
        
    endTime = time.time() - t0
    print('Time taken: %.2f' % endTime)
    
    # see differences in policy
    policies = data['policy']
        
    # replace all nan's
    data.fillna(0, inplace=True)
    data.head()
        
    return data



#10000000
def Q_learning(env, rows, cols, t, r, gammas, alphas, alpha_decays=[0.99], epsilon_decays=[0.99], n_iterations=[10000], showResults=False):
    # create data structure to save off
    columns = ['gamma', 'alpha', 'alpha_decay', 'epsilon_decay', 'Iterations', 'Time', 'Reward', 'average_steps', 'steps_stddev', 'success_pct', 'policy', 'mean_rewards', 'max_rewards', 'error']
    numTests = len(gammas)*len(alphas)*len(alpha_decays)*len(epsilon_decays)*len(n_iterations)
    data = pd.DataFrame(0.0, index=np.arange(numTests), columns=columns)
    
    print('Gamma,\tAlpha,\tTime,\tIter,\tReward')
    print(80*'_')
    
    testNum = 0
    for g in gammas:
        for a in alphas:
            for a_decay in alpha_decays:
                for e_decay in epsilon_decays:
                    for n in n_iterations:
                        print('Test Num %d/%d' %(testNum+1, numTests))
                        print('Gamma: %.2f,\tAlpha: %.2f,\tAlpha Decay:%.3f,\tEpsilon Decay:%.3f,\tIterations:%d' 
                             %(g, a, a_decay, e_decay, n))
                        
                        test = QLearning(t, r, gamma=g, alpha=a, alpha_decay=a_decay, epsilon_decay=e_decay, n_iter=n)
                        
                        runs  = test.run()
                        time  = runs[-1]['Time']
                        iters = runs[-1]['Iteration']
                        maxR  = runs[-1]['Max V']
                        
                        max_rewards, mean_rewards, errors = [], [], []
                        for run in runs:
                            max_rewards.append(run['Max V'])
                            mean_rewards.append(run['Mean V'])
                            errors.append(run['Error'])
                        
                        policy = np.array(test.policy)
                        policy = policy.reshape(4,4)
                        
                        data['gamma'][testNum]         = g
                        data['alpha'][testNum]         = a
                        data['alpha_decay'][testNum]   = a_decay
                        data['epsilon_decay'][testNum] = e_decay
                        data['Time'][testNum]          = time
                        data['Iterations'][testNum]    = iters
                        data['Reward'][testNum]        = maxR
                        data['mean_rewards'][testNum]  = {tuple(mean_rewards)}
                        data['max_rewards'][testNum]   = {tuple(max_rewards)}
                        data['error'][testNum]         = {tuple(errors)}
                        data['policy'][testNum]        = {test.policy}
                        
                        print('%.2f,\t%.2f,\t%.2f,\t%d,\t%f' % (g, a, time, iters, maxR))
                        
                        if showResults:
                            Result = 'FrozenLake_QL_' + str(rows) + 'x' + str(cols) + '_g' + str(g) + '_a' + str(a) + '_adecay' + str(a_decay) + '_edecay' + str(e_decay) + '_iter' + str(n)
                            plot_lake(env, policy, 'Frozen Lake Q-Learning \n Optimal Policy')
                            print('-'*30)
                            print(Result)
                            print('-'*30)
                        
                        testNum = testNum + 1
            
    # See differences in policy
    policies = data['policy']
    
    for i,p in enumerate(policies):
        pol = list(p)[0]
        steps, steps_stddev, failures = get_score(env, pol, showResults)
        data['average_steps'][i] = steps
        data['steps_stddev'][i]  = steps_stddev
        data['success_pct'][i]   = 100-failures      
        
    # replace all NaN's
    data.fillna(0, inplace=True)
    data.head()
        
    return data


#10000000
def Q_learning_v2(t, r, gammas, alphas, alpha_decays=[0.99], epsilon_decays=[0.99], n_iterations=[10000], showResults=False):
    # create data structure to save off
    columns = ['gamma', 'alpha', 'alpha_decay', 'epsilon_decay', 'iterations', 'time', 'reward', 'average_steps', 'steps_stddev', 'success_pct', 'policy', 'mean_rewards', 'max_rewards', 'error']
    numTests = len(gammas)*len(alphas)*len(alpha_decays)*len(epsilon_decays)*len(n_iterations)
    data = pd.DataFrame(0.0, index=np.arange(numTests), columns=columns)
    
    print('Gamma,\tAlpha,\tTime,\tIter,\tReward')
    print(80*'_')
    
    testNum = 0
    for g in gammas:
        for a in alphas:
            for a_decay in alpha_decays:
                for e_decay in epsilon_decays:
                    for n in n_iterations:
                        print('Test Num %d/%d' %(testNum+1, numTests))
                        print('Gamma: %.2f,\tAlpha: %.2f,\tAlpha Decay:%.3f,\tEpsilon Decay:%.3f,\tIterations:%d' 
                             %(g, a, a_decay, e_decay, n))
                        
                        test = QLearning(t, r, gamma=g, alpha=a, alpha_decay=a_decay, epsilon_decay=e_decay, n_iter=n)
                        
                        runs  = test.run()
                        time  = runs[-1]['Time']
                        iters = runs[-1]['Iteration']
                        maxR  = runs[-1]['Max V']
                        
                        max_rewards, mean_rewards, errors = [], [], []
                        for run in runs:
                            max_rewards.append(run['Max V'])
                            mean_rewards.append(run['Mean V'])
                            errors.append(run['Error'])
                        
                        #policy = np.array(test.policy)
                        #policy = policy.reshape(4,4)
                        
                        data['gamma'][testNum]         = g
                        data['alpha'][testNum]         = a
                        data['alpha_decay'][testNum]   = a_decay
                        data['epsilon_decay'][testNum] = e_decay
                        data['time'][testNum]          = time
                        data['iterations'][testNum]    = iters
                        data['reward'][testNum]        = maxR
                        data['mean_rewards'][testNum]  = {tuple(mean_rewards)}
                        data['max_rewards'][testNum]   = {tuple(max_rewards)}
                        data['error'][testNum]         = {tuple(errors)}
                        data['policy'][testNum]        = {test.policy}
                        
                        print('%.2f,\t%.2f,\t%.2f,\t%d,\t%f' % (g, a, time, iters, maxR))
                        
                        if showResults:
                            pass
                        testNum = testNum + 1
            
    # See differences in policy
    policies = data['policy']
    
    '''
    for i,p in enumerate(policies):
        pol = list(p)[0]
        steps, steps_stddev, failures = get_score(env, pol, showResults)
        data['average_steps'][i] = steps
        data['steps_stddev'][i]  = steps_stddev
        data['success_pct'][i]   = 100-failures      
    '''
        
    # replace all NaN's
    data.fillna(0, inplace=True)
    data.head()
        
    return data