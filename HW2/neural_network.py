import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive
import time
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardsScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# set seed 
np.random.seed(123)

RHC = mlrose_hiive.random_hill_climb
SA = mlrose_hiive.simulated_annealing
GA = mlrose_hiive.genetic_alg
MIMIC = mlrose_hiive.mimic
NN = mlrose_hiive.NeuralNetwork

FlipFlop =  mlrose_hiive.FlipFlop() # GA
ContinuousPeaks = mlrose_hiive.ContinuousPeaks() # SA
FourPeaks = mlrose_hiive.FourPeaks() #MIMIC

exp_decay = mlrose_hiive.ExpDecay()
arith_decay = mlrose_hiive.ArithDecay()
geom_decay = mlrose_hiive.GeomDecay()
DiscreteOpt = mlrose_hiive.DiscreteOpt

seed = 123
bc = load_breast_cancer()
X = bc.data
y = bc.target
df = pd.DataFrame(data=X, columns=bc.feature_names)
df['label'] = y

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = seed)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

alpha = [0.00001, 0.0001, 0.01, 0.1]
restarts = [2, 5, 10, 15, 20]
schedules = [exp_decay, arith_decay, geom_decay]
pop_size = [50, 100, 150, 200, 250, 300, 350]

# Accuracy Initialization 
train_acc_rhc, val_acc_rhc, test_acc_rhc = np.zeros((len(alpha),len(restarts))),  np.zeros((len(alpha),len(restarts))), np.zeros((len(alpha),len(restarts)))
train_acc_sa, val_acc_sa, test_acc_sa = np.zeros((len(alpha),len(schedules))), np.zeros((len(alpha),len(schedules))), np.zeros((len(alpha),len(schedules)))
train_acc_ga, val_acc_ga, test_acc_ga = np.zeros((len(alpha),len(pop_size))), np.zeros((len(alpha),len(pop_size))),  np.zeros((len(alpha),len(pop_size)))
train_acc_bp, val_acc_bp, test_acc_bp = np.zeros((len(alpha),1)), np.zeros((len(alpha),1)), np.zeros((len(alpha),1))

# Index Initialization
rhc_best_idx1, rhc_best_idx2 = 0, 0
sa_best_idx1, sa_best_idx2 = 0, 0
ga_best_idx1, ga_best_idx2 = 0,0 
bp_best_idx1, bp_best_idx2  = 0, 0

# Best Accuracy Initialization 
val_acc_rhc_best = 0.0
val_acc_sa_best = 0.0
val_acc_ga_best = 0.0
val_acc_bp_best = 0.0

# Time Initialization
time_rhc = np.zeros((len(alpha),len(restarts)))
time_sa = np.zeros((len(alpha),len(schedules)))
time_ga = np.zeros((len(alpha),len(pop_size)))
time_bp = np.zeros((len(alpha),1))


# Random Hill Climb
print('RHC Hyperparameter Search Begins...')
nn_model_rhc_best = NN(hidden_nodes = [16], activation ='relu', algorithm ='random_hill_climb', 
					max_iters = 2000, max_attempts = 100,learning_rate = 0.00001, 
					bias = True, is_classifier = True, early_stopping = True, 
					random_state = seed, curve = True)

for i, lr in enumerate(alpha):
	for j, restart in enumerate(restarts):
		nn_model_rhc = NN(hidden_nodes = [16], activation ='relu', algorithm ='random_hill_climb', 
						max_iters = 2000, max_attempts = 100, learning_rate = lr,
						early_stopping = True, bias = True, restarts = restart, is_classifier = True, 
						random_state = seed, curve = True)

		start = time.time()
		nn_model_rhc.fit(X_train, y_train)
		end = time.time()
		time_rhc_current = end - start

		y_train_pred_rhc = nn_model_rhc.predict(X_train)
		#f1-score
		y_train_accuracy_rhc = accuracy_score(y_train, y_train_pred_rhc)
		train_acc_rhc[i][j] = y_train_accuracy_rhc

		y_val_pred_rhc = nn_model_rhc.predict(X_val)
		y_val_accuracy_rhc = accuracy_score(y_val, y_val_pred_rhc)
		val_acc_rhc[i][j]= y_val_accuracy_rhc

		y_test_pred_rhc = nn_model_rhc.predict(X_test)
		y_test_accuracy_rhc = accuracy_score(y_test, y_test_pred_rhc)
		test_acc_rhc[i][j] = y_test_accuracy_rhc
		time_rhc[i][j]= time_rhc_current

		if y_val_accuracy_rhc > val_acc_rhc_best:
			nn_model_rhc_best = nn_model_rhc
			print("Learning Rate:", lr)
			print("Restarts:", restart)
			print("Time:", time_rhc_current)
			rhc_best_idx1 = i
			rhc_best_idx2 = j
			val_acc_rhc_best = y_val_accuracy_rhc

	print("RHC Iteratiion Completed")

# Simulated Annealing
print('SA Hyperparameter Search Begins...')
nn_model_sa_best = NN(hidden_nodes = [16], activation ='relu', algorithm ='simulated_annealing', 
					max_iters = 2000, learning_rate = 0.00001, max_attempts = 100,
					early_stopping = True, bias = True, is_classifier = True, 
					random_state = seed, curve = True)

for i, lr in enumerate(alpha):
	for j, schedule in enumerate(schedules):
		nn_model_sa = NN(hidden_nodes = [16], activation ='relu', algorithm ='simulated_annealing', 
						max_iters = 2000, max_attempts = 100, learning_rate = lr,
						early_stopping = True, bias = True, schedule = schedule, is_classifier = True, 
						random_state = seed, curve = True)

		start = time.time()
		nn_model_sa.fit(X_train, y_train)
		end = time.time()
		time_sa_current = end - start

		y_train_pred_sa = nn_model_sa.predict(X_train)
		#f1-score
		y_train_accuracy_sa = accuracy_score(y_train, y_train_pred_sa)
		train_acc_sa[i][j] = y_train_accuracy_sa

		y_val_pred_sa = nn_model_sa.predict(X_val)
		y_val_accuracy_sa = accuracy_score(y_val, y_val_pred_sa)
		val_acc_sa[i][j]= y_val_accuracy_sa

		y_test_pred_sa = nn_model_sa.predict(X_test)
		y_test_accuracy_sa = accuracy_score(y_test, y_test_pred_sa)
		test_acc_sa[i][j] = y_test_accuracy_sa
		time_sa[i][j]= time_sa_current

		if y_val_accuracy_sa > val_acc_sa_best:
			nn_model_sa_best = nn_model_sa
			print("Learning Rate:", lr)
			print("Schedule:", j)
			print("Time:", time_sa_current)
			sa_best_idx1 = i
			sa_best_idx2 = j
			val_acc_sa_best = y_val_accuracy_sa

	print("SA Iteratiion Completed")


# Genetic Algorithm 
print('GA Hyperparameter Search Begins...')
nn_model_ga_best = NN(hidden_nodes = [16], activation ='relu', algorithm ='genetic_alg', 
					max_iters = 2000, max_attempts = 100, learning_rate = 0.00001,
					bias = True, is_classifier = True,  early_stopping = True, 
					random_state = seed, curve = True)

for i, lr in enumerate(alpha):
	for j, population in enumerate(pop_size):
		nn_model_ga = NN(hidden_nodes = [16], activation ='relu', algorithm ='genetic_alg', 
						max_iters = 2000, max_attempts = 100, learning_rate = lr,
						bias = True, pop_size = population, is_classifier = True, early_stopping = True, 
						random_state = seed, curve = True)

		start = time.time()
		nn_model_ga.fit(X_train, y_train)
		end = time.time()
		time_ga_current = end - start

		y_train_pred_ga = nn_model_ga.predict(X_train)
		#f1-score
		y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
		train_acc_ga[i][j] = y_train_accuracy_ga

		y_val_pred_ga = nn_model_ga.predict(X_val)
		y_val_accuracy_ga = accuracy_score(y_val, y_val_pred_ga)
		val_acc_ga[i][j]= y_val_accuracy_ga

		y_test_pred_ga = nn_model_ga.predict(X_test)
		y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
		test_acc_ga[i][j] = y_test_accuracy_ga
		time_ga[i][j]= time_ga_current

		if y_val_accuracy_ga > val_acc_ga_best:
			nn_model_ga_best = nn_model_ga
			print("Learning Rate:", lr)
			print("Population:", population)
			print("Time:", time_ga_current)
			ga_best_idx1 = i
			ga_best_idx2 = j
			val_acc_ga_best = y_val_accuracy_ga

	print("GA Iteratiion Completed")


# Back Propagation
print('BP Hyperparameter Search Begins...')
nn_model_bp_best = NN(hidden_nodes = [16], activation ='relu', algorithm ='gradient_descent', 
		            max_iters = 2000,  max_attempts = 100,learning_rate = 0.00001,
		            early_stopping = True,  bias = True, is_classifier = True, random_state = seed, curve = True)

for i, lr in enumerate(alpha):
	for j in range(1):
		nn_model_bp = NN(hidden_nodes = [16], activation ='relu', algorithm ='gradient_descent', 
						max_iters = 2000,  max_attempts = 100,learning_rate = lr,
		            	early_stopping = True,  bias = True, is_classifier = True, random_state = seed, curve = True)

		start = time.time()
		nn_model_bp.fit(X_train, y_train)
		end = time.time()
		time_bp_current = end - start
		#f1-score
		y_train_pred_bp = nn_model_bp.predict(X_train)
		y_train_accuracy_bp = accuracy_score(y_train, y_train_pred_bp)
		train_acc_bp[i][j] = y_train_accuracy_bp

		y_val_pred_bp = nn_model_bp.predict(X_val)
		y_val_accuracy_bp = accuracy_score(y_val, y_val_pred_bp)
		val_acc_bp[i][j]= y_val_accuracy_bp

		y_test_pred_bp = nn_model_bp.predict(X_test)
		y_test_accuracy_bp = accuracy_score(y_test, y_test_pred_bp)
		test_acc_bp[i][j] = y_test_accuracy_bp
		time_bp[i][j]= time_bp_current

		if y_val_accuracy_bp > val_acc_bp_best:
			nn_model_bp_best = nn_model_bp
			print("Learning Rate:", lr)
			print("Time:", time_bp_current)
			bp_best_idx1 = i
			bp_best_idx2 = j
			val_acc_bp_best = y_val_accuracy_bp

		print("BP Iteratiion Completed")


print('RHC Result')
y_test_pred_rhc = nn_model_rhc_best.predict(X_test)
confusion_matrix_rhc = confusion_matrix(y_test, y_test_pred_rhc)
print("Confusion Matrix:", confusion_matrix_rhc)
print("Test Accuracy:", test_acc_rhc[rhc_best_idx1][rhc_best_idx2])
print("Average Time", np.mean(time_rhc))
print("Time:", time_rhc)

print('SA Result')
y_test_pred_sa = nn_model_sa_best.predict(X_test)
confusion_matrix_sa = confusion_matrix(y_test, y_test_pred_sa)
print("Confusion Matrix:", confusion_matrix_sa)
print("Test Accuracy:", test_acc_sa[sa_best_idx1][sa_best_idx2])
print("Average Time", np.mean(time_sa))
print("Time:", time_sa)

print('GA Result')
y_test_pred_ga = nn_model_ga_best.predict(X_test)
confusion_matrix_ga = confusion_matrix(y_test, y_test_pred_ga)
print("Confusion Matrix:", confusion_matrix_ga)
print("Test Accuracy:", test_acc_ga[ga_best_idx1][ga_best_idx2])
print("Average Time", np.mean(time_ga))
print("Time:", time_ga)

print('BP Result')
y_test_pred_bp = nn_model_bp_best.predict(X_test)
confusion_matrix_bp = confusion_matrix(y_test, y_test_pred_bp)
print("Confusion Matrix:", confusion_matrix_bp)
print("Test Accuracy:", test_acc_bp[bp_best_idx1][bp_best_idx2])
print("Average Time", np.mean(time_bp))
print("Time:", time_bp)


# Plots
print('Plotting Begins...')

#### Loss Curve ####
plt.figure()
plt.grid(axis ='x')
plt.plot(nn_model_rhc_best.fitness_curve[:,0], color='cornflowerblue', label='RHC', alpha = 0.8)
plt.plot(nn_model_sa_best.fitness_curve[:,0], color= 'rosybrown', label='SA')
plt.plot(nn_model_ga_best.fitness_curve[:,0],  color = 'darkseagreen', label = 'GA')
plt.plot(-nn_model_bp_best.fitness_curve, color = 'indianred', label = 'NN', alpha = 0.9)
plt.grid()
plt.legend(loc='upper right', fancybox=True, shadow=True, ncol=1)
plt.title('Loss Curve of Best Models')
plt.xlabel('Iterations')
plt.ylabel('Training Loss')
plt.savefig('nn_loss.png',   bbox_inches="tight")
plt.close()

#### Time Comparison ####
plt.figure()
plt.bar(['RHC', 'SA', 'GA', 'BP'], [time_rhc[rhc_best_idx1][rhc_best_idx2], time_sa[sa_best_idx1][sa_best_idx2], time_ga[ga_best_idx1][ga_best_idx2], time_bp[bp_best_idx1][bp_best_idx2]], alpha =0.7)
plt.xlabel("Algorithms")
plt.ylabel("Time")
plt.title('Time Comparison')
plt.savefig('nn_best_time_comparison.png', bbox_inches="tight")
plt.close()

#### Score Comparison ####
fig = plt.figure()
test_sets = np.array([test_acc_rhc[rhc_best_idx1][rhc_best_idx2], test_acc_sa[sa_best_idx1][sa_best_idx2], test_acc_ga[ga_best_idx1][ga_best_idx2], test_acc_bp[bp_best_idx1][bp_best_idx2]])
train_sets = np.array([train_acc_rhc[rhc_best_idx1][rhc_best_idx2], train_acc_sa[sa_best_idx1][sa_best_idx2], train_acc_ga[ga_best_idx1][ga_best_idx2], train_acc_bp[bp_best_idx1][bp_best_idx2]])
val_sets = np.array([val_acc_rhc[rhc_best_idx1][rhc_best_idx2], val_acc_sa[sa_best_idx1][sa_best_idx2], val_acc_ga[ga_best_idx1][ga_best_idx2], val_acc_bp[bp_best_idx1][bp_best_idx2]])

ax = fig.add_axes([0.1, 0.5, 0.7, 0.7])
ax.grid(axis='y')
ax.plot(test_sets, color='r', alpha = 0.6, marker = 'o', label = 'Best Test Score')
ax.plot(train_sets, color='b', alpha = 0.6, marker = 'o', label = 'Best Train Score')
ax.plot(val_sets, color='g', alpha = 0.6, marker = 'o', label = 'Best Val Score')
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['RHC','SA','GA','BP'])
plt.ylabel("Best Scores")
plt.legend(loc='bottom right', fancybox=True, shadow=True, ncol=1)
plt.title('Algorithms Comparisons')
plt.savefig('nn_best_score_comparison.png', bbox_inches="tight")
plt.close(fig)

#### Neural Network with Different Algorithms ####
print('Neural Network with Different Opt Begins...')
test_sizes = [0.99, 0.9, 0.75, 0.5, 0.25, 0.1, 0.01]

train_f1_ga_lrc = []
test_f1_ga_lrc = []
train_f1_bp_lrc = []
test_f1_bp_lrc = []
train_f1_sa_lrc = []
test_f1_sa_lrc = []
train_f1_rhc_lrc = []
test_f1_rhc_lrc = []

train_acc_ga_lrc = []
test_acc_ga_lrc = []
train_acc_bp_lrc = []
test_acc_bp_lrc = []
train_acc_sa_lrc = []
test_acc_sa_lrc = []
train_acc_rhc_lrc = []
test_acc_rhc_lrc = []


for i in test_sizes:
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = i, random_state = seed)

	nn_model_rhc = NN(hidden_nodes = [16], activation ='relu', algorithm ='random_hill_climb', 
						max_iters = 2000, max_attempts = 100, restarts = restarts[rhc_best_idx2], learning_rate = alpha[rhc_best_idx1],  
						bias = True, early_stopping = True,  is_classifier = True, curve = True, random_state = seed)

	nn_model_sa = NN(hidden_nodes = [16], activation ='relu', algorithm ='simulated_annealing', 
						max_iters = 2000, max_attempts = 100, restarts = restarts[rhc_best_idx2], learning_rate = alpha[rhc_best_idx1],  
						bias = True, early_stopping = True,  is_classifier = True, curve = True, random_state = seed)

	nn_model_ga = NN(hidden_nodes = [16], activation ='relu', algorithm ='genetic_alg', 
						max_iters = 2000,max_attempts = 100, learning_rate = alpha[ga_best_idx1], pop_size = pop_size[ga_best_idx2], 
						is_classifier = True, early_stopping = True,  bias = True,curve = True, random_state = seed)

	nn_model_bp = NN(hidden_nodes = [16], activation ='relu', algorithm ='gradient_descent', 
						max_iters = 2000, max_attempts = 100, learning_rate = alpha[bp_best_idx1], 
						early_stopping = True, bias = True, is_classifier = True, curve = True, random_state = seed)

	#### RHC ####
	nn_model_rhc.fit(X_train, y_train)
	y_train_pred_rhc = nn_model_rhc.predict(X_train)
	# train acc, f1-score
	y_train_accuracy_rhc = accuracy_score(y_train, y_train_pred_rhc)
	y_train_f1_rhc = f1_score(y_train, y_train_pred_rhc)
	train_acc_rhc_lrc.append(y_train_accuracy_rhc)
	train_f1_rhc_lrc.append(y_train_f1_rhc)
	# test acc, f1-score
	y_test_pred_rhc = nn_model_rhc.predict(X_test)
	y_test_accuracy_rhc = accuracy_score(y_test, y_test_pred_rhc)
	y_test_f1_rhc = f1_score(y_test, y_test_pred_rhc)
	test_acc_rhc_lrc.append(y_test_accuracy_rhc)
	test_f1_rhc_lrc.append(y_test_f1_rhc)

	#### SA ####
	nn_model_sa.fit(X_train, y_train)
	y_train_pred_sa = nn_model_sa.predict(X_train)
	# train acc, f1-score
	y_train_accuracy_sa = accuracy_score(y_train, y_train_pred_sa)
	y_train_f1_sa = f1_score(y_train, y_train_pred_sa)
	train_acc_sa_lrc.append(y_train_accuracy_sa)
	train_f1_sa_lrc.append(y_train_f1_sa)
	# test acc, f1-score
	y_test_pred_sa = nn_model_sa.predict(X_test)
	y_test_accuracy_sa = accuracy_score(y_test, y_test_pred_sa)
	y_test_f1_sa = f1_score(y_test, y_test_pred_sa)
	test_acc_sa_lrc.append(y_test_accuracy_sa)
	test_f1_sa_lrc.append(y_test_f1_sa)

	#### GA ####
	nn_model_ga.fit(X_train, y_train)
	# train acc, f1-score
	y_train_pred_ga = nn_model_ga.predict(X_train)
	y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
	y_train_f1_ga = f1_score(y_train, y_train_pred_ga)
	train_acc_ga_lrc.append(y_train_accuracy_ga)
	train_f1_ga_lrc.append(y_train_f1_ga)
	# test acc, f1-score
	y_test_pred_ga = nn_model_ga.predict(X_test)
	y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
	y_test_f1_ga = f1_score(y_test, y_test_pred_ga)
	test_acc_ga_lrc.append(y_test_accuracy_ga)
	test_f1_ga_lrc.append(y_test_f1_ga)

	#### BP ####
	nn_model_bp.fit(X_train, y_train)
	y_train_pred_bp = nn_model_bp.predict(X_train)
	# train acc, f1-score
	y_train_accuracy_bp = accuracy_score(y_train, y_train_pred_bp)
	y_train_f1_bp = f1_score(y_train, y_train_pred_bp)
	train_acc_bp_lrc.append(y_train_accuracy_bp)
	train_f1_bp_lrc.append(y_train_f1_bp)
	# test acc, f1-score
	y_test_pred_bp = nn_model_bp.predict(X_test)
	y_test_accuracy_bp = accuracy_score(y_test, y_test_pred_bp)
	y_test_f1_bp = f1_score(y_test, y_test_pred_bp)
	test_acc_bp_lrc.append(y_test_accuracy_bp)
	test_f1_bp_lrc.append(y_test_f1_bp)

train_sizes = [1 - i for i in test_sizes]

print('Plotting Begins...')
plt.figure(1)
plt.grid(axis='x')
plt.plot(train_sizes, train_acc_rhc_lrc, label = 'Training Set', color = 'b', alpha = 0.7)
plt.plot(train_sizes, test_acc_rhc_lrc, label = 'Test Set', color = 'r', alpha = 0.6)
plt.grid()
plt.legend()
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('RHC Learning Curve')
plt.savefig('nn_RHC_acc.png', bbox_inches="tight")
plt.close()

plt.figure(2)
plt.grid(axis='x')
plt.plot(train_sizes, train_acc_sa_lrc, label = 'Training Set', color = 'b', alpha = 0.7)
plt.plot(train_sizes, test_acc_sa_lrc, label = 'Test Set', color = 'r', alpha = 0.6)
plt.grid()
plt.legend()
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('SA Learning Curve')
plt.savefig('nn_SA_acc.png', bbox_inches="tight")
plt.close()

plt.figure(3)
plt.grid(axis='x')
plt.plot(train_sizes, train_acc_ga_lrc, label = 'Training Set', color = 'b', alpha = 0.7)
plt.plot(train_sizes, test_acc_ga_lrc, label = 'Test Set', color = 'r', alpha = 0.6)
plt.grid()
plt.legend()
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('GA Learning Curve')
plt.savefig('nn_GA_acc.png', bbox_inches="tight")
plt.close()

plt.figure(4)
plt.grid(axis='x')
plt.plot(train_sizes, train_acc_bp_lrc, label = 'Training Set', color = 'b', alpha = 0.7)
plt.plot(train_sizes, test_acc_bp_lrc, label = 'Test Set', color = 'r', alpha = 0.6)
plt.grid()
plt.legend()
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('BP Learning Curve')
plt.savefig('nn_BP_acc.png')
plt.close()

plt.figure(5)
plt.grid(axis='x')
plt.plot(train_sizes, train_f1_rhc_lrc, label = 'Training Set', color = 'b', alpha = 0.7)
plt.plot(train_sizes, test_f1_rhc_lrc, label = 'Test Set', color = 'r', alpha = 0.6)
plt.grid()
plt.legend()
plt.xlabel('Training Set Size (%)')
plt.ylabel('F1-score')
plt.title('RHC Learning Curve')
plt.savefig('nn_RHC_f1.png', bbox_inches="tight")
plt.close()

plt.figure(6)
plt.grid(axis='x')
plt.plot(train_sizes, train_f1_sa_lrc, label = 'Training Set', color = 'b', alpha = 0.7)
plt.plot(train_sizes, test_f1_sa_lrc, label = 'Test Set', color = 'r', alpha = 0.6)
plt.grid()
plt.legend()
plt.xlabel('Training Set Size (%)')
plt.ylabel('F1-score')
plt.title('SA Learning Curve')
plt.savefig('nn_SA_f1.png', bbox_inches="tight")
plt.close()

plt.figure(7)
plt.grid(axis='x')
plt.plot(train_sizes, train_f1_ga_lrc, label = 'Training Set', color = 'b', alpha = 0.7)
plt.plot(train_sizes, test_f1_ga_lrc, label = 'Test Set', color = 'r', alpha = 0.6)
plt.grid()
plt.legend()
plt.xlabel('Training Set Size (%)')
plt.ylabel('F1-score')
plt.title('GA Learning Curve')
plt.savefig('nn_GA_f1.png', bbox_inches="tight")
plt.close()

plt.figure(8)
plt.grid(axis='x')
plt.plot(train_sizes, train_f1_bp_lrc, label = 'Training Set', color = 'b', alpha = 0.7)
plt.plot(train_sizes, test_f1_bp_lrc, label = 'Test Set', color = 'r', alpha = 0.6)
plt.grid()
plt.legend()
plt.xlabel('Training Set Size (%)')
plt.ylabel('F1-score')
plt.title('BP Learning Curve')
plt.savefig('nn_BP_f1.png', bbox_inches="tight")
plt.close()