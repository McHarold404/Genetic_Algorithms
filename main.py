#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from client import * 
import numpy as np 
import pickle
from datetime import datetime 
import time
import os


# In[ ]:


key = "11yEVzHqnhB6ufF3DX93dzQ8SvIzTsnINY8bnCQTtmTOYioBVn"
key_ = "blah"
overfit_vector = np.array([0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10])


# In[ ]:


def book_keeping():
    date_today = datetime.today().strftime('%Y-%m-%d')
    cwd = os.getcwd()
    path = f'{cwd}/{date_today}'
    try:
        os.mkdir(path)
    except:
        pass
    return path

path = book_keeping()


# In[ ]:


# logs store all the relevant information about the generation
logs = []
called = 1 


# In[ ]:


# stores the logs of the best vector in every (list with 0 index->next_pop, 1->fitness,2->errors)
best_logs = []


# In[ ]:


#global variables
population = 8
pool_size = 4
n_parents = 0


# In[ ]:


def get_initial_logs(n = population):
    log = {"initial_pop": False, "mutations": False, "parents": [], "next_pop": []}
    population = np.zeros(shape = (n, 11))
    for i in range(n):
        rng = np.random.uniform(low = -0.30, high = 0.30, size=(1, 11))
        #rng = np.random.normal(size=(1, 11))
        population[i, :] = overfit_vector + rng*overfit_vector
        
    fitness_array = [fitness_function(list(v), weight = 1) for v in population]
    fitness = [i[0] for i in fitness_array]
    errors = [i[1] for i in fitness_array]
    
    log["errors"] = errors
    log["fitness"] = fitness
    log["next_pop"] = population
    return log


# In[ ]:


def fitness_function(vector, weight = 1):
    errors = get_errors(key, list(vector))
    #fitness = (abs(errors[1]-5e10) +abs(errors[0]-5e10))
    #fitness = (abs(errors[1]) + weight*(abs(errors[0]))
    fitness = abs(errors[0] - errors[1])
    return fitness, errors

#def fitness_function(vector, weight = 0.75)


# In[ ]:


# Blend Crossover
def crossover(parent1, parent2):
    alpha = 0.3
    child = np.zeros(11)
    for i in range(11):
        max_p = max(parent2[i], parent1[i])
        min_p = min(parent2[i], parent1[i])
        low = min_p - alpha*(max_p - min_p)
        high = max_p + alpha*(max_p - min_p)
        child[i] = np.random.uniform(low = low, high = high)
    return child


# In[ ]:


def mutation(vector,prob):
    rng = (np.random.uniform(size = 11)) < prob
    delta = rng*np.random.uniform(low = -0.07, high = 0.07, size = 11)
    vector = vector + delta*vector
    return vector


# In[ ]:


def get_next_pop(initial_pop, fitness, errors, n_parents = n_parents):
    log = {"initial_pop": initial_pop, "mutations": False, "parents": [], "next_pop": []}
    
    triple = sorted(list(zip(initial_pop, fitness, errors)), key = lambda x:  x[1])
    fitness = np.array(sorted(fitness))
    p = [(1/i)/np.sum(1/fitness[:pool_size]) for i in fitness[:pool_size]]
    
    next_fitness = []
    next_errors = []
    next_pop = []
    parents = []
    mutations = []
    
    for _ in range(population-n_parents):
        parent1 = np.array(triple[np.random.choice(range(pool_size), p = p)][0])
        parent2 = np.array(triple[np.random.choice(range(pool_size), p = p)][0])
        parents.append((parent1, parent2))
        
        child = crossover(parent1, parent2)
        if(np.random.uniform() < .5 or (list(parent1) == list(parent2))):
            mutations.append(child)
            child = mutation(child, prob = 1)
        else:
            mutations.append(False)
            
        c_fitness, c_error = fitness_function(child)
        next_fitness.append(c_fitness)
        next_errors.append(c_error)
        next_pop.append(child)
    
    for i in range(n_parents):
        # print(i, initial_pop[i], errors[i])
        next_pop.append(np.array(triple[i][0]))
        next_fitness.append(triple[i][1])
        next_errors.append(triple[i][2])
       
    log["errors"] = next_errors
    log["fitness"] = next_fitness
    log["parents"] = parents
    log["next_pop"] = next_pop
    log["mutations"] = mutations
    
    f = open(f'{path}/{time.time()}.txt', 'wb')
    pickle.dump(log, f)
    return log


# In[ ]:


#function to get best_vectors from every generation (sorting based on validation error and fitness function)
def get_best_vector(new_log):
    #zip vectors(next_pop), fitness, errors into 1 to sort based on validation error
    zip_log = zip(new_log["next_pop"],new_log["fitness"],new_log["errors"])
    temp_log = np.array(sorted(zip_log, key = lambda x : x[1]))
    print('Train Error :',temp_log[0][2][0]/1e11,'1e11')
    print('Validation Error :',temp_log[0][2][1]/1e11,'1e11')
    return temp_log[0] #returns


# In[ ]:


def genetic_algorithm(pop, fitness, errors, n = 1):
    log1 = []
    for _ in range(n):
        log = get_next_pop(pop, fitness, errors)
        pop = np.array(log["next_pop"])
        fitness = log["fitness"]
        errors = log["errors"]
        log1.append(log)
        a = get_best_vector(log)
        print(len(a)) 
        best_logs.append(a)
        
    return log1


# ### To start with new vectors

# In[ ]:


# If training has to start from the initial given overfit vector
if(called == 1):
    new_logs = get_initial_logs(population)
    logs.append(new_logs)
    fitness_gens.append(np.median(new_logs["fitness"]))
    best_errors.append(sorted(new_logs["errors"], key = lambda x: x[1])[0])
    next_pop, fitness, errors = logs[-1]["next_pop"], logs[-1]["fitness"], logs[-1]["errors"]
    called = 0
    print(best_errors)
    print(fitness_gens)
    print("Done")


# In[ ]:


# to load logs from a text file and train further (not from the start)
next_pop, fitness, errors = np.array(logs[-1]["next_pop"]), logs[-1]["fitness"], logs[-1]["errors"]


# In[ ]:


new_logs = genetic_algorithm(next_pop, fitness, errors,1)


# In[ ]:


logs.extend(new_logs)

