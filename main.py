# %%
#BANK LENDING DECISION OPTIMIZATION
#..................Parameter init
from random import random
from random import randint
import math

D=60
K=0.15
loan_size=[10, 25, 4, 11, 18, 3, 17, 15, 9, 10]         #loan size
int_rate=[0.021, 0.022, 0.021, 0.027, 0.025, 0.026, 0.023, 0.021, 0.028, 0.022]
rating=["AAA","BB","A","AA","BBB","AAA","BB","AAA","A","A"]
loss= [.0002, 0.0058, 0.001, .0003, .0024, .0002, .0058, .0002, .001, .001]

pop_size=60
no_of_iter=60
crossoverp=0.8
mutationp=0.006

# %%
#...........................Calculate fitness for single sol
def calc_fitness(x):
    rt=0.01
    rd=0.009
    loan_revenue=0
    loan_cost=0
    trans_cost=0
    loss_lambda=0
    beta=rd*D
    for i in range(len(loan_size)):
        loan_revenue=loan_revenue+ (int_rate[i]*loan_size[i]-loss[i])*x[i]
        #loan_cost=loan_cost+(loan_size[i]*delta)*x[i]
        T=(1-K)*D-loan_size[i]
        trans_cost=trans_cost+rt*T*x[i]
        #trans_cost=trans_cost+int_rate[i]*T
        loss_lambda=loss_lambda+(loss[i])*x[i]
    F=loan_revenue+trans_cost-beta-loss_lambda
    return F

# %%
# ......................Check RGC
def check_sol(x):
    total_loan=0
    for i in range(len(x)):
        total_loan=total_loan+x[i]*loan_size[i]
    if total_loan <= (1-K)*D:
        return 1
    return 0

# %% 
# ....................Randomly Generate a single chromosome
def generate_chromosome():
    chromosome=[]

    for _ in range(len(loan_size)):
        chromosome.append(randint(0,1))
    return chromosome

# %% 
# .................Generate a pool of chromosomes
def init_population():
    i=0
    parent_gen=[]
    while i<pop_size:
        temp=generate_chromosome()
        flag=check_sol(temp)
        if flag==1:
            i=i+1
            parent_gen.append(temp)

    return parent_gen


# %%
# .................Crossover
def crossover(p1,p2):
    r=random()
    if r<crossoverp:
        i=randint(1,len(p1)-1)
        c1=p1[:i]+p2[i:]
        c2=p2[:i]+p1[i:]
        return [c1, c2]
    return [p1, p2]

# %%
# .................Mutation
def mutation(ch):
    for i in range(len(ch)):
        r=random()
        if r<mutationp:
            if ch[i]==1:
                ch[i]=0
            else:
                ch[i]=1
    return ch

# %%
# ................Roulette wheel selection 
def roulette_selection(parent_gen):
    fit=[]
    for x in parent_gen:
        fit.append(calc_fitness(x))
    
    cummu_fit=[fit[0]]
    for i in range(1,len(fit)):
        val=fit[i]
        cummu_fit.append(val+cummu_fit[-1])
    cumm_prob=[]
    for val in cummu_fit:
        cumm_prob.append(val/cummu_fit[-1])
    mating_pop=[]
    
    for _ in range(len(parent_gen)):
        r=random()
        for i in range(len(cumm_prob)):
            if r<cumm_prob[i]:
                index=i
                break
        mating_pop.append(parent_gen[index])
    return mating_pop

# %%
# ..............Search the best sol in a gen
def best_sol_in_gen(gen):
    fit=[]
    for x in parent_gen:
        fit.append(calc_fitness(x))
    gen_best=fit[0]
    index=0
    for i in range(1,len(fit)):
        if fit[i]>gen_best:
            index=i
            gen_best=fit[i]
    return [gen[index], gen_best]
    

# %%
# ..................Genetic Algorithm Main

parent_gen=init_population()
g =best_sol_in_gen(parent_gen)
global_best_sol=g[0]
global_best_val=g[1]
curr_best_sol=global_best_sol
curr_best_val=global_best_val
runs=0
ga_history=[]                   #........stores optimal sol for every iteration
while runs<no_of_iter:
    runs=runs+1
    mating_pop=roulette_selection(parent_gen)
    next_gen=[]
    while len(next_gen)<pop_size:
        r1=randint(0,pop_size-1)
        r2=randint(0,pop_size-1)
        c= crossover(mating_pop[r1], mating_pop[r2])
        m1= mutation(c[0])
        m2= mutation(c[1])
        flag=check_sol(m1)
        if flag:
            next_gen.append(m1)
        flag=check_sol(m2)
        if flag:
            next_gen.append(m2)
    
    next_gen=next_gen[:pop_size]
    temp= best_sol_in_gen(next_gen)
    curr_best_sol=temp[0]
    curr_best_val=temp[1]
    if curr_best_val> global_best_val:
        global_best_val=curr_best_val
        global_best_sol=curr_best_sol
    ga_history.append(global_best_val)
    parent_gen=next_gen

print("GA Best Solution: ",global_best_sol)
print("GA Best Solution value: ",global_best_val)

# %%
# ................. New Sol from prev sol Simulated annealing
def newSolution(prev_sol):
    flag=1
    while flag:
        new_sol=prev_sol
        r1=randint(0,len(prev_sol)-1)
        r2=randint(0,len(prev_sol)-1)
        if new_sol[r1]== 1:
            new_sol[r1]=0
        else:
            new_sol[r1]=1
        if new_sol[r2]== 1:
            new_sol[r2]=0
        else:
            new_sol[r2]=1
        temp= check_sol(new_sol)
        if temp==1:
            break
    return new_sol

# %%
# ................Simulated Annealing 
n_iter= no_of_iter
tmax=1000
tmin=0
delta_t=(tmax-tmin)/n_iter
# delta_t=50
nk=50
sa_history=[]            #..........stores optimal value for every iteration
# initial Solution
flag=1
while flag:
    curr_sol=generate_chromosome()
    temp= check_sol(curr_sol)
    if temp==1:
        break
gs_best=calc_fitness(curr_sol)
gs_sol=curr_sol

while tmax>tmin:

    for i in range(nk):
        new_sol=newSolution(curr_sol)
        fnew= calc_fitness(new_sol)
        fcurr= calc_fitness(curr_sol)
        if fnew > fcurr:
            curr_sol=new_sol
        elif math.exp( (fnew-fcurr)/tmax ) > random():
            curr_sol=new_sol
        curr_best_val=calc_fitness(curr_sol)
        if gs_best < curr_best_val:
            gs_best=curr_best_val
            gs_sol=curr_sol
    sa_history.append(gs_best)
    tmax=tmax-delta_t
print("SA Best solution: ",gs_sol)
print("SA Best solution Value: ",gs_best)

# %%............Plots 
import matplotlib.pyplot as plt

xaxis=[i for i in range(1,no_of_iter+1)]

x1 = xaxis
y1 = ga_history[:no_of_iter]
plt.plot(x1, y1, label = "Genetic Algorithm")
  
x2 = xaxis
y2 = sa_history[:no_of_iter]
plt.plot(x2, y2, label = "Simulated Annealing")
  
# naming the x axis
plt.xlabel('No. of iterations')
# naming the y axis
plt.ylabel('Best Solution')
# giving a title to my graph
plt.title('Genetic Algorithm vs Simulated Annealing')

plt.legend()
plt.show()

# %%
