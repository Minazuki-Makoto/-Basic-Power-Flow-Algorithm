import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def transformer(k,ZT):
    Z=k*ZT
    y1=(k-1)*ZT/k*ZT
    y2=(1-k)*ZT/k**2*ZT
    Y=1/Z
    G=[np.real(Y),np.real(y1),np.real(y2)]
    B=[np.imag(Y),np.imag(y1),np.imag(y2)]
    return G,B

def department(key):
    for i in range(1,5):
        for j in range(1,5):
            h1=str(i)
            h2=str(j)
            res=h1+'-'+h2
            if key==res:
                return i,j

def theta_relative(theta):
    relative_theta=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            relative_theta[i,j]=theta[i]-theta[j]
    return relative_theta
class Newton_laphson_method():
    def __init__(self,G,B,P_in,P_out,Q_in,Q_out,U_primary,theta_primary,m=2,n=4):#m个PQ节点，n个节点，1个平衡节点
        self.G=G
        self.B=B
        self.P_in=P_in
        self.P_out=P_out
        self.Q_in=Q_in
        self.Q_out=Q_out
        self.U_primary=U_primary
        self.theta_primary=theta_primary
        self.m=m
        self.n=n

    def theta_relative(self,theta):
        n=self.n
        relative_theta=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                relative_theta[i,j]=theta[i]-theta[j]
        return relative_theta

    def power_loss(self,U,theta):
        relative_theta=self.theta_relative(theta)
        P_loss=np.zeros(3)
        Q_loss=np.zeros(3)
        for i in range(3):
            P_loss[i]=self.P_in[i]-self.P_out[i]
            Q_loss[i]=self.Q_in[i]-self.Q_out[i]
            for j in range(4):
                P_loss[i]-=U[i]*U[j]*(self.G[i,j]*np.cos(relative_theta[i,j])+self.B[i,j]*np.sin(relative_theta[i,j]))
                Q_loss[i]-=U[i]*U[j]*(self.G[i,j]*np.sin(relative_theta[i,j])-self.B[i,j]*np.cos(relative_theta[i,j]))
        return P_loss,Q_loss

    def Jacobian(self,U,theta):
        n=self.n
        m=self.m
        P_loss, Q_loss=self.power_loss(U,theta)
        Jacobian_matrix=np.zeros((n+m-1,n+m-1))
        for i in range(n+m-1):
            for j in range(n+m-1):
                if i < n-1:
                    if j< n-1:
                        theta_copy=theta.copy()
                        theta_copy[j]+=1e-8
                        P_loss_delta,_=self.power_loss(U,theta_copy)
                        Jacobian_matrix[i,j]=(P_loss_delta[i]-P_loss[i])/1e-8
                    else:
                        U_copy=U.copy()
                        U_copy[j-n+1]+=1e-8
                        P_loss_delta,_=self.power_loss(U_copy,theta)
                        Jacobian_matrix[i,j]=U[j-n+1]*(P_loss_delta[i]-P_loss[i])/1e-8
                else:
                    if j< n-1:
                        theta_copy=theta.copy()
                        theta_copy[j]+=1e-8
                        _,Q_loss_delta=self.power_loss(U,theta_copy)
                        Jacobian_matrix[i,j]=(Q_loss_delta[i-n+1]-Q_loss[i-n+1])/1e-8
                    else:
                        U_copy=U.copy()
                        U_copy[j-n+1]+=1e-8
                        _,Q_loss_delta=self.power_loss(U_copy,theta)
                        Jacobian_matrix[i,j]=U[j-n+1]*(Q_loss_delta[i-n+1]-Q_loss[i-n+1])/1e-8
        return Jacobian_matrix

    def regeneration(self,U,theta):
        n=self.n
        m=self.m
        U_new=U.copy()
        theta_new=theta.copy()
        P_loss,Q_loss=self.power_loss(U,theta)
        Jacobian_matrix=self.Jacobian(U,theta)
        delta_loss=np.hstack((P_loss[:n-1],Q_loss[:m]))
        delta_U_theta=-np.linalg.inv(Jacobian_matrix)@delta_loss
        for i in range(m+n-1):
            if i <n-1:
                theta_new[i]+=delta_U_theta[i]
            else:
                U_new[i-n+1]+=U[i-n+1]*delta_U_theta[i]
        return U_new ,theta_new

    def run(self):
        n=self.n
        m=self.m
        theta_primary=self.theta_primary
        U_primary=self.U_primary
        for alt in range(10):
            P_loss,Q_loss=self.power_loss(U_primary,theta_primary)
            U_new,theta_new=self.regeneration(U_primary,theta_primary)
            P_loss_alt,Q_loss_alt=self.power_loss(U_new,theta_new)
            delta_loss = np.hstack((P_loss_alt[:n - 1], Q_loss_alt[:m]))
            if np.sum([delta_loss[i]**2 for i in range(n+m-1)])<=1e-15:
                return U_new,theta_new
                break
            else:
                U_primary=U_new
                theta_primary=theta_new
            if alt==9:
                return None,None
def all_loss(U,theta):
    P_dot_loss=[]
    Q_dot_loss=[]
    relative_theta=theta_relative(theta)
    for i in range(len(U)):
        P_i_sum=0
        Q_i_sum=0
        for j in range(len(U)):
            P_i_sum+=U[i]*U[j]*(G[i,j]*np.cos(relative_theta[i,j])+B[i,j]*np.sin(relative_theta[i,j]))
            Q_i_sum+=U[i]*U[j]*(G[i,j]*np.sin(relative_theta[i,j])-B[i,j]*np.cos(relative_theta[i,j]))
        P_dot_loss.append(P_i_sum)
        Q_dot_loss.append(Q_i_sum)
    P_allloss=np.sum(P_dot_loss)
    Q_allloss=np.sum(Q_dot_loss)
    return P_allloss,Q_allloss

class GeneticAlgorithm():
    def __init__(self,popsize,chormlength,G,B,P_in,P_out,Q_in,Q_out,U_primary,theta_primary,pm,pc,m=2,n=4):
        self.popsize=popsize
        self.chormlength=chormlength
        self.G=G
        self.B=B
        self.P_in=P_in
        self.P_out=P_out
        self.Q_in=Q_in
        self.Q_out=Q_out
        self.U_primary=U_primary
        self.theta_primary=theta_primary
        self.pm=pm
        self.pc=pc
        self.m=m
        self.n=n
        self.population=self.init()
        self.value_cache = {}
        self.history_value=[]
        self.best_choice=[]

    def init(self):
        population=[]
        for i in range(self.popsize):
            pop=np.random.randint(0,2,size=self.chormlength)
            population.append(pop)
        return population

    def transform(self,chorm):
        ans=0
        for i in range(1,len(chorm)):
            ans+=chorm[i]*2**(self.chormlength-1-i)
        ans=ans/2**19
        if chorm[0]==1:
            ans=-ans
        return ans

    def get_value(self,chorm):
        key=tuple(chorm)
        if key in self.value_cache:
            return self.value_cache[key]
        delta_B=self.transform(chorm)
        B_after=B.copy()
        B_after[1,1]+=delta_B
        waveform=Newton_laphson_method(G,B_after,P_in, P_out, Q_in, Q_out, U_primary, theta_primary)
        U_, theta_=waveform.run()
        P_loss,Q_loss=all_loss(U_,theta_)
        P_loss=np.abs(P_loss)
        if U_==None:
            P_loss=9999999999
        if U_[1]<=0.9 or U_[1]>1.05 or U_[0]<0.95:
            P_loss=9999999999
        self.value_cache[key]=P_loss
        return P_loss

    def get_best_choice(self,population):
        compare=[]
        for i in range(len(population)):
            P_loss=self.get_value(population[i])
            res=1.0/P_loss
            compare.append(res)
        flag=False
        for i in range(len(compare)):
            if compare[i]!=1.0/9999999999:
                flag=True
        if flag==False:
            return self.init()
        else:
            comp=[compare[i]/np.sum(compare) for i in range(len(population))]
            idx=[i for i in range(len(population))]
            idx_list=np.random.choice(idx,size=self.popsize,p=comp)
            new_population=[]
            for i in range(self.popsize):
                new_population.append(population[idx_list[i]])
            return new_population

    def mutation(self, chorm):
        for i in range(2):
            if np.random.rand() < self.pm:
                place = np.random.randint(0, self.chormlength)
                chorm[place] = (chorm[place] + 1) % 2
        return chorm

    def crossover(self,chorm1,chorm2):
        if np.random.rand()<self.pc:
            place=np.random.randint(0,self.chormlength)
            if place<=self.chormlength//2:
                chorm1_copy=chorm1.copy()
                chorm1[:place]=chorm2[:place]
                chorm2[:place]=chorm1_copy[:place]
            else:
                chorm1_copy=chorm1.copy()
                chorm1[place:]=chorm2[place:]
                chorm2[place:]=chorm1_copy[place:]
        return chorm1,chorm2

    def run(self):
        population_primary=self.population
        P_loss_min=99999
        best_choice=0
        for alt in range(2000):
            for i in range(self.popsize):
                P_loss=self.get_value(population_primary[i])
                if P_loss<P_loss_min:
                    P_loss_min=P_loss
                    best_choice=self.transform(population_primary[i])
            self.history_value.append(P_loss_min)
            for i in range(self.popsize):
                population_primary[i]=self.mutation(population_primary[i])
            for i in range(self.popsize):
                for j in range(i+1,self.popsize):
                    population_primary[i],population_primary[j]=self.crossover(population_primary[i],population_primary[j])
            population=self.get_best_choice(population_primary)
            population_primary=population.copy()
        return self.history_value,P_loss_min,best_choice

if __name__ == '__main__':
    r_connection={'1-2':0.5901,'2-3':0.6103,'2-4':0.5130,'3-4':0.7122 }
    x_connenction={'1-2':10.302,'2-3':4.1909,'2-4':1.3330,'3-4':8.2212,'1-1':0.1044,'2-2':0.2793,'3-3':0.3421,'4-4':0.1333 }
    G_transformer,B_transformer=transformer(1.049,complex(0.91,1.28))
    G=np.zeros((4,4))
    B=np.zeros((4,4))
    n=4
    for key in r_connection:
        line,column=department(key)
        G[line-1,column-1]-=r_connection[key]
        G[column-1,line-1]=G[line-1,column-1]
        for k in range(1,5):
            if line==k or column==k:
                G[k-1,k-1]+=r_connection[key]
    for key in x_connenction:
        line,column=department(key)
        B[line-1,column-1]-=x_connenction[key]
        B[column-1,line-1]=B[line-1,column-1]
        for k in range(1,5):
            if line==k or column==k:
                B[k-1,k-1]+=x_connenction[key]

    G[0,0]+=G_transformer[0]+G_transformer[1]
    G[1,1]+=G_transformer[1]+G_transformer[2]
    G[1,2]-=G_transformer[0]
    G[2,1]=G[1,2]
    B[0,0]+=B_transformer[0]+B_transformer[1]
    B[1,1]+=B_transformer[1]+B_transformer[2]
    B[1,2]-=B_transformer[0]
    B[2,1]=B[1,2]
    P_in=[0.0,0.0,1.1,0.0]
    P_out=[1.5,0.0,0.0,0.0]
    Q_in=[0.0,0.0,0.0,0.0]
    Q_out=[0.6,0.0,0.0,0.0]
    U_primary=[1.0,1.0,1.0,1.05]
    theta_primary=[0.0,0.0,0.0,0.0]

    popsize = 50
    chormlength=20
    pm = 0.02
    pc = 0.05
    GA=GeneticAlgorithm(popsize,chormlength,G,B,P_in,P_out,Q_in,Q_out,U_primary,theta_primary,pm,pc,m=2,n=4)
    history_value, P_loss_min, best_choice=GA.run()
    print(f'最小供电负荷为{P_loss_min}')
    print(f'节点2并联电抗器的大小为{best_choice}')
    B+=best_choice
    wave=Newton_laphson_method(G,B,P_in,P_out,Q_in,Q_out,U_primary,theta_primary,m=2,n=4)
    U_best, theta_best=wave.run()
    history_value_min=min(history_value)
    history_value_max=max(history_value)
    print(f'节点1的电压为{U_best[0]}，节点2的电压为{U_best[1]}')
    if U_best[0]>=0.95 and 0.9<U_best[1]<=1.05:
        print('符合电力系统基本要求')
    else:
        print('不符合电力系统基本要求')
    alts=[i for i in range(len(history_value))]
    plt.figure(figsize=(14,10))
    plt.grid(True)
    plt.xlim((0,len(history_value)))
    plt.ylim((history_value_min-0.001,history_value_max+0.001))
    plt.xlabel('迭代次数')
    plt.ylabel('最小值')
    plt.title('遗传算法求最小供电')
    plt.plot(alts,history_value,'r',lw=1.5)
    plt.show()