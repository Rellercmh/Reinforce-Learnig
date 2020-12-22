import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def random_index(rate):
    # this modulue is to return the index selected according to the propability given in the input list
    # exp:[10,20,30,40] the probability of selected for 0,1,2,3 is seperately 10% 20% 30% 40%
    #it was quoted from web

    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index

def Get_action(x,Table):
    '''
    :param x: the current state
    :param Table: Q_sa table
    :return: the action under epsilon soft policy
    '''
    list_action = list(Table[x])
    max_num = max(list_action)
    max_location = list_action.index(max_num)

    p_list = []  # if there is more than zero value in four action ,use traditional epsilon-soft policy
    for i in range(len(list_action)):  # the p_list will like [10,10,10,70] when epsilon is 0.6.
        p = 0
        if i == max_location:
            p = 1 - epso + epso / (len(list_action))
            p = int(p * 100)
            p_list.append(p)

        else:
            p = epso / len(list_action)
            p = int(p * 100)
            p_list.append(p)


    return random_index(p_list)

def Do_action(n):  # given action number ,return direction of action
    '''
    :param n: action index
    :return: adjust signal
    The signal return will finally mutiply with a index 10.
    '''

    if n == 0:
        return 0
    elif n == 1:
        return 0.01
    elif n == 2:
        return -0.01
    elif n == 3:
        return 0.02
    else :
        return -0.02

def test(Tab):
    print('-------------------------------------  start test -------------------------------------------------------')
    agent=prob()
    list_error = []
    list_agent_x=[]
    list_z=[]
    list_z_target=[]
    period=15
    # set test scenairo
    map_test=([13,14,15,16,15,14,13,14,16,17,18,19,20,21,22,21,20,19,18,17,15,16,14,15,16,15,15,16,17])

    while True:
        height = map_test[agent.x // period]
        list_z_target.append(height)

        f = int(round(force_feedback(agent.z - height), 3) * 1000)
        action = Get_action(f, Tab)
        output = agent.signal + Do_action(action) * 10
        agent.operate(output)
        list_z.append(agent.z)
        f_next = int(round(force_feedback(agent.z - height), 3) * 1000)

        list_error.append(np.abs(agent.z - height - agent.setheight))
        agent.x+=1
        list_agent_x.append(agent.x)
        if agent.x == len(map_test) * period :
            break
#______________________________________________________________________________________________________________
    # make chart
    print('sum of error:',np.sum(list_error))
    print('error per unit:',np.sum(list_error)/(period*len(map)))
    plt.scatter(list_agent_x, list_z, s=1)
    plt.scatter(list_agent_x, list_z_target, s=1)
    plt.title(('error per unit:',np.sum(list_error)/(period*len(map))), fontsize=25)
#--------------------------------------------------------------------------------------------------------------
    plt.style.use('seaborn-pastel')

    fig = plt.figure()
    ax = plt.axes(xlim=(0, len(list_agent_x)), ylim=(0, 30))

    line, = ax.plot([], [], ':')
    line2, = ax.plot([], [], ':')

    def init():
        line.set_data([], [])
        line2.set_data([], [])
        return line,line2,

    x=[]
    y=[]
    s=[]
    def animate(i):
        x.append(list_agent_x[i])
        y.append((list_z[i]))
        s.append(list_z_target[i])
        line.set_data(x, y)
        line2.set_data(x,s)
        return line,  line2,

    anim = FuncAnimation(fig, animate,init_func=init,
                         frames=int(len(list_agent_x)), interval=0.1, blit=True,repeat=False)
    # anim.save('T-RL_200_15_p20.gif',writer='imagemagick')
    plt.show()
#___________________________________________________________________________________________________________________

class prob():
    def __init__(self):
        self.x=0
        self.y=0
        self.center=20
        self.z=15.6
        self.setforce=1/25
        self.signal=self.z/2
        self.Hysteresis=0.1
        self.setheight = 5

    def operate(self,signal):
        if self.signal<signal:
            if signal<0.5*(15-self.signal)+self.signal:
                x1=self.signal
                y1=self.z
                x2=0.5*(15-self.signal)+self.signal
                y2=(0.5*(30-self.z)+self.z)*(1-self.Hysteresis)
                a=(y2-y1)/(x2-x1)
                b=y2-(a*x2)
                self.z=a*signal+b
            else:
                x1 = 0.5*(15-self.signal)+self.signal
                y1 = (0.5*(30-self.z)+self.z)*(1-self.Hysteresis)
                x2 = 15
                y2 = 30
                a = (y2 - y1) / (x2 - x1)
                b = y2 - (a * x2)
                self.z = a * signal + b


        if self.signal>signal:
            if signal<0.5*(self.signal-0):
                x1=0.5*self.signal
                y1=0.5*self.z*(1+self.Hysteresis)
                x2=0
                y2=0
                a=(y2-y1)/(x2-x1)
                b=y2-(a*x2)
                self.z=a*signal+b
            else:
                x1 = self.signal
                y1 = self.z
                x2 = 0.5*self.signal
                y2 = 0.5*self.z*(1+self.Hysteresis)
                a = (y2 - y1) / (x2 - x1)
                b = y2 - (a * x2)
                self.z = a * signal + b
        self.signal=signal

def force_feedback(distance):
    return 1/(distance*distance)

map=np.array([13,14,15,16,17,18,19,20,21,22,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,3,4,5,6,5,4,3,5,6,7,8,9,8,7,6,5,4,3])
print(map)

episods = 2170
epso=1
alpha=0.5
threshold=1e-3
gamma=0.5
period = 200                            # period parameter indicate operate speed on x direction higer value means slower
Q_sa=np.ones((1500,5))                  # set Q_sa table  has 1500 state and 5 action ,initial value is 1
Q_sa_estimate=np.copy(Q_sa)
agent=prob()
error_map=[]

for epsiod in range(episods):
    print('New start---------------------------,',epsiod,'/2170---------------------------------')
    agent=prob()                                    #initiate the agent
    print('epsilon is :', epso)
    Q_sa_estimate_updated = np.copy(Q_sa_estimate)

    step=1
    list_error=[]

    while True:

        height = map[agent.x//period]                          # get the current step height


        f=int(round(force_feedback(agent.z-height),3)*1000)    # get current state according to forcefeedback
        # print('f',f)
        if f > 800:
            # the thresh value to anti overflow
            break

        # signal_s=agent.signal
        action=Get_action(f,Q_sa)
        output=agent.signal+Do_action(action)*10
        agent.operate(output)
        # print('agent.z:',agent.z,'height',height,'agent.x',agent.x,'episod',epsiod,'epso',epso)
        f_next=int(round(force_feedback(agent.z-height),3)*1000)
        # reward depends on the forece feedback,closer to standard value,higher reward
        rew=1/(1+np.abs(agent.setforce-force_feedback(agent.z-height)))


        # when force feedback is far from standard value, or the length is out of the working range then break,the reward
        # is -1 update Q_Sa
        if agent.z>30:
            rew=-1
            Q_sa[f][action] = Q_sa[f][action] + (alpha / step) * (rew  - Q_sa[f][action])
            break

        if f_next>= 1200 or f_next<10:
            rew=-1
            Q_sa[f][action] = Q_sa[f][action] + (alpha / step) * (rew - Q_sa[f][action])
            break
        #if not break ,also update the Q_sa
        max_1=max(Q_sa[f_next])
        Q_sa[f][action]=Q_sa[f][action]+(alpha/step)*(rew+gamma*max_1-Q_sa[f][action])


        list_error.append(np.abs(agent.z - height-agent.setheight))  #put distance error into error list.
        step += 1
        agent.x+=1

        if (agent.x+1)%period==0:   #when comes to new step ,the alpha value should be initialized
            step=1


        if agent.x ==len(map)*period-1:
            break

    Q_sa_estimate = np.copy(Q_sa)
    error = np.sum(np.fabs(Q_sa_estimate_updated - Q_sa_estimate))
    print('-------------------------------------------------------------error:-------',error)
    if error <= threshold:
        print('Has converge itera time is:', epsiod + 1, 'epso is ', epso, 'error is ', error)
        break

    if (epsiod+1)%10==0 and epsiod>1900 and epsiod<=2050:
        epso=epso*0.9
        print('epso becomes:',epso,'------------------------------------------------------------------------------')
    if (epsiod + 1) % 10 == 0 and epsiod > 2050 and epsiod <= 2100:
        epso = epso * 0.1
        print('epso becomes:', epso, '-----------------------------------------------------------------------------')

    # print(agent.x)
    error_map.append(np.sum(list_error)/agent.x)
#-------------------------------------------------------------------------------------------------------------------
#set epsilon as 0 then start test.
epso = 0
test(Q_sa)
#--------------------------------------------------------------------------------------------------------------------

plt.show()

