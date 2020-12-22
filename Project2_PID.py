import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

map=np.array([13,14,15,16,17,18,19,20,21,22,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,3,4,5,6,5,4,3,5,6,7,8,9,8,7,6,5,4,3])
print(map)

def force_feedback(distance):
    return 1/(distance*distance)

class prob():
    def __init__(self):
        self.x=0
        self.y=0
        self.center=20
        self.z=15.6
        self.setforce=1/25
        self.signal=self.z/2
        self.Hysteresis=0.1
        self.setheigh=5
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
agent=prob()

Kp = 8          #set three parameter
Ki = 0.5
Kd =2
period=1000    #the speed of  PID control can not be too fast,if use 200, algorithm can figure out the outline
dt=0.001       #dt indicate the control time accuracy
integral = 0
previous_error = 0
list_error = []                        #error list
list_agent_x=[]                        #x coordinate list
list_agent_x.append(agent.x)
list_z=[]                              #agent height lis5t
list_z.append(agent.z)
list_z_target=[]
list_z_target.append(13)
ep=0
i=0

while True:
    height = map[agent.x//period]        #record step height
    list_z_target.append(height)
    print(agent.z)

    error = agent.setforce-force_feedback(agent.z-height)  #caluculate error according to forcefeed back


    integral = integral + error * dt                      #traditional PID control method
    derivative = (error - previous_error) / dt
    output = agent.signal - (Kp * error + Ki * integral + Kd * derivative) /100  #after test  8 0.5 2 maybe too big, so divided by 100
    agent.operate(output)      #purt new
    list_z.append(agent.z)
    previous_error = error
    list_error.append(agent.z-height)

    # time.sleep(dt)
    
    ep+=1
    agent.x+=1
    list_agent_x.append(agent.x)

    if ep% period ==0:
        if i<len(map)-1:
            i+=1
            ep=0
        else:break
print(len(list_agent_x))
print(len(list_z))
print(len(list_z_target))

plt.scatter(list_agent_x,list_z,s=1)
plt.scatter(list_agent_x,list_z_target,s=1)


print(agent.x)
print('errpr per unit: ',np.sum(np.fabs(list_error))/(period*len(map)))
# plt.title(('error per unit:',round(np.sum(np.fabs(list_error))/(period*len(map)),3)),fontsize=25)
#------------------------------------------------------------------------------------------------------------

plt.style.use('seaborn-pastel')
#定制画布
fig = plt.figure()
ax = plt.axes(xlim=(0, len(map)*period), ylim=(0, 30))
#折线图对象line
line, = ax.plot([], [], ':')
line2, = ax.plot([], [], ':')

def init():
    line.set_data([], [])
    line2.set_data([], [])
    return line,line2,
# 生成每一帧所用的函数
x=[]
y=[]
s=[]
def animate(i):
    # x = np.linspace(0, len(map), period)
    # y=  list_agent_x[i]
    # y = np.sin(2 * np.pi * (x - 0.01 * i))
    x.append(list_agent_x[i*10])
    y.append((list_z[i*10]))
    s.append(list_z_target[i*10])
    line.set_data(x, y)
    line2.set_data(x,s)
    return line,  line2,

anim = FuncAnimation(fig, animate,init_func=init,
                     frames=int(period*len(map)/10), interval=10, blit=True,repeat=False)
# anim.save('sine_wave4.gif')

plt.show()