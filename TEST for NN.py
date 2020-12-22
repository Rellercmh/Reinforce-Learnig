import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



period =15
GAMMA = 0.5
LEARNING_RATE = 0.001
episods=1
MEMORY_SIZE = 1000000
BATCH_SIZE = 7
EXPLORATION_MAX = 0.1


def Do_action(n):  # given action number ,return direction of action

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
def ifterminal(next_state):
    if next_state>250/1200 or next_state<10/1200:
        return True

    else:return False


map=np.array([13,14,15,16,15,14,13,14,16,17,18,19,18,17,15,16,14,15,16,15,15,16,17])
print(map)

list_height=[]

class DQNSolver:

    def __init__(self, observation_space=1, action_space=5):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update

            self.model.fit(state, q_values, verbose=0)






observation_space = 1
action_space = 5
dqn_solver = DQNSolver(observation_space, action_space)
dqn_solver.model.load_weights('my_model_p20_15.h5')
dqn_solver.exploration_rate=0
run = 0
list_step=[0]
list_z_=[]
list_x_=[]
list_error_=[]
list_height_=[]
for episod in range(episods):
    print('new episod------------------------------------',episod,'-----------------------------------------')
    agent = prob()
    height = map[agent.x // period]
    run += 1
    # state = env.reset()
    # state = np.reshape(state, [1, observation_space])
    state=round(force_feedback(agent.z-height),3)*1000/1200
    state = np.reshape(state, [1, observation_space])
    step = 0
    list_z = []
    list_x = []
    list_error= []
    list_height=[]

    while True:
        height=map[agent.x//period]
        list_height.append(height)
        step += 1

        # env.render()
        action = dqn_solver.act(state)

        output = agent.signal + Do_action(action) * 10
        agent.operate(output)

        list_z.append(agent.z)
        list_error.append(np.abs(agent.setheight+height-agent.z))

        state_next=round(force_feedback(agent.z-height),3)*1000/1200
        reward=1/(1+(np.abs(agent.setforce-force_feedback(agent.z-height)*5)))
        terminal= ifterminal(state_next)
        # state_next, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -1

        state_next = np.reshape(state_next, [1, observation_space])

        dqn_solver.remember(state, action, reward, state_next, terminal)
        print('run',run,'step',step,'exploration_rate', dqn_solver.exploration_rate,'max setp',max(list_step))
        state = state_next


        if terminal:
            list_step.append(step)
            break
        agent.x += 1
        list_x.append((agent.x))

        # dqn_solver.experience_replay()



        if agent.x ==len(map)*period:
            list_x_=np.copy(list_x)
            list_z_=np.copy(list_z)
            list_error_=np.copy(list_error)
            list_height_=np.copy(list_height)
            plt.title(('error per unit:', np.sum(list_error_) / (period * len(map))), fontsize=25)

            list_step.append(step)
            # dqn_solver.model.save('my_model.h5')
            # print('has save')



            plt.scatter(list_x_, list_height_, s=1)
            plt.scatter(list_x_, list_z_, s=1)
            # plt.scatter(list_x_, list_error_, s=1)
            #~`````````````````````````````````````````````````````````````````````````````````````````````````
            plt.style.use('seaborn-pastel')
            # 定制画布
            fig = plt.figure()
            ax = plt.axes(xlim=(0, len(map) * period), ylim=(0, 30))
            # 折线图对象line
            line, = ax.plot([], [], ':')
            line2, = ax.plot([], [], ':')


            def init():
                line.set_data([], [])
                line2.set_data([], [])
                return line, line2,
            x = []
            y = []
            s = []
            def animate(i):
                x.append(list_x_[i])
                y.append((list_z_[i]))
                s.append(list_height_[i])
                line.set_data(x, y)
                line2.set_data(x, s)
                return line, line2,

            anim = FuncAnimation(fig, animate, init_func=init,
                                 frames=int(len(list_x_) / 1), interval=0.1, blit=True, repeat=False)
            # anim.save('DQN_P19_15_15_2.gif', writer='imagemagick')


            plt.show()

            print('----------------------------------Got final--------------------------------------------')
            break
#https://github.com/gsurma/cartpole/blob/master/cartpole.py

