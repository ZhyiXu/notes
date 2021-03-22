## DDPG (Deep Deterministic Policy Gradient)

### 背景描述

概括来说，RL要解决的问题是：让agent学习在一个环境中的如何行为动作(act)， 从而获得最大的奖励值总和(total reward)。这个奖励值一般与agent定义的任务目标关联。

agent需要的主要学习内容：第一是行为策略(action policy)， 第二是规划(planning)。其中，行为策略的学习目标是最优策略， 也就是使用这样的策略，可以让agent在特定环境中的行为获得最大的奖励值，从而实现其任务目标。

行为(action)可以简单分为：
- 连续的：如赛车游戏中的方向盘角度、油门、刹车控制信号，机器人的关节伺服电机控制信号。
- 离散的：如围棋、贪吃蛇游戏。 Alpha Go就是一个典型的离散行为agent。

DDPG是针对连续行为的策略学习方法。

![这里写图片描述](https://img-blog.csdn.net/2018062216054550?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwNjE1OTAz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### DDPG的定义和应用场景

在RL领域，DDPG主要从：PG -> DPG -> DDPG 发展而来。

先复述一下相关的基本概念：

- $s_t$:  在t时刻，agent观察到的环境状态，比如观察到的环境图像，agent在环境中的位置、速度、机器人关节角度等；
- $a_t$ : 在t时刻，agent选择的行为（action），通过环境执行后，环境状态由 $s_t$ 转换为 $s_{t+1}$；
- $r(s_t,a_t)$ 函数: 环境在状态 $s_t$ 执行行为 $a_t$ 后，返回的单步奖励值；

上述关系可以用一个状态转换图来表示：

![state transition img](https://img-blog.csdn.net/20171107230547867?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2VubmV0aF95dQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

$R_t$：是从当前状态直到将来某个状态，期间所有行为所获得奖励值的加权总和，即discounted future reward:
$$
R_{t} = \sum ^{T} _{i=t} \gamma ^{i-t} r(s_{i} , a_{i})
$$
其中 $\gamma$ 叫做discounted rate, ∈[0,1],通常取0.99.

### PG

R.Sutton 在2000年提出的Policy Gradient 方法，是RL中，学习连续的行为控制策略的经典方法，其提出的解决方案是：通过一个概率分布函数 $π_θ(s_t|θ^π)$ ， 来表示每一步的最优策略， 在每一步根据该概率分布进行action采样，获得当前的最佳action取值；即：

$$
a_{t}\sim\pi_{\theta}(s_{t} | \theta^{\pi})
$$
生成action的过程，本质上是一个随机过程；最后学习到的策略，也是一个随机策略(stochastic policy).

### DPG

Deepmind的D.Silver等在2014年提出DPG： Deterministic Policy Gradient， 即确定性的行为策略，每一步的行为通过函数μ直接获得确定的值：
$$
a_{t} = \mu(s_{t} | \theta^{\mu})
$$
这个==函数μ即最优行为策略==，不再是一个需要采样的随机策略。为何需要确定性的策略？简单来说，PG方法有以下缺陷：

- 即使通过PG学习得到了随机策略之后，在每一步行为时，我们还需要对得到的最优策略概率分布进行采样，才能获得action的具体值；
- 而action通常是高维的向量，比如25维、50维，在高维的action空间的频繁采样，无疑是很耗费计算能力的；
- 在PG的学习过程中，每一步计算policy gradient都需要在整个action space进行积分:

$$
\triangledown_{\theta} =  \int_{\mathcal{S}} \int_{A} \rho(s) \pi_{\theta}(a|s)Q^{\pi} (s,a)dads
$$
(Q,ρ 参见下面DDPG部分的概念定义.)

这个积分我们一般通过Monte Carlo 采样来进行估算，需要在高维的action空间进行采样，耗费计算能力。
如果采取简单的Greedy策略，即每一步求解 $ argmax_a Q(s,a)$ 也不可行，因为在连续的、高维度的action空间，如果每一步都求全局最优解，太耗费计算性能。

在这之前，业界普遍认为，环境模型无关 (model-free) 的确定性策略是不存在的，在2014年的DPG论文中，D.Silver等通过严密的数学推导，证明了DPG的存在， 其数学表示参见DDPG算法部分给出的公式 (3)。

然后将DPG算法融合进actor-critic框架，结合Q-learning或者Gradient Q-learning这些传统的Q函数学习方法，经过训练得到一个确定性的最优行为策略函数。

### DDPG

Deepmind在2016年提出DDPG，全称是：Deep Deterministic Policy Gradient, 是将深度学习神经网络融合进DPG的策略学习方法。

相对于DPG的核心改进是： ==采用卷积神经网络作为策略函数μ和Q函数的模拟，即策略网络和Q网络；然后使用深度学习的方法来训练上述神经网络==。

Q函数的实现和训练方法，采用了DeepMind 2015年发表的DQN方法 ,即 AlphaGo使用的Q函数方法。

### DDPG算法相关基本概念定义

我们以Open Gym 作为环境为例来讲解。先复述一下DDPG相关的概念定义：

- 确定性行为策略μ:  定义为一个函数，每一步的行为可以通过 $a_t=μ(s_t)$ 计算获得。
- 策略网络：用一个卷积神经网络对μ函数进行模拟，这个网络我们就叫做策略网络，其参数为 $θ^μ$；
- behavior policy β: 在RL训练过程中，我们要兼顾2个e: exploration和exploit；exploration的目的是探索潜在的更优策略，所以训练过程中，我们为action的决策机制引入随机噪声：

将action的决策从确定性过程变为一个随机过程， 再从这个随机过程中采样得到action，下达给环境执行。过程如下图所示：

![behaviour policy](https://img-blog.csdn.net/20171107230849577?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2VubmV0aF95dQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

上述这个策略叫做behavior策略，用β来表示, 这时RL的训练方式叫做off-policy. 

这里与ϵ−greedy的思路是类似的。

DDPG中，使用Uhlenbeck-Ornstein随机过程（下面简称UO过程），作为引入的随机噪声：UO过程在时序上具备很好的相关性，可以使agent很好的探索具备动量属性的环境。

注意：
– 这个β不是我们想要得到的最优策略，仅仅在训练过程中，生成下达给环境的action， 从而获得我们想要的数据集，比如状态转换(transitions)、或者agent的行走路径等，然后利用这个数据集去训练策略 μ，以获得最优策略。

– 在test 和 evaluation 时，使用μ，不会再使用β。

Q函数: 即action-value 函数，定义在状态$s_t$下，采取动作$a_t$后，且如果持续执行策略 $μ$ 的情况下， 所获得的$R_t$ 期望值, 用Bellman 等式来定义：
$$
Q^{\mu}(s_{t}, a_{t}) = E\ [r(s_{t}, a_{t}) + \gamma Q^{\mu}(s_{t+1}, \mu(s_{t+1})) ]
$$
可以看到，Q函数的定义是一个递归表达，在实际情况中，我们不可能每一步都递归计算Q的值，可行的方案是通过一个函数对Bellman等式表达进行模拟。

Q网络：DDPG中，我们用一个卷积神经网络对Q函数进行模拟，这个网络我们就叫做Q网络， 其参数为$θ^Q$。采用了DQN相同的方法。

如何衡量一个策略μ的表现：用一个函数 $J$ 来衡量，我们叫做 performance objective，针对off-policy学习的场景，定义如下：
$$
J_{\beta}(\mu) =  \int  _{S} \rho^{\beta}(s) Q^{\mu}(s, \mu(s))ds
=  E _{s\sim\rho^{\beta}}[ Q^{\mu}(s, \mu(s))]
$$
其中：

- s是环境的状态，这些状态(或者说agent在环境中走过的状态路径)是基于agent的behavior策略产生的，它们的分布函数(pdf) 为$ρ^β$；
- $Q^μ(s,μ(s))$  是在每个状态下，如果都按照μ策略选择acton时，能够产生的Q值。也即，$J_β(μ)$ 是在s根据$ρ^β$分布时，$Q^μ(s,μ(s))$ 的期望值。

训练的目标： 最大化$J_β(μ)$，同时最小化Q网络的Loss(下面描述算法步骤时会给出)。

最优行为策略μ的定义: 即最大化$J_β(μ)$的策略：
$$
\mu = \mathop{argmax}_{\mu}J(\mu)
$$
训练μ网络的过程，就是寻找μ网络参数θμ的最优解的过程，我们使用SGA (stochastic gradient ascent)的方法。

最优Q网络定义：具备最小化的Q网络Loss；

训练Q网络的过程，就是寻找Q网络参数$θ^Q$的最优解的过程，我们使用SGD的方法。

### DDPG实现框架和算法

**online 和 target 网络**

以往的实践证明，如果只使用单个”Q神经网络”的算法，学习过程很不稳定，因为Q网络的参数在频繁gradient update的同时，又用于计算==Q网络==和==策略网络==的gradient, 参见下面等式(1),(2),(3).

基于此，DDPG分别为策略网络、Q网络各创建两个神经网络拷贝,一个叫做online，一个叫做target:
$$
策略网络 \quad    
    \begin{cases}
    online:  \mu(s|\theta^{\mu})  :   gradient更新\theta^{\mu} \\
     target:  \mu^{\prime}(s|\theta^{\mu^{\prime}})  : soft \ update \ \theta^{\mu^{\prime}}
    \end{cases}
$$

$$
Q网络 \quad
    \begin{cases}
     online:  Q(s,a|\theta^{Q}) :  gradient更新\theta^{Q}  \\
     target :  Q^{\prime}(s,a|\theta^{Q^{\prime}}) :soft \ update \ \theta^{Q^{\prime}}
    \end{cases}
$$

在训练完一个mini-batch的数据之后，通过SGA/SGD算法更新online网络的参数，然后再通过soft update算法更新 target 网络的参数。soft update是一种running average的算法：
$$
\mathop{soft \ update: } _{\tau 一般取值0.001}
    \begin{cases}
    \theta^{Q^{\prime}} \leftarrow \tau\theta^{Q} + (1-\tau)\theta^{Q^{\prime}} \\
    \theta^{\mu^{\prime}} \leftarrow \tau\theta^{\mu} + (1-\tau)\theta^{\mu^{\prime}} 
    \end{cases}
$$
优点：target网络参数变化小，用于在训练过程中计算online网络的gradient，比较稳定，训练易于收敛。

代价：参数变化小，学习过程变慢。

### **DDPG实现框架，如下图所示：**

![ddpg total arch](https://img-blog.csdn.net/20171108090350229?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2VubmV0aF95dQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**DDPG算法流程如下：**

将 online 网络的参数拷贝给对应的target网络参数：$\theta^{Q^{\prime}} \leftarrow \theta^{Q},   
   \theta^{\mu^{\prime}}  \leftarrow \theta^{\mu}$ ;
初始化 replay memory buffer R ;
for each episode:
   初始化 $UO$ 随机过程；
   for t = 1, T:
   下面的步骤与DDPG实现框架图中步骤编号对应：

​	1、actor 根据 behavior策略选择一个 $a_t$ , 下达给gym执行该 $a_t$.
$$
a_{t} = \mu(s_{t} | \theta^{\mu}) + \mathcal{N}_{t}
$$
​	behavior策略是一个根据当前online策略 μ 和随机UO噪声生成的随机过程, 从这个随机过程采样 获得 $a_t$ 的值。
​	2、gym执行 $a_t$ ，返回 reward $r_t$ 和新的状态 $s_{t+1}$;

​	3、actor将这个状态转换过程(transition): $(s_t,a_t,r_t，s_{t+1})$ 存入replay memory buffer R中，作为训练online网络的数据集。

​	4、从replay memory buffer R中，随机采样N个 transition 数据，作为online策略网络、 online Q网络的一个mini-batch训练数据。我们用 $(s_i,a_i,r_i，s_{i+1})$ 表示mini-batch中的单个transition数据。

   5、计算online Q网络的 gradient：
		Q 网络的loss定义：使用类似于监督式学习的方法，定义==loss为MSE: mean squared error==：
$$
L = \frac1N \sum_{i}  ( y_{i} -Q(s_{i},a_{i} | \theta^{Q})  )^2  \quad  \quad  \quad (1)
$$
​		其中， yi 可以看做”标签”：
$$
y_{i} = r_{i} + \gamma Q^{\prime}(s_{i+1}, 
           \mu^{\prime}(s_{i+1} | \theta^{\mu^{\prime}}) | \theta^{Q^{\prime}})  \quad  \quad  \quad (2)
$$
​		基于标准的back-propagation方法，就可以求得L针对 $θ^Q $ 的gradient：$\triangledown_{\theta^{Q}} L$。
​		有两点值得注意：1. $y_i$ 的计算，使用的是 target 策略网络μ′和 target Q 网络Q′, 这样做是为了Q网络参数的学习过程更加稳定，易于收敛。2. 这个标签本身依赖于我们正在学习的target网络，这是区别于监督式学习的地方。

​	6、update online Q： ==采用Adam optimizer==更新 $θ^Q $ ;

​	7、 计算策略网络的policy gradient：

​	policy gradient的定义：表示performance objective的函数JJ针对 $θ^μ$ 的 gradient。 根据2015 D.Silver 的DPG 论文中的数学推导，在采用off-policy的训练方法时，policy gradient算法如下：
$$
\triangledown_{\theta^{\mu}}J_{\beta}(\mu) \approx E_{s\sim\rho^{\beta}} 
        [\triangledown_{a}Q(s, a | \theta^{Q} ) | _{a =\mu(s)}\cdot \triangledown_{\theta^{\mu}} \mu(s|\theta^{\mu})]   \quad  \quad  \quad (3)
$$
也即，policy gradient是在 s 根据 $ρ^β$ 分布时，$\triangledown_{a}Q \cdot \triangledown_{\theta^{\mu}} \mu$ 的期望值。 我们用Monte-carlo方法来估算这个期望值： 在replay memory buffer中存储的(transition) : $(s_i,a_i,r_i，s_{i+1})$, 是基于 agent 的behavior 策略 $β$ 产生的，它们的分布函数(pdf)为$ρ^β$，所以当我们从replay memory buffer中随机采样获得mini-batch数据时，根据Monte-carlo方法，使用mini-batch数据代入上述policy gradient公式，可以作为对上述期望值的一个无偏差估计 (un-biased estimate), 所以policy gradient 可以改写为：
$$
\triangledown_{\theta^{\mu}}J_{\beta}(\mu) \approx \frac1N\sum_{i} 
        (\triangledown_{a}Q(s, a | \theta^{Q} ) | _{s=s_{i}, a=\mu(s_{i})} 
        \cdot \triangledown_{\theta^{\mu}} \mu(s|\theta^{\mu}  ) | _{s=s_{i}}) \quad\quad (4)
$$
​	8、 update online策略网络：==采用Adam optimizer==更新 $θ^μ$;

​	9 、soft-update target 网络 $μ′$ 和 $Q′$:
​		  使用running average 的方法，将online网络的参数，soft-update 给 target 网络的参数：
$$
\mathop{soft \ update: } _{\tau 一般取值0.001}
            \begin{cases}
            \theta^{Q^{\prime}} \leftarrow \tau\theta^{Q} + (1-\tau)\theta^{Q^{\prime}} \\
            \theta^{\mu^{\prime}} \leftarrow \tau\theta^{\mu} + (1-\tau)\theta^{\mu^{\prime}} 
            \end{cases}
$$
​	end for time step
end for episode 



### **总结一下：**

actor-critic框架是一个在循环的episode和时间步骤条件下，通过环境、actor和critic三者交互，来迭代训练策略网络、Q网络的过程。

![image-20200407200632961](D:\Notes\raw_images\image-20200407200632961.png)



### DDPG对于DPG的关键改进

- 使用==卷积神经网络来模拟策略函数和Q函数==，并用深度学习的方法来训练，证明了在RL方法中，非线性模拟函数的准确性和高性能、可收敛；而DPG中，可以看成使用线性回归的机器学习方法：使用带参数的线性函数来模拟策略函数和Q函数，然后使用线性回归的方法进行训练。
- experience replay memory的使用：actor同环境交互时，产生的transition数据序列是在时间上高度关联(correlated) 的，如果这些数据序列直接用于训练，会导致神经网络的overfit，不易收敛。==DDPG的actor将transition数据先存入experience replay buffer, 然后在训练时，从experience replay buffer中随机采样mini-batch数据，这样采样得到的数据可以认为是无关联的==。
- target 网络和online 网络的使用， 使的学习过程更加稳定，收敛更有保障。



原文链接：https://blog.csdn.net/kenneth_yu/article/details/78478356