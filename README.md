# Reinforcement_learning
The demo code to study reinforcement learning. 

---
## 八大要素   
环境状态S  
个体动作A  
环境奖励R  
个体policy  
采取行动之后的价值value  
r奖励衰减因子  
状态转化模型  
探索率  

---
## 马尔可夫决策过程
$v_{\pi}(s) = \sum\limits_{a \in A} \pi(a|s)(R_s^a + \gamma \sum\limits_{s' \in S}P_{ss'}^av_{\pi}(s')) \\
q_{\pi}(s,a) = R_s^a + \gamma \sum\limits_{s' \in S}P_{ss'}^a\sum\limits_{a' \in A} \pi(a'|s')q_{\pi}(s',a')
$

---

### 最优价值函数推理  
$$
v_{*}(s) = \max_{\pi}v_{\pi}(s) \\ 
q_{*}(s,a) = \max_{\pi}q_{\pi}(s,a) \\
\pi_{*}(a|s)= \begin{cases} 1 & {if\;a=\arg\max_{a \in A}q_{*}(s,a)}\\ 0 & {else} \end{cases} \\
v_{*}(s) = \max_{a}q_{*}(s,a) \\
q_{*}(s,a) = R_s^a + \gamma \sum\limits_{s' \in S}P_{ss'}^av_{*}(s') \\  
v_{*}(s) = \max_a(R_s^a + \gamma \sum\limits_{s' \in S}P_{ss'}^av_{*}(s')) \\
q_{*}(s,a) = R_s^a + \gamma \sum\limits_{s' \in S}P_{ss'}^a\max_{a'}q_{*}(s',a')
$$

---
## Dynamic Programming  


---
## Monte Carlo  

$$
\mu_k = \frac{1}{k}\sum\limits_{j=1}^k x_j = \frac{1}{k}(x_k + \sum\limits_{j=1}^{k-1}x_j) =  \frac{1}{k}(x_k + (k-1)\mu_{k-1}) = \mu_{k-1} +  \frac{1}{k}(x_k -\mu_{k-1}) \\ 
N(S_t) = N(S_t)  +1 \\
V(S_t) = V(S_t)  + \frac{1}{N(S_t)}(G_t -  V(S_t) ) \\
$$

---
### TD   

$$
G(t) = R_{t+1} + \gamma V(S_{t+1}) \\
V(S_t) = V(S_t)  + \alpha(G_t -  V(S_t) ) \\
Q(S_t, A_t) = Q(S_t, A_t) + \alpha(G_t -  Q(S_t, A_t))
$$


---
### SARSA  
$$
Q(S,A) = Q(S,A) + \alpha(R+\gamma Q(S',A') - Q(S,A))
$$

---
### Q learning  

$$
Q(S,A) = Q(S,A) + \alpha(R+\gamma \max_aQ(S',a) - Q(S,A))
$$

---
### DQN 

$$
y_j= \begin{cases} R_j& {is\_end_j\; is \;true}\\ R_j + \gamma\max_{a'}Q(\phi(S'_j),A'_j,w) & {is\_end_j \;is\; false} \end{cases}
$$

---
### Nature DQN  
　　这里目标Q值的计算使用到了当前要训练的Q网络参数来计算𝑄(𝜙(𝑆′𝑗),𝐴′𝑗,𝑤)，而实际上，我们又希望通过𝑦𝑗来后续更新Q网络参数。这样两者循环依赖，迭代起来两者的相关性就太强了。不利于算法的收敛。因此，一个改进版的DQN: Nature DQN尝试用两个Q网络来减少目标Q值计算和要更新Q网络参数之间的依赖关系。

$$
y_j= \begin{cases} R_j& {is\_end_j\; is \;true}\\ R_j + \gamma\max_{a'}Q'(\phi(S'_j),A'_j,w') & {is\_end_j \;is\; false} \end{cases}
$$

---
### Double DQN  

$$
y_j= \begin{cases} R_j& {is\_end_j\; is \;true}\\ R_j + \gamma Q'(\phi(S'_j),\arg\max_{a'}Q(\phi(S'_j),a,w),w')& {is\_end_j\; is \;false} \end{cases}
$$

---
### Prioritized Replay DQN


---
### Dueling DQN  

---
### DDPG
 
