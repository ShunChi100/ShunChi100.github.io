---
layout: post
title: Q-learning in Reinforcement Learning
---

Most people have played the video game *Super Mario Bros*. In the game, in order to rescue the kidnapped Princess Peach, Mario has to get over many dangers and defeat many enemies in each themed level. 

![Link to Mario](https://media.giphy.com/media/QvSO4r5yDvGKY/giphy.gif)

[Gif source](https://media.giphy.com/media/QvSO4r5yDvGKY/giphy.gif)

At the first time of playing this game, I tried many possible actions, such as running and jumping in all directions. In the meanwhile, I observed the consequences of each action: sometimes nothing happened, sometimes Mario was killed by a Goomba, sometimes Mario got a mushroom, *etc.*. I learned from these action-and-consequence pairs and gradually became better and better in playing the game. One day, I could make little mistakes and have Mario rescued the Princess Peach, and I was super happy about it. 

This is a typical process for a human being to learn and find the way to award:

1. Fresh our memory.
2. Observe the current state.
3. Decide for the next action based on our knowledge.
4. Take the action
5. Enter in a new state, get a consequence, safe, killed, reward, etc.
6. Save the action-consequence in our memory and update our knowledge for making decisions next time.
6. Repeat 2, 3, 4, 5, 6 until winning (rescue Princess Peach).

Essentially, it follows the **law of effect** developed by Edward Thorndike.
> "Response that produces a satisfying effect in a particular situation become more likely to occur again in that situation, and response that produces a discomforting effect become less likely to occur again in that situation."

Can a computer also learn how to play Super Mario just like a human? 
We know that computer is not intelligent. It can only execute certain tasks that we programmed them to do. However, we, human, are smart enough that we can design some algorithms that allow a computer to learn to play the game. These algorithms are called **Reinforcement Learning** and they are an analogy to the learning process of a human. Technically, reinforcement learning is a sub-branch of machine learning. Through exploring a particular environment (the *Mushroom Kingdom* in *Super Mario Bros*), it trains the computer to automatically search the optimal (usually semi-optimal) policy (series of actions), to maximize its rewards. 

**Q-learning** is a type of reinforcement learning that optimizes the behavior of a machine through action and consequences. In analogy to human's learning. We need the following key ingredients: storing and updating knowledge, the mechanism for determining the next action, values for evaluating the consequences. 

The variables for Q-learning are

| variable | meaning |
| --- | -------- |
| `s` | state |
| `a` | action |
| `r` | reward (consequence) |
| `Q(s,a)` | Q-table (knowledge) |

The pseudo-code for Q-learning is the following [^p1]: 

```
Initialize Q(s,a)
Repeat (for each episode):
    Initialized s
    Repeat (for each step):
        Choose action a either from knowledge Q(s,a) or experiment a new random action
        Take action a, observe reward r and new state s'
        Update Q(s,a) from our observations on r and s'
        change current state s to s'
    until s is terminal
```

In the beginning, a computer does not have any knowledge of the game, so the Q-table is set to *Zero* state. When the computer starts to explore the Mario world, it decides the next move based on this zero-valued Q-table, resulting in random actions. However, as the computer plays more and more, it automatically updates Q-table based on the rewards and punishments. In particular, as seen in the pseudo-code, `Q(s,a)` is updated from the reward in the next state `s'` after taking action a. That means the reward in the next state back propagates to the current state. After many iterations, the future rewards can propagate back to many steps back. In this way, Q-table not only updates information based on instant rewards/punishments but also increasing its vision to the long-run benefit, the biggest reward in the end. In addition, at each step, Q-learning also allows certain probability to take random new action. This ensures that the algorithm can explore new actions and states, allowing finding the optimal path instead of staying in the old but maybe not the optimal path. Through the delayed reward, namely seeing the reward in the future from the current state, and certain probability to explore new states, Q-learning algorithm enables a computer to find an optimal (sometimes semi-optimal if the state space is huge) way to end. 

[^p1]: Reinforcement Learning: An Introduction, Sutton and Barto, (2012)

