---
layout: post
title: Q-learning vs Sarsa
---

Q-learning (off-policy) and Sarsa (on-policy) are two basic methods for reinforcement learning. The difference between two is the way they update Q table.

In Q learning, the procedures to update Q table are:

![_config.yml]({{ site.baseurl }}/images/Q-learning-pseudocode.png)

In Saras, the procedures to update Q table are:

![_config.yml]({{ site.baseurl }}/images/Sarsa-pseudocode.png)

For Q-learning, it updates Q-table by always assuming Q of next state has the best action. For Saras, it updates Q-table by already choosing Q of next state using $\epsilon$-greedy method. That means there is certain chance $1-\epsilon$ to update $Q(s, a)$ using not maximum $Q(s', a')$. So if there is $Q(s', a')$ that is very negative (dangerous), then $Q(s, a)$ actually become smaller. Therefore, Sarsa method actually tends to choose a safer actions/paths if there are large enough punishment.
