# DQN-Project

The first combined application of neural networks (NNs), Q-Learningand experience replay to reinforcement learning (RL) problems was in aPhD disseration in 1992 [1]. 
In 2015, DeepMind introduced the Deep Q-Network (DQN) [2] which surprised and excited many when it was shownthat an end-to-end reinforcement learning (RL) agent modeled by a neuralnetwork (NN) was capable of learning successful policies for all the Atari2600 games usingonlythe screen pixels and game score as inputs. 
Thisis remarkable because it is intuitively much more similar to how animals,ourselves included, solve such problems in daily life. 
Simultaneously, devel-opments in the fields of recurrent neural networks (RNNs) and attentionhave yielded impressive models like the Long-Short Term Memory (LSTM)[3] model and the Transformer [4], among others. 
In 2016, inspired by theexcellent performance of NNs in RL, OpenAI introduced the OpenAI Gym[5], a collection of benchmark RL problems unified under a common inter-face. 
This project investigates the performance of variants of DQNs on thecartpole-v1 task. 
The models are Deep Recurrent Q-Learning (DRQN) [6],as well as novel variants including the Deep Vision Transformer Q-Network(DVTQN) and the Deep Attention Q-Network (DAQN). 
The results showthat DQN outperforms all three variants on the cartpole task
