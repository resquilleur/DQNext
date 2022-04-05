# README

My learning project based on the book https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On

The author no longer supports ptan, so to install it on the latest versions of other libraries
I do `git clone` ptan repo, so the path must be used `ptan.ptan`.

In ptan\ptan\ignite.py change `import ptan` to `import ptan.ptan as ptan`


In with DQN has extensions:
- n - steps (4 in this case)
- Dueling DQN
- Noisy networks

To start training, enter the command:
`python dqn_ext_speed.py --cuda --envs 3`

To start monitoring via tensorboardX:
`tensorboard --logdir runs`
runs - folder in root of project

To play back the results of the model:
`python dqn_play.py -m `

Only extensions:
`Game solved in 1:45:21, after 292 episodes and 577441 iterations!`

Extensions with 3 environments:
`Game solved in 1:34:10, after 352 episodes and 250189 iterations!`