# README

My learning project based on the book https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On

The author no longer supports ptan, so to install it on the latest versions of other libraries
I do `git clone` ptan repo, so the path must be used `ptan.ptan`.

In ptan\ptan\ignite.py change `import ptan` to `import ptan.ptan as ptan`

To start training, enter the command:
`python main.py --cuda`

To start monitoring via tensorboardX:
`tensorboard --logdir runs`
runs - folder in root of project

To play back the results of the model:
`python dqn_play.py -m `