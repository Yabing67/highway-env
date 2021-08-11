# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test
   Description :
   Author :       yabing
   date：          2021/7/6
-------------------------------------------------

"""

import gym
import highway_env
# ==================================
#        Main script
# ==================================
if __name__ == "__main__":
    env = gym.make("exit-v1")
    obs=None
    for _ in range(2000):
        #action = env.action_type.actions_indexes["IDLE"]
        print(env.action_type.space() )
        action = 2
        obs, reward, done, info = env.step(action)
        #print(obs)
        env.render()
        #plt.imshow(env.render(mode="rgb_array"))
        #plt.show()
        if done:
            env.reset()