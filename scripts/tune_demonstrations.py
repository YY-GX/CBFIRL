import pickle
import numpy as np
import envs.config as config

with open('src/demonstrations/safe_demo_correct.pkl', 'rb') as f:
    demonstrations = pickle.load(f)

print(len(demonstrations))
# TODO: modify demos to make their velocity reduce to 0
for traj in demonstrations:
    if len(traj['observations']) < 50:
        traj['actions'][-1][-1, :] = np.array([- traj['observations'][-1][-1, 2] / config.TIME_STEP, - traj['observations'][-1][-1, 3] / config.TIME_STEP])
        traj['observations'][-1][-1, 2:] = 0


        supple_num = 50 - len(traj['observations'])
        supple_ob, supple_ac = traj['observations'][-1], np.zeros_like(traj['actions'][-1])
        supple_ob[:, 2:] = 0
        for _ in range(supple_num):
            traj['observations'].append(supple_ob)
            traj['actions'].append(supple_ac)

with open('src/demonstrations/safe_demo_5.pkl', 'wb') as f:
    pickle.dump(demonstrations, f)


# with open('src/demonstrations/safe_demo_5.pkl', 'rb') as f:
#     demonstrations = pickle.load(f)
#
# for traj in demonstrations:
#     for ob in traj['observations']:
#         print(ob)
