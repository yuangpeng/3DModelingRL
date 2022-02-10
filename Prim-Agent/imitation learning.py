import torch
import numpy as np
from memory import Memory
import torch.nn as nn
import torch.nn.functional as F
import copy
DAGGER_EPOCH = 1
DAGGER_ITER = 4
DAGGER_LEARN = 4000
GAMMA = 0.9                  # reward discount
BOX_NUM = 27
BATCH_SIZE = 64
TARGET_REPLACE_ITER = 4000 
ACTION_NUM = 756
MEMORY_LONG_CAPACITY = 2000
MEMORY_SELF_CAPACITY = 1000  # shared by D_short and D_self
LR = 0.00008    
class IL():
    def __init__(self, weights_path=None):
        self.eval_net, self.target_net = Net(), Net()

        if weights_path is not None:
            self.eval_net.load_state_dict(torch.load(weights_path))
            self.target_net.load_state_dict(torch.load(weights_path))

        # the memory_self is shared by D_short in IL and D_self in RL
        self.memory_long = Memory(MEMORY_LONG_CAPACITY)
        self.memory_self = Memory(MEMORY_SELF_CAPACITY)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        action_id = 0

        # map action id to box_id (27), [c1:x,y,z; c2:x,y,z; delete] (7), range[-2,-1,1,2] (4)
        # 3 types of actions, 3 moving directions (x,y and z) for the drag actions (6) + delete (1) = (7)
        action_map = np.zeros((27 * 7 * 4, 3), dtype=int)
        map_action = np.zeros((27, 7, 4), dtype=int)
        for i in range(27):
            for j in range(7):
                for k in range(1, 5):

                    map_action[i][j][k-1] = action_id

                    if k >= 3:
                        k = k - 5
                    para = np.array([i, j, k])
                    action_map[action_id] = para

                    action_id += 1

    def get_virtual_expert_action(self, env,valid_mask, agent,random=False):

        box_id = env.step_count % env.box_num

        max_action = -1
        max_reward = -1000

        box_level_range = 6

        # allow delete actions
        if env.step_count > env.max_step*0.5:
            box_level_range = 7

        for i in range(box_level_range):
            for j in range(4):
                action = self.map_action[box_id, i, j]

                if valid_mask[action] == 0:
                    continue

                s_, boxes_, step_, reward, done = agent.next_no_update(action)

                if reward > max_reward:
                    max_reward = reward
                    max_action = action

        return max_action
    def get_valid_action_mask(self, env,boxes_normalized):

        box_id = env.step_count % env.box_num
        boxes = boxes_normalized*env.vox_size_l
        valid_mask = np.ones((env.action_num), dtype=np.int)

        for a in range(env.action_num):

            i, j, k = self.action_map[a]
            box_id = env.step_count % env.box_num
            # only edit an designated box
            if i != box_id:
                valid_mask[a] = 0
            # delete action
            if j == 6:
                if boxes[i][3]-boxes[i][0] == 0 or boxes[i][4]-boxes[i][1] == 0 or boxes[i][5]-boxes[i][2] == 0:
                    valid_mask[a] = 0
            # edit action
            else:
                bc = boxes[i][j] + k
                if j <= 2:
                    if bc > boxes[i][j+3] or bc < 0:
                        valid_mask[a] = 0
                elif j >= 3:
                    if bc < boxes[i][j-3] or bc > self.vox_size_l:
                        valid_mask[a] = 0

        return valid_mask
    def tweak_box(self, action,env):

        box_id = env.step_count % env.box_num
        try_box = copy.copy(env.all_boxes)
        i, j, k = self.action_map[action]

        # delete action
        if j == 6:
            midx = try_box[i][0]
            midy = try_box[i][1]
            midz = try_box[i][2]
            try_box[i][0:6] = [midx, midy, midz, midx, midy, midz]
        # edit action
        else:
            try_box[i][j] = try_box[i][j] + k*env.m_unit

        return try_box

    def learn(self, learning_mode, is_ddqn=True):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1

        # only IL by supervised-error only
        if learning_mode == 0:
            samples_from_expert = int(BATCH_SIZE*0.5)

        # only RL by td-error only
        elif learning_mode == 1:
            samples_from_expert = 0

        # jointly by td-error and supervised error
        elif learning_mode == 2:
            samples_from_expert = int(BATCH_SIZE*0.5)

        # jointly by td-error
        elif learning_mode == 3:
            samples_from_expert = int(BATCH_SIZE*0.5)

        bs1, ba1, br1, bs_1, done = self.memory_long.sample(samples_from_expert)
        bs2, bstep2, ba2, br2, bs_2, done= self.memory_self.sample(BATCH_SIZE-samples_from_expert)

        bs = torch.FloatTensor(np.concatenate((bs1, bs2), axis=0))
        ba = torch.FloatTensor(np.concatenate((ba1, ba2), axis=0)).long()
        br = torch.FloatTensor(np.concatenate((br1, br2), axis=0))
        bs_ = torch.FloatTensor(np.concatenate((bs_1, bs_2), axis=0))

        # expert's action's q_value
        # q_all_actions = Q(s), current Q network
        q_all_actions = self.eval_net(bs)
        q_a_expert = q_all_actions.gather(1, ba)

        # margin function
        margin = torch.FloatTensor(np.ones((BATCH_SIZE, ACTION_NUM)))*0.8
        t = np.linspace(0, BATCH_SIZE-1, BATCH_SIZE).astype(np.int)
        margin[t, ba[t, 0]] = 0

        # predicted maximal q_value with margin function
        q_a_predicted = q_all_actions + margin
        q_a_predicted = q_a_predicted.max(1)[0].view(BATCH_SIZE, 1)

        # compute the supervised loss
        loss1 = torch.mean(q_a_predicted - q_a_expert)

        # compute td error by target net
        if not is_ddqn:
            # the loss function of DQN
            q_a_next = self.target_net(bs_).detach()
            q_a_target = br + GAMMA * q_a_next.max(1)[0].view(BATCH_SIZE, 1)
            loss2 = self.loss_func(q_a_expert, q_a_target)

        else:
            # the loss function of DDQN
            q_a_next = self.target_net(bs_).detach()
            # action choose from current Q network
            best_action = self.eval_net(bs_).max(1)[1].view(BATCH_SIZE, 1)
            q_a_target = br + GAMMA * q_a_next.gather(1, best_action)
            loss2 = self.loss_func(q_a_expert, q_a_target)

        # only IL by supervised-error only
        # only RL by td-error only
        # jointly by td-error and supervised error
        # jointly by td-error
        if learning_mode == 0:
            loss = loss1
        elif learning_mode == 1:
            loss = loss2
        elif learning_mode == 2:
            loss = loss1 + loss2
        elif learning_mode == 3:
            loss = loss2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def next_no_update(self, action, env):

        try_boxes = env.tweak_box(action,env)
        IOU, local_IOU, delete_count = env.compute_increment(try_boxes)
        reward = env.compute_reward(IOU, local_IOU, delete_count)

        step_vec = np.zeros((env.max_step), dtype=np.int)
        if env.step_count+1 >= env.max_step:
            done = True
        else:
            done = False
            step_vec[env.step_count+1] = 1

        return env.ref, try_boxes/self.vox_size_l, step_vec, reward, done

    def imitation_learning(agent, env, writer, shape_list):
        box_id = env.step_count % env.box_num
        episode_count = 0
        # DAGGER_EPOCH = 1
        for epoch in range(DAGGER_EPOCH):
            # 对于在 shape_list 中的每一个shape
            for shape_count in range(len(shape_list)):

                shape_name = shape_list[shape_count]
                vox_l_fn = shape_vox_path + shape_name+'-16.binvox'
                vox_h_fn = shape_vox_path + shape_name+'-64.binvox'
                ref_fn = shape_ref_path + shape_name + '.png'
                shape_infopack = [shape_name, vox_l_fn, vox_h_fn, shape_ref_type, ref_fn]

                # DAGGER_ITER = 4
                for episode in range(DAGGER_ITER):

                    print('Shape', shape_name, 'Dagger episode', episode)

                    s, box, step = env.reset(shape_infopack)

                    episode_count += 1
                    agent.memory_self.clear()
                    acm_r = 0

                    while True:

                        valid_mask = agent.get_valid_action_mask(box)

                        # poll the expert
                        a = agent.get_virtual_expert_action(valid_mask)
                        s_, box_, step_, r, done = agent.next_no_update(a)
                        expert_action = self.action_map[a]

                        agent.memory_long.store(s, a, r, s_, done)
                        agent.memory_self.store(s, a, r, s_, done)

                        # update the state
                        if episode != 0:
                            a = agent.choose_action(s, box, step, valid_mask, 1.0)
                        real_action = self.action_map[a]
                        s_, r, done, info = env.step(a)

                        acm_r += r

                        if done:
                            # uncomment the following lines to output the intermediate results
                            # log_info='IL_'+str(epoch)+'_shape_'+str(shape_count)+'_epi_'+str(episode)+'_r_'+str(format(acm_r, '.4f'))+'_'+shape_name
                            # env.output_result(log_info, save_tmp_result_path)
                            writer.add_scalar('Prim_IL/'+shape_category, acm_r, episode_count)
                            break

                        s = s_

                    print('reward:', acm_r)

                    for learn in range(DAGGER_LEARN):
                        agent.learn(learning_mode=2, is_ddqn=True)
