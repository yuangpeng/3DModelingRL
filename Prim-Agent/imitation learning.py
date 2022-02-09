 
class IL():
    def get_virtual_expert_action(self, valid_mask, random=False):

        if not random:
            box_id = self.step_count % self.box_num
        else:
            box_id = np.random.randint(0, self.box_num)

        max_action = -1
        max_reward = -1000

        box_level_range = 6

        # allow delete actions
        if self.step_count > self.max_step*0.5:
            box_level_range = 7

        for i in range(box_level_range):
            for j in range(4):
                action = self.map_action[box_id, i, j]

                if valid_mask[action] == 0:
                    continue

                s_, boxes_, step_, reward, done = self.next_no_update(action)

                if reward > max_reward:
                    max_reward = reward
                    max_action = action

        return max_action
    def get_valid_action_mask(self, boxes_normalized):

        boxes = boxes_normalized*self.vox_size_l
        valid_mask = np.ones((self.action_num), dtype=np.int)

        for a in range(self.action_num):

            i, j, k = self.action_map[a]
            box_id = self.step_count % self.box_num
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

    def imitation_learning(agent, env, writer, shape_list):
        episode_count = 0
        # DAGGER_EPOCH = 1
        for epoch in range(p.DAGGER_EPOCH):
            # 对于在 shape_list 中的每一个shape
            for shape_count in range(len(shape_list)):

                shape_name = shape_list[shape_count]
                vox_l_fn = shape_vox_path + shape_name+'-16.binvox'
                vox_h_fn = shape_vox_path + shape_name+'-64.binvox'
                ref_fn = shape_ref_path + shape_name + '.png'
                shape_infopack = [shape_name, vox_l_fn, vox_h_fn, shape_ref_type, ref_fn]

                # DAGGER_ITER = 4
                for episode in range(p.DAGGER_ITER):

                    print('Shape', shape_name, 'Dagger episode', episode)

                    s, box, step = env.reset(shape_infopack)

                    episode_count += 1
                    agent.memory_self.clear()
                    acm_r = 0

                    while True:

                        valid_mask = env.get_valid_action_mask(box)

                        # poll the expert
                        a = env.get_virtual_expert_action(valid_mask)
                        s_, box_, step_, r, done = env.next_no_update(a)
                        expert_action = env.action_map[a]

                        agent.memory_long.store(s, a, r, s_, done)
                        agent.memory_self.store(s, a, r, s_, done)

                        # update the state
                        if episode != 0:
                            a = agent.choose_action(s, box, step, valid_mask, 1.0)
                        real_action = env.action_map[a]
                        s_, box_, step_, r, done = env.next(a)

                        acm_r += r

                        if done:
                            # uncomment the following lines to output the intermediate results
                            # log_info='IL_'+str(epoch)+'_shape_'+str(shape_count)+'_epi_'+str(episode)+'_r_'+str(format(acm_r, '.4f'))+'_'+shape_name
                            # env.output_result(log_info, save_tmp_result_path)
                            writer.add_scalar('Prim_IL/'+shape_category, acm_r, episode_count)
                            break

                        s = s_
                        box = box_
                        step = step_

                    print('reward:', acm_r)

                    for learn in range(p.DAGGER_LEARN):
                        agent.learn(learning_mode=2, is_ddqn=True)
