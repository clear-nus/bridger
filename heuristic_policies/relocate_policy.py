import numpy as np


class BatchRelocatePolicy:
    def __init__(
            self,
            pred_length: int = 64,
            interpolation_num: int = 5,
            is_querying_sequence: bool = True
    ):
        """
        Handcrafted policy for relacate task for batched tasks.
        :param pred_length: the action seq length to be generated each time
        :param interpolation_num: number of steps/actions between any two keypoints in completing the task. Larger value means slower.
        """
        self.pred_length = pred_length
        self._is_querying_sequence = is_querying_sequence
        self.interpolation_num = interpolation_num
        # self.policy = RelocatePolicy(pred_length=pred_length, interpolation_num=interpolation_num, is_querying_sequence=is_querying_sequence)
        self.policies = None

        # define vectorised operation for get action batched, only once
        calc_action = lambda x, y: x.get_action(y)  # x policy obj, y state
        self.calc_action = np.vectorize(calc_action, signature="(),(m)->(p,q)")

    def initialise_policies(self, batch_size: int):
        policies = [
            RelocatePolicy(pred_length=self.pred_length, interpolation_num=self.interpolation_num) for _ in range(batch_size)
        ]
        self.policies = np.array(policies)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        if self.policies is None:
            self.initialise_policies(
                batch_size=state.shape[0]
            )
        return self.calc_action(self.policies, state)
        # batch_size = state.shape[0]
        # res_list = []
        # for idx in range(batch_size):
        #     res_list.append(
        #         self.policy.get_action(state=state[idx])
        #     )
        #
        # return np.array(res_list)

    def reset(self):
        if self.policies is None:
            return
        for policy in self.policies:
            policy.reset()




class RelocatePolicy:
    def __init__(self, pred_length: int = 16, interpolation_num: int = 200, is_querying_sequence: bool = True):
        # config
        self._interpolation_num = interpolation_num
        self._pred_length = pred_length
        self._is_querying_sequence = is_querying_sequence

        # params
        self.act_mean = np.array([0.0, 0.1, 0.1])
        self.act_rng = np.array([0.25, 0.1, 0.4])

        self.hand_openclose_indices = np.array([
            7,
            9, 10, #11,
            13, 14, #15,
            17, 18, #19,
            20, 22, 23, #24,
                # 26,         29
        ])
        self.hand_leftright_indices = np.array([
            8, 12, 16, 21
        ])

        # states
        self._has_ball = False
        self._action_seq = None
        self._action_idx = 0
        self._latest_action = np.zeros(30)
        self._target_subgoal = None

    def reset(self) -> None:
        self._has_ball = False
        self._action_seq = None
        self._action_idx = 0
        self._latest_action = np.zeros(30)
        self._target_subgoal = None

    def get_action(self, state) -> np.ndarray:
        # init w getting ball

        if self._target_subgoal is None:
            self.set_action_seq_for_object(latest_state=state)

        elif self._target_subgoal == "get_obj":
            if self._action_idx >= self._action_seq.shape[0]:
                # self._target_subgoal = "move_target"
                self.set_action_seq_move_to_origin(latest_state=state)
                # return np.repeat([self._latest_action], self._pred_length, axis=0)
        elif self._target_subgoal == "to_origin_then_target":
            if self._action_idx >= self._action_seq.shape[0]:
                self.set_action_seq_for_target(latest_state=state)
        elif self._target_subgoal == "move_target":
            if not self.check_has_ball(state=state):
                # reset to first, getting ball
                self.set_action_seq_for_object(latest_state=state)
            # else:
            #     return self.get_action_to_target(latest_state=state)
        # else:
        #     self._target_subgoal = "finetune"
        #     return self.get_action_to_target_finetune(latest_state=state)

        # when moving to get ball

        pred_seq = self._action_seq[
                   self._action_idx: min(self._action_seq.shape[0]-1, self._action_idx+self._pred_length)
        ]
        if pred_seq.shape[0] < self._pred_length:
            pred_seq = np.concatenate([
                pred_seq,
                np.repeat(
                    [self._action_seq[self._action_seq.shape[0] - 1]],
                    self._pred_length - pred_seq.shape[0],
                    axis=0
                )
            ])
        self._action_idx += self._pred_length

        if self._is_querying_sequence:
            self._latest_action = pred_seq[pred_seq.shape[0]-1]
        else:
            self._latest_action = pred_seq[0]
        return pred_seq

    # ----------------------------------------
    # Get action seq for different subgoals
    # ----------------------------------------

    def set_action_seq_for_object(self, latest_state):
        # to rest action
        init_action = self._latest_action.copy()
        init_action[self.hand_openclose_indices] = -0.8  # hand open
        init_action[29] = 0.4
        action_seq = np.linspace(self._latest_action, init_action, num=self._interpolation_num//2)

        # move to ball
        next_action = init_action.copy()
        next_action[np.array([0, 2, 1])] -= 0.4 * latest_state[30:33] * np.array([-1, 1, 1])
        next_action[:3] = (next_action[:3] - self.act_mean) / self.act_rng

        next_action[2] += 0.01
        next_action[1] += 0.01

        action_seq = np.concatenate([
            action_seq,
            np.linspace(init_action, next_action, num=self._interpolation_num//2)
        ])
        curr_action = next_action.copy()

        # wrist down first, thumb open
        next_action[3] = 0.08
        # next_action[7] = 0.8
        next_action[25] = -1
        next_action[26] = 0.8
        next_action[27] = -1.0
        next_action[28] = 0
        # DIPs not bent
        next_action[np.array([11, 15, 19, 24])] = -0.25

        # 4 fingers wide
        # for idx in range(4):
        #     next_action[self.hand_leftright_indices[idx]] = 1.0 - 0.6 * idx

        next_action[self.hand_leftright_indices] = np.array([1.0, 0.4, -0.2, -0.8])

        action_seq = np.concatenate([
            action_seq,
            np.linspace(curr_action, next_action, num=int(self._interpolation_num*0.75))
        ])
        curr_action = next_action.copy()

        # hold
        next_action[self.hand_openclose_indices] = 0.6
        next_action[6] = 0  # wrist not go left
        # next_action[7] = 1.0    # wrist max down
        # next_action[26] = 1
        next_action[27] = -1.0
        next_action[28] = -0.5
        next_action[29] = -0.1
        action_seq = np.concatenate([
            action_seq,
            np.linspace(curr_action, next_action, num=self._interpolation_num // 5)
        ])

        # set policy info
        self._action_seq = action_seq
        self._action_idx = 0
        self._target_subgoal = "get_obj"

    def check_has_ball(self, state):
        distance = np.linalg.norm(state[30:33])
        return distance < 0.25  # TODO:???

    def set_action_seq_move_to_origin(self, latest_state):
        init_action = self._latest_action.copy()

        # move to ball
        next_action = init_action.copy()
        next_action[:3] = np.zeros(3)

        action_seq = np.linspace(init_action, next_action, num=self._interpolation_num//2)
        self._action_seq = action_seq
        self._action_idx = 0
        self._target_subgoal = "to_origin_then_target"

    def set_action_seq_for_target(self, latest_state):
        init_action = self._latest_action.copy()

        # move to ball
        next_action = init_action.copy()
        next_action[3] = 0
        next_action[7] = 0
        next_action[np.array([0, 2, 1])] = -0.36 * latest_state[33:36] * np.array([-1, 1, 1])
        # next_action[np.array([0, 2, 1])] -= 0.4 * latest_state[30:33] * np.array([-1, 1, 1])
        next_action[:3] = next_action[:3] / self.act_rng
        # next_action[:3] = (next_action[:3] - self.act_mean) / self.act_rng
        # next_action[2] += 0.12
        # next_action[1] += 0.01

        action_seq = np.linspace(init_action, next_action, num=int(0.75*self._interpolation_num))

        # set policy info
        self._action_seq = action_seq
        self._action_idx = 0
        self._target_subgoal = "move_target"

    def get_action_to_target_finetune(self, latest_state):
        next_action = self._latest_action.copy()
        # wrist and elbow no rotate down
        next_action[3] = 0
        next_action[7] = 0
        pred_seq = np.repeat([next_action.copy()], self._pred_length, axis=0)
        # for i in range(self._pred_length):
        #     ratio = 0.025 + 0.002 * i
        #     pred_seq[i][np.array([0, 2, 1])] -= ratio * latest_state[33:36] * np.array([-1, 1.5, 1])
        #     pred_seq[i][2] += 0.12
        #     pred_seq[i][1] += 0.01

        ratio = 0.025 + 0.002 * np.arange(self._pred_length)
        pred_seq[:, np.array([0, 2, 1])] -= ratio[:, np.newaxis] * np.array([-1, 1.5, 1])[np.newaxis, :]
        pred_seq[:, 2] += 0.12
        pred_seq[:, 1] += 0.01

        # set policy info
        if self._is_querying_sequence:
            self._latest_action = pred_seq[pred_seq.shape[0]-1]
        else:
            self._latest_action = pred_seq[0]

        self._target_subgoal = "finetune"

        return pred_seq

'''
    def get_ball(self, state) -> list[np.ndarray]:
        init_action = np.zeros(30)
        init_action[self.hand_openclose_indices] = -0.8  # hand open
        init_action[29] = 0.4

        # move to ball
        next_action = init_action.copy()
        next_action[np.array([0, 2, 1])] -= 0.4 * state[30:33] * np.array([-1, 1, 1])
        next_action[:3] = (next_action[:3] - self.act_mean) / self.act_rng
        next_action[2] += 0.01
        next_action[1] += 0.01
        action_seq = np.linspace(init_action, next_action, num=self._interpolation_num)
        curr_action = next_action.copy()

        # wrist down first, thumb open
        next_action[3] = 0.08
        # next_action[7] = 0.8
        next_action[25] = -1
        next_action[26] = 0.8
        next_action[27] = -1.0
        next_action[28] = 0
        action_seq = np.concatenate([
            action_seq,
            np.linspace(curr_action, next_action, num=self._interpolation_num)
        ])
        curr_action = next_action.copy()

        # hold
        next_action[self.hand_openclose_indices] = 0.6
        next_action[6] = 0      # wrist not go left
        # next_action[7] = 1.0    # wrist max down
        # next_action[26] = 1
        next_action[27] = -1.0
        next_action[28] = -0.5
        next_action[29] = -0.1
        action_seq = np.concatenate([
            action_seq,
            np.linspace(curr_action, next_action, num=self._interpolation_num//5)
        ])
        curr_action = next_action.copy()

        # move to target`
        next_action[np.array([0, 2, 1])] -= 0.4 * state[33:36] * np.array([-1, 1, 1])
        next_action[:3] = (next_action[:3] - self.act_mean) / self.act_rng
        action_seq = np.concatenate([
            action_seq,
            np.linspace(curr_action, next_action, num=self._interpolation_num)
        ])
        curr_action = next_action.copy()
        return action_seq
'''

