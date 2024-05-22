import numpy as np

# TODO: add in dimension for batch


class HammerPolicy:
    def __init__(self, pred_length: int = 64, interpolation_num: int = 200):
        """
        Hand-crafted policy for getting action conditioned on current state. The order of action sequences is:
        0. Init (with palm facing down)
        1. Open palm
        2. Hold handle
        3. Move to horizon (from down)
        4. Swing right (to hit target)
        """

        # config
        self._interpolation_num = interpolation_num
        self._pred_length = pred_length

        # Params
        self.hand_openclose_indices = np.array([
            7,
            9, 10,  # 11,
            13, 14,  # 15,
            17, 18,  # 19,
            20, 22, 23,  # 24,
            # 26,         29
        ]) - 4
        self.hand_leftright_indices = np.array([
            8, 12, 16, 21
        ]) - 4

        self.vectorised_get_action = np.vectorize(
            lambda x: self._get_one_action(x),
            signature="(m)->(n,p)"
        )

        # states
        self._keypoint_arr = self._generate_keypoint_arr()

    def get_action(self, state) -> np.ndarray:
        return self.vectorised_get_action(state)

    def unpack_state_info(self, state) -> dict:
        res_dict = {
            "palm_pos": state[33:36],
            "obj_pos": state[36:39],
            "obj_rot": state[39:42],
            "target_pos": state[42:45],
            "nail_impact": state[45]        # 1.0 for pushed in, 0.0 for not
        }
        res_dict["palm_handle_dist"] = np.linalg.norm(res_dict["palm_pos"] - res_dict["obj_pos"])

        return res_dict

    def _get_one_action(self, state) -> np.ndarray:
        state_info = self.unpack_state_info(state)

        if state_info["nail_impact"] > 0.7:
            # completed, do nothing
            return np.repeat(
                [self._keypoint_arr[-1]],
                self._pred_length,
                axis=0
            )

        if state_info["palm_handle_dist"] > 0.05:
            lin_num = int(self._pred_length*0.2)
            return np.concatenate([
                np.linspace(self._keypoint_arr[0], self._keypoint_arr[1], num=lin_num),
                np.linspace(self._keypoint_arr[1], self._keypoint_arr[2], num=self._pred_length-lin_num),
            ])

            # if state_info["palm_pos"][-1] > 0.075:
            #     # move down and open palm
            #     return np.linspace(self._keypoint_arr[0], self._keypoint_arr[1], num=self._pred_length)
            # else:
            #     # grasp
            #     return np.linspace(self._keypoint_arr[1], self._keypoint_arr[2], num=self._pred_length),
            #     # return np.concatenate([
            #     #     np.linspace(self._keypoint_arr[1], self._keypoint_arr[2], num=self._pred_length // 2),
            #     #     np.linspace(self._keypoint_arr[2], self._keypoint_arr[], num=self._pred_length // 2),
            #     # ])
        else:
            lin_num = int(self._pred_length * 0.4)
            return np.concatenate([
                np.linspace(self._keypoint_arr[2], self._keypoint_arr[3], num=lin_num),
                np.linspace(self._keypoint_arr[3], self._keypoint_arr[4], num=self._pred_length - lin_num)
            ])

    # ----------------------------------------
    # Get action seq for different keypoints
    # ----------------------------------------
    def _generate_keypoint_arr(self) -> np.ndarray:
        res_lst = []

        # -------------------------
        # Lean down and open palm
        # -------------------------
        action = np.zeros(26)

        action[21] = -1
        action[22] = 0.85
        action[23] = 1
        action[24] = 0
        # DIPs not bent
        action[np.array([7, 11, 15, 20])] = -0.25
        # 4 fingers wide
        action[self.hand_leftright_indices] = np.array([1.0, 0.4, -0.2, -0.8])

        res_lst.append(action.copy())                   # init, open_palm

        # action[2] += 0.05 * min(10, self._retried)
        # next_action[7] = 0.8

        action[0] = 0.6
        res_lst.append(action.copy())                   # move_down

        # -------------------------
        # Hold the handle
        # -------------------------
        action[self.hand_openclose_indices] = 0.6
        action[2] = 0  # wrist not go left
        action[21] = 0.8
        action[22] = 0.8
        action[23] = 1
        action[24] = 0
        action[25] = -1
        res_lst.append(action.copy())                   # hold_handle

        # -------------------------
        # Move up to horizontal
        # -------------------------
        action[0] = 0
        res_lst.append(action.copy())                   # hold_to_horizon

        # -------------------------
        # Swing to the right
        # -------------------------
        # action[0] = 0.1
        action[1] = -1  # rotate arm to right
        action[2] = -1  # rotate wrist to right
        res_lst.append(action.copy())                   # swing_right

        return np.array(res_lst)
