import numpy as np


class DoorPolicy:
    def __init__(self, pred_length: int = 64, interpolation_num: int = 200):
        """
        Hand-crafted policy for getting action conditioned on curr state. The order of action sequences is:
        1. Move up and right
        2. Open hand
        3. Move far (from camera)
        4. Close hand to grasp handle
        5. Rotate clockwise 90 deg
        6. Rotate (Pull) to the right 90 deg
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
        ]) - 2
        self.hand_leftright_indices = np.array([
            8, 12, 16, 21
        ]) - 2

        self.vectorised_get_action = np.vectorize(
            lambda x: self._get_one_action(x),
            signature="(m)->(n,p)"
        )

        # states
        self._keypoint_arr = self._generate_keypoint_arr()

    def unpack_state_info(self, state) -> dict:
        palm_handle_dist = np.linalg.norm(state[35:38])
        return {
            "latch_val": state[27],
            "door_val": state[28],
            "palm_pos": state[29:32],
            "handle_pos": state[32:35],
            "palm_handle_dist": palm_handle_dist,
            "door_opened": state[38]
        }

    def get_action(self, state) -> np.ndarray:
        return self.vectorised_get_action(state)

    def _get_one_action(self, state) -> np.ndarray:
        state_info = self.unpack_state_info(state)

        if state_info["door_opened"] > 0 or state_info["door_val"] > 0.3:
            # completed, do nothing
            return np.repeat(
                [self._keypoint_arr[-1]],
                self._pred_length,
                axis=0
            )

        if state_info["latch_val"] < 0.15:
            val = state_info["palm_pos"]

            # if state_info["palm_pos"][-1] > 0.24:
            if state_info["palm_pos"][-1] > 0.22:
                # move closer to handle
                lin_num = self._pred_length // 3
                # return np.linspace(self._keypoint_arr[0], self._keypoint_arr[3], num=self._pred_length)
                return np.concatenate([
                    np.linspace(self._keypoint_arr[0], self._keypoint_arr[1], num=lin_num),  # init to up_and_right
                    np.linspace(self._keypoint_arr[1], self._keypoint_arr[2], num=lin_num),  # up_and_right to hand_open
                    np.linspace(
                        self._keypoint_arr[2],
                        self._keypoint_arr[3],
                        num=self._pred_length - 2 * lin_num
                    ),  # hand_open to open_on_handle
                ])

            lin_num = self._pred_length // 2
            return np.linspace(
                self._keypoint_arr[3],
                self._keypoint_arr[5],
                num=self._pred_length
            )
            return np.concatenate([
                np.linspace(self._keypoint_arr[3], self._keypoint_arr[4], num=lin_num),  # init to up_and_right
                np.linspace(
                    self._keypoint_arr[4],
                    self._keypoint_arr[5],
                    num=self._pred_length - lin_num
                )
            ])

            if state_info["palm_handle_dist"] < 0.1:
                return np.linspace(
                    self._keypoint_arr[3],
                    self._keypoint_arr[4],
                    num=self._pred_length
                )  # open_on_handle to closed_on_handle

            # rotate down
            return np.linspace(
                self._keypoint_arr[4],
                self._keypoint_arr[5],
                num=self._pred_length
            )

        if state_info["palm_handle_dist"] < 0.1:
            to_right_start = self._keypoint_arr[5].copy()
            to_right_start[1] -= 0.1
            return np.linspace(
                to_right_start,
                self._keypoint_arr[6],
                num=self._pred_length
            )
        else:
            lin_num = self._pred_length // 2
            to_right_start = self._keypoint_arr[5].copy()
            to_right_start[1] -= 0.1
            return np.linspace(self._keypoint_arr[3], self._keypoint_arr[5], num=self._pred_length)
            # return np.concatenate([
            #     np.linspace(self._keypoint_arr[3], self._keypoint_arr[5], num=lin_num),
            #     np.linspace(to_right_start, self._keypoint_arr[6], num=self._pred_length - lin_num)
            # ])
        # hold alr, rotate

        # lin_num = self._pred_length // 2
        to_right_start = self._keypoint_arr[5].copy()
        to_right_start[1] -= 0.1
        return np.linspace(
            to_right_start,
            self._keypoint_arr[6],
            num=self._pred_length
        )

    def _generate_keypoint_arr(self) -> np.ndarray:
        res_dict = []

        # ------------------------------
        # move to handle and open palm
        # ------------------------------
        curr_action = np.zeros(28)
        curr_action[3] = -0.25
        curr_action[0] = -0.2  # move back to open palm easier
        next_action = curr_action.copy()
        res_dict.append(next_action.copy())  # init

        # Move up & right a bit
        curr_action = next_action.copy()
        next_action[1] = -0.15
        next_action[2] = -0.05
        # action_seq = np.linspace(curr_action, next_action, num=self._interpolation_num // 3)
        res_dict.append(next_action.copy())  # up_and_right

        # open hand
        curr_action = next_action.copy()
        next_action[23] = -1
        next_action[24] = 0.85
        next_action[25] = 1
        next_action[26] = 0
        # DIPs not bent
        next_action[np.array([9, 13, 17, 22])] = -0.25
        # hand open
        next_action[self.hand_openclose_indices] = -0.8
        # 4 fingers wide
        next_action[self.hand_leftright_indices] = np.array([1.0, 0.4, -0.2, -0.8])

        # action_seq = np.concatenate([
        #     action_seq,
        #     np.linspace(curr_action, next_action, num=self._interpolation_num // 3)
        # ])
        # self._action_seq = action_seq
        res_dict.append(next_action.copy())  # hand_open

        # Then Move farther from camera , need to be separate
        curr_action = next_action.copy()
        # next_action[0] = 0.15
        next_action[0] = 0.25  # z, to closer to camera
        next_action[1] = -0.2  # slightly up
        # action_seq = np.concatenate([
        #     action_seq,
        #     np.linspace(curr_action, next_action, num=self._interpolation_num // 3)
        # ])
        res_dict.append(next_action.copy())  # open_on_handle

        # ------------------------------
        # Close grip to hold handle
        # ------------------------------
        curr_action = next_action.copy()
        next_action[self.hand_openclose_indices] = 0.6
        next_action[4] = 0  # wrist not go left
        next_action[23] = 0.8
        next_action[24] = 0.8
        next_action[25] = 1
        next_action[26] = 0
        next_action[27] = -1
        # action_seq = np.concatenate([
        #     action_seq,
        #     np.linspace(curr_action, next_action, num=self._interpolation_num // 5)
        # ])

        res_dict.append(next_action.copy())  # closed_on_handle

        # ------------------------------
        # Rotate down to open
        # ------------------------------
        curr_action = next_action.copy()
        next_action[3] = 0
        next_action[1] += 0.25  # to down, i.e. y
        next_action[2] += 0.3  # to left, i.e. x
        # action_seq = np.concatenate([
        #     action_seq,
        #     np.linspace(curr_action, next_action, num=self._interpolation_num // 2)
        # ])
        res_dict.append(next_action.copy())  # rotated_down

        # ------------------------------
        # Pull the handle right
        # ------------------------------
        curr_action = next_action.copy()
        next_action[2] -= 0.55  # y back to right
        next_action[0] -= 0.45  # z, to closer to camera
        next_action[5] += 1  # wrist down also
        # action_seq = np.concatenate([
        #     action_seq,
        #     np.linspace(curr_action, next_action, num=self._interpolation_num // 2)
        # ])
        res_dict.append(next_action.copy())  # rotated_right

        return np.array(res_dict)
