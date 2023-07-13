import numpy as onp
from mdp.utils import DataHistoryNamedTuple


class DataHistory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state_list = []
        self.action_u_list = []
        self.action_v_list = []
        self.reward_u_list = []
        self.reward_v_list = []
        self.done_list = []

    def add(self, state, action_u, action_v, reward_u, reward_v, done):
        self.state_list.append(state)
        self.action_u_list.append(action_u)
        self.action_v_list.append(action_v)
        self.reward_u_list.append(reward_u)
        self.reward_v_list.append(reward_v)
        self.done_list.append(done)

    def add_dict(self, history_dict):
        self.state_list.extend(history_dict['states'])
        self.action_u_list.extend(history_dict['actions_u'])
        self.action_v_list.extend(history_dict['actions_v'])
        self.reward_u_list.extend(history_dict['rewards_u'])
        self.reward_v_list.extend(history_dict['rewards_v'])
        self.done_list.extend(history_dict['dones'])

    def get_history_dict(self):
        history_dict = dict(
            states=self.state_list,
            actions_u=self.action_u_list,
            actions_v=self.action_v_list,
            rewards_u=self.reward_u_list,
            rewards_v=self.reward_v_list,
            dones=self.done_list
        )
        return history_dict

    def get_history_tuple(self):
        history_dict = self.get_history_dict()
        return self.dict_of_list_to_named_tuple(history_dict)

    @staticmethod
    def dict_of_list_to_named_tuple(history_dict):
        history_dict = {k: onp.stack(v) for k, v in history_dict.items()}
        return DataHistoryNamedTuple(**history_dict)
