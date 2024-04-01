"""Define the observation space."""

from collections import OrderedDict
from typing import List
from typing import Optional

import numpy as np

from poker.decisionmaker.dqn.action import Action
from poker.decisionmaker.dqn.action import ActionBlind


class PlayerBase:

    def action(self, action_space, observation, info):
        pass


class PlayerData:
    """Main player data."""

    def __init__(self, num_actions: int):
        """Initialize."""
        self.data = OrderedDict()

        self.data['stack'] = 0
        self.data['is_dealer'] = False
        self.data['blind'] = 0
        self.data['equity'] = 0
        self.data['amount_to_call'] = 0

        for i in range(num_actions):
            self.data[f'is_legal_action_{i + 1}'] = False

        self.num_actions = num_actions

    def ndim(self):
        """Get number of dimensions."""
        return len(self.data)

    def labels(self):
        return list(self.data.keys())

    def to_array(self):
        """Convert to numpy array."""
        return np.array(list(self.data.values()))

    def __str__(self):
        """Get print-friendly version of the observation."""
        s = "Player data:\n"
        for k, v in self.data.items():
            s += f"  {k}: {v}\n"
        return s[:-1]

    def set(
            self,
            main_player: PlayerBase,
            dealer_pos: int,
            legal_moves: List,
            amount_to_call: float,
            equity: float,
            small_blind: float,
            big_blind: float,
            pot_norm: Optional[float] = 1.0
    ):
        """Set all the values."""
        self.data['stack'] = main_player.stack / pot_norm
        self.data['is_dealer'] = main_player.seat == dealer_pos

        for i in range(self.num_actions):
            self.data[f"is_legal_action_{i + 1}"] = Action(i) in legal_moves

        self.data['amount_to_call'] = amount_to_call
        self.data['equity'] = equity

        if main_player.actions:
            if ActionBlind.SMALL in main_player.actions:
                self.data['blind'] = small_blind / pot_norm
            elif ActionBlind.BIG in main_player.actions:
                self.data['blind'] = big_blind / pot_norm


class CommunityData:
    """Data available to everybody."""

    def __init__(self, num_opponents: int):
        """Initialize."""
        self.data = OrderedDict()

        self.data['is_preflop'] = False
        self.data['is_flop'] = False
        self.data['is_turn'] = False
        self.data['is_river'] = False
        self.data['community_pot'] = 0
        self.data['current_round_pot'] = 0

        for i in range(num_opponents):
            self.data[f'opp_stack_{i + 1}'] = 0
            self.data[f'opp_blind_{i + 1}'] = 0

    def ndim(self):
        """Get number of dimensions."""
        return len(self.data)

    def labels(self):
        return list(self.data.keys())

    def to_array(self):
        """Convert to numpy array."""
        return np.array(list(self.data.values()))

    def __str__(self):
        """Get print-friendly version of the observation."""
        s = "Community data:\n"
        for k, v in self.data.items():
            s += f"  {k}: {v}\n"
        return s[:-1]

    def set(
            self,
            opponents: List[PlayerBase],
            community_pot: float,
            current_round_pot: float,
            stage: int,
            small_blind: float,
            big_blind: float,
            pot_norm: Optional[float] = 1.0
    ):
        """Set all the values."""
        self.data['community_pot'] = community_pot / pot_norm
        self.data['current_round_pot'] = current_round_pot / pot_norm

        stage_val = min(stage, 3)
        self.data['is_preflop'] = self.data['is_flop'] = self.data['is_turn'] = self.data['is_river'] = False
        if stage_val == 0:
            self.data['is_preflop'] = True
        elif stage_val == 1:
            self.data['is_flop'] = True
        elif stage_val == 2:
            self.data['is_turn'] = True
        else:
            self.data['is_river'] = True
        
        for idx, opponent in enumerate(opponents):
            self.data[f'opp_stack_{idx + 1}'] = opponent.stack / pot_norm
            if opponent.actions:
                if ActionBlind.SMALL in opponent.actions:
                    self.data[f'opp_blind_{idx + 1}'] = small_blind / pot_norm
                elif ActionBlind.BIG in opponent.actions:
                    self.data[f'opp_blind_{idx + 1}'] = big_blind / pot_norm


class StageData:
    """Preflop, flop, turn and river."""

    def __init__(self, num_players: int):
        """Initialize."""
        self.data = OrderedDict()

        for i in range(num_players):
            self.data[f"player_call_{i + 1}"] = False
            self.data[f"player_raise_{i + 1}"] = False
            self.data[f"player_min_call_action_{i + 1}"] = 0
            self.data[f"player_contrib_{i + 1}"] = 0
            self.data[f"player_stack_action_{i + 1}"] = 0
            self.data[f"player_community_pot_action_{i + 1}"] = 0

    def ndim(self):
        """Get number of dimensions."""
        return len(self.data)

    def to_array(self):
        """Convert to numpy array."""
        return np.array(list(self.data.values()))

    def labels(self):
        return list(self.data.keys())

    def __str__(self):
        """Get print-friendly version of the observation."""
        s = "Stage data:\n"
        for k, v in self.data.items():
            s += f"  {k}: {v}\n"
        return s[:-1]

    def set(
                self,
                player_pos: int,
                stack: float,
                action: int,
                min_call: float,
                community_pot: float,
                contribution: float,
                pot_norm: Optional[float] = 1.0
        ):
            """Set all the values."""
            self.data[f"player_call_{player_pos + 1}"] = action == Action.CALL
            self.data[f"player_raise_{player_pos + 1}"] = action == Action.RAISE_POT
            self.data[f"player_min_call_action_{player_pos + 1}"] = min_call / pot_norm
            self.data[f"player_community_pot_action_{player_pos + 1}"] = community_pot / pot_norm
            self.data[f"player_contrib_{player_pos + 1}"] += contribution / pot_norm
            self.data[f"player_stack_action_{player_pos + 1}"] = stack / pot_norm


class Observation:
    """Class to encapsulate all the observations."""

    num_stages = 8

    def __init__(self, num_opponents: int, num_actions: int):
        self.player_data = PlayerData(num_actions)
        self.community_data = CommunityData(num_opponents)
        self.stage_data = [StageData(num_opponents + 1) for _ in range(self.num_stages)]

    def ndim(self):
        """Get number of dimensions."""
        return self.player_data.ndim() + self.community_data.ndim()

    def labels(self):
        return self.player_data.labels() + self.community_data.labels()

    def to_array(self):
        """Convert to numpy array."""
        return np.concatenate([self.player_data.to_array(), self.community_data.to_array()])

    def __str__(self):
        """Get print-friendly version of the observation."""
        return f"{str(self.player_data)}\n{str(self.community_data)}"
