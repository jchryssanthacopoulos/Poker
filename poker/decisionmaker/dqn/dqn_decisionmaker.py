"""Decisionmaker based on a DQN agent."""

from enum import Enum
import json
import logging

import numpy as np
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam

from poker.decisionmaker.base import DecisionBase
from poker.decisionmaker.dqn.action import Action
from poker.decisionmaker.dqn.observation import Observation
from poker.decisionmaker.dqn.processor import LegalMovesProcessor


log = logging.getLogger(__name__)


class DecisionTypes(Enum):
    fold, check, call, bet, bet_max = ['Fold', 'Check', 'Call', 'Bet', 'Bet max']


class TrumpPolicy(BoltzmannQPolicy):
    """Custom policy when making decision based on neural network."""

    def select_action(self, q_values: np.array):
        """Return the selected action.

        Arguments
            q_values: List of the estimations of Q for each action

        Returns
            Selection action

        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        log.info(f"Chosen action by keras-rl {action} - probabilities: {probs}")

        return action


class PlayerShell:
    """Player shell."""

    def __init__(self, stack_size, name):
        """Initiaization of an agent."""
        self.stack = stack_size
        self.seat = None
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.agent_obj = None


class DQNDecision(DecisionBase):

    def __init__(self, strategy):
        """Initialize DQN agent."""
        self.nb_actions = len(DecisionTypes)
        self.num_opponents = strategy.selected_strategy["numOpponents"]
        self.small_blind = strategy.selected_strategy["smallBlind"]
        self.big_blind = strategy.selected_strategy["bigBlind"]
        self.pot_norm = 100 * self.big_blind

        model_name = strategy.selected_strategy["modelName"]
        nb_steps_warmup = strategy.selected_strategy["nbStepsWarmup"]
        nb_steps = strategy.selected_strategy["nbSteps"]

        # load model from disk
        self.model = self.load(model_name)

        # create memory
        window_length = 1
        memory_limit = int(nb_steps / 5)
        memory = SequentialMemory(limit=memory_limit, window_length=window_length)

        # maybe these should be configurable in the future
        policy = TrumpPolicy()
        processor = LegalMovesProcessor(self.num_opponents, self.nb_actions)

        # create DQN
        self.dqn = DQNAgent(
            model=self.model, nb_actions=self.nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
            target_model_update=1e-2, policy=policy, processor=processor, batch_size=500,
            train_interval=100, enable_double_dqn=False
        )

        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        # set these other required variables
        self.finalCallLimit = 99999999
        self.finalBetLimit = 99999999
        self.outs = 0
        self.pot_multiple = 0
        self.maxCallEV = 0

    def load(self, model_name: str):
        """Load a model."""

        # Load the architecture
        with open(f'decisionmaker/dqn/models/dqn_{model_name}_json.json', 'r') as architecture_json:
            dqn_json = json.load(architecture_json)

        model = model_from_json(dqn_json)
        model.load_weights(f'decisionmaker/dqn/models/dqn_{model_name}_weights.h5')

        return model

    def make_decision(self, table, history, strategy, game_logger):
        # select equity
        if table.gameStage != 'PreFlop' and strategy.selected_strategy['use_relative_equity']:
            table.equity = table.relative_equity
        else:
            table.equity = table.abs_equity

        # these need to be set
        table.minCall = 9999
        table.minBet = 9999
        table.totalPotValue = 9999
        table.minEquityCall = 9999
        table.minEquityBet = 9999
        table.power1 = 0.2
        table.power2 = 0.2
        table.bigBlind = float(strategy.selected_strategy['bigBlind'])
        table.smallBlind = float(strategy.selected_strategy['smallBlind'])

        # get observation
        observation = self._get_observation(table)

        # process the observation (e.g., to set legal moves)
        observation = self.dqn.processor.process_observation(observation)

        # get action using a forward pass
        action = self.dqn.forward(observation)
        log.info(f"Ran inference with observation = {observation}")

        # process the action (e.g., to set to legal move)
        action = self.dqn.processor.process_action(action)

        # convert to form recognized by mouse mover
        # TODO: think of a more clever way to do this
        if action == 0:
            decision = DecisionTypes.fold
        elif action == 1:
            decision = DecisionTypes.check
        elif action == 2:
            decision = DecisionTypes.call
        elif action == 3:
            decision = DecisionTypes.bet
        else:
            decision = DecisionTypes.bet_max

        self.decision = decision.value

    def _get_observation(self, table):
        # make sure current funds are the latest for all players
        table.get_players_funds()

        main_player = PlayerShell(name='keras-rl', stack_size=table.player_funds[0])
        main_player.seat = 0
        main_player.actions = []

        other_players = []
        for i in range(self.num_opponents):
            bot = PlayerShell(name=f'bot_{i}', stack_size=table.player_funds[i + 1])
            bot.actions = []
            other_players.append(bot)

        if table.gameStage == 'PreFlop':
            game_stage = 0
        elif table.gameStage == 'Flop':
            game_stage = 1
        elif table.gameStage == 'Turn':
            game_stage = 2
        else:
            game_stage = 3

        legal_moves = self._get_legal_moves(table)

        observation = Observation(self.num_opponents, self.nb_actions)

        observation.community_data.set(
            other_players,
            table.total_pot,
            table.current_round_pot,
            game_stage,
            self.small_blind,
            self.big_blind,
            self.pot_norm
        )

        observation.player_data.set(
            main_player,
            table.dealer_position,
            legal_moves,
            float(table.currentCallValue) / self.pot_norm,
            table.equity,
            self.small_blind,
            self.big_blind,
            self.pot_norm
        )

        observation = observation.to_array()

        return observation

    def _get_legal_moves(self, table):
        """Determine what moves are allowed in the current state based on OCR."""
        legal_moves = []

        if table.checkButton:
            legal_moves.append(Action.CHECK)
        else:
            legal_moves.append(Action.CALL)
            legal_moves.append(Action.FOLD)

        if table.bet_button_found:
            legal_moves.append(Action.RAISE_POT)

        if table.allInCallButton:
            legal_moves.append(Action.ALL_IN)

        return legal_moves
