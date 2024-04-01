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
from poker.decisionmaker.dqn.processor import LegalMovesProcessor


log = logging.getLogger(__name__)


# from poker.gui.action_and_signals import StrategyHandler
# from poker.decisionmaker.dqn.dqn_decisionmaker import DQNDecision
# strategy = StrategyHandler()
# strategy.read_strategy()
# d = DQNDecision(None, None, strategy, None)


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


class DQNDecision(DecisionBase):

    def __init__(self, table, history, strategy, game_logger):
        """Initialize DQN agent."""
        nb_actions = len(DecisionTypes)

        model_name = strategy.selected_strategy["modelName"]
        nb_steps_warmup = strategy.selected_strategy["nbStepsWarmup"]
        nb_steps = strategy.selected_strategy["nbSteps"]
        num_opponents = strategy.selected_strategy["numOpponents"]

        # load model from disk
        self.model = self.load(model_name)

        # create memory
        window_length = 1
        memory_limit = int(nb_steps / 5)
        memory = SequentialMemory(limit=memory_limit, window_length=window_length)

        # maybe these should be configurable in the future
        policy = TrumpPolicy()
        processor = LegalMovesProcessor(num_opponents, nb_actions)

        # create DQN
        self.dqn = DQNAgent(
            model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
            target_model_update=1e-2, policy=policy, processor=processor, batch_size=500,
            train_interval=100, enable_double_dqn=False
        )

        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    def load(self, model_name: str):
        """Load a model."""

        # Load the architecture
        with open(f'decisionmaker/dqn/models/dqn_{model_name}_json.json', 'r') as architecture_json:
            dqn_json = json.load(architecture_json)

        model = model_from_json(dqn_json)
        model.load_weights(f'decisionmaker/dqn/models/dqn_{model_name}_weights.h5')

        return model

    def make_decision(self, table, history, strategy, game_logger):
        # get observation
        observation = [table.equity]

        # process the observation (e.g., to set legal moves)
        observation = self.dqn.processor.process_observation(observation)

        # get action using a forward pass
        action = self.dqn.forward(observation)

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
