# TODOs:
# - [ ] We might want to turn .winners into a
#       numpy.ndarray(dtype=numpy.uint32) for efficiency.

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import heapq
import collections
import types

# Configurable assembly model for simulations
# Author Daniel Mitropolsky, 2018

EMPTY_MAPPING = types.MappingProxyType({})


class Area:
    """A brain area.

    Attributes:
      name: the area's name (symbolic tag).
      n: number of neurons in the area.
      k: number of neurons that fire in this area.
      beta: Default value for activation-`beta`.
      winners: List of winners, as set by previous action.
      saved_winners: List of lists of all winners, per-round.
      fixed_assembly: Whether the assembly (of winners) in this area
        is considered frozen.
    """

    def __init__(self, name, n, k, beta=0.05, beta_by_area=None):
        """Initializes the instance.

        Args:
          name: Area name (symbolic tag), must be unique.
          n: number of neurons(?)
          k: number of firing neurons when activated.
          beta: default activation-beta.
          beta_by_area: default activation-beta by area (otherwise all set to beta)
        """
        self.name = name
        self.n = n
        self.k = k
        self.beta = beta
        self.beta_by_area = {} if beta_by_area is None else beta_by_area
        self.winners = []
        # Value of `winners` since the last time that `.project()` was called.
        # only to be used inside `.project()` method.
        self.saved_winners = []
        self.fixed_assembly = False

    def update_area_beta(self, name, new_beta):
        self.beta_by_area[name] = new_beta

    def fix_assembly(self):
        if not self.winners:
            raise ValueError(f"Area {self.name!r} does not have assembly; cannot fix.")
        self.fixed_assembly = True

    def unfix_assembly(self):
        self.fixed_assembly = False

    def get_num_ever_fired(self):
        return [winner for L in self.saved_winners for winner in L]


@dataclass
class Stimulus:
    """A stimulus.

    beta: The plasticity-beta for this stimulus.
    name: The name of the stimulus.
    size: Number of neurons that fire in this stimulus.
    """

    beta: float
    name: str
    size: int


class Brain:
    """A model brain.

    p: neuron connection-probability.
    area_by_name: Mapping from brain area-name tag to corresponding Area instance.
    stimuli: Mapping from stimulus-name to corresponding Stimulus instance.
    fully_simulate: Boolean flag, whether to fully simulate weight updates.

    Attributes:
      area_by_name: Mapping from brain area-name tag to corresponding Area
        instance. (Original code: .areas).
      stimuli: Mapping from a stimulus-name to a Stimulus instance.
      connectomes_by_stimulus: Mapping from stimulus-name to a mapping
        from area-name to an activation-vector for that area.
        (Original code: .stimuli_connectomes)
      connectomes: Mapping from a 'source' area-name to a mapping from a
        'target' area-name to a [source_size, target_size]-bool-ndarray
        with connections. (TODO(tfish): Rename and replace with index-vector.)
        The source-index, respectively target-index, reference neurons in the
        "active assembly".
      p: Neuron connection-probability.
      save_size: Boolean flag, whether to save sizes.
      save_winners: Boolean flag, whether to save winners.
      disable_plasticity: Debug flag for disabling plasticity.
    """

    def __init__(
        self,
        p: float,
        area_by_name: Dict[str, Area],
        stimuli: Dict[str, Stimulus],
        seed=0,
        fully_simulate=True,
    ):
        self.p = p
        self.area_by_name = area_by_name
        self.stimuli = stimuli
        self.disable_plasticity = False
        self._rng = np.random.default_rng(seed=seed)
        self.fully_simulate = fully_simulate

        self.connectomes_by_stimulus = {}
        self.connectomes = {area_name: {} for area_name in area_by_name}

        # Initialize stimulus -> area
        for stim_name, stimulus in self.stimuli.items():
            self.connectomes_by_stimulus[stim_name] = {}
            for area_name in self.area_by_name:
                self.connectomes_by_stimulus[stim_name][area_name] = self._rng.binomial(
                    stimulus.size, self.p, size=self.area_by_name[area_name].n
                ).astype(np.float32)

        # Initialize area -> self:
        for area_name, area in self.area_by_name.items():
            self.connectomes[area_name][area_name] = self._rng.binomial(
                1, self.p, size=(area.n, area.n)
            ).astype(np.float32)

        # Initialize area -> other areas:
        for area_name_1, area_1 in self.area_by_name.items():
            for area_name_2, area_2 in self.area_by_name.items():
                if area_name_2 not in area_1.beta_by_area:
                    area_1.beta_by_area[area_name_2] = area_1.beta
                if area_name_1 == area_name_2:
                    continue
                self.connectomes[area_name_1][area_name_2] = self._rng.binomial(
                    1, self.p, size=(area_1.n, area_2.n)
                ).astype(np.float32)

        # Set area -> self diagonal to 0
        for area_name, area in self.area_by_name.items():
            np.fill_diagonal(self.connectomes[area_name][area_name], 0)

        # Normalize connectome weights
        for area_name in self.area_by_name:
            self.normalize_connectome_weights(area_name)

    def normalize_connectome_weights(
        self,
        area_name: str,
    ) -> None:
        """
        normalize the connectome weights between areas and stimuli such that the
        sum of incoming weights to each neuron is 1

        area_name: str
            name of the area
        """
        aggregate_weights = np.zeros(self.area_by_name[area_name].n, dtype=np.float32)
        for stim in self.stimuli:
            aggregate_weights += self.connectomes_by_stimulus[stim][area_name]

        for from_area in self.area_by_name:
            aggregate_weights += self.connectomes[from_area][area_name].sum(axis=0)

        for stim in self.stimuli:
            self.connectomes_by_stimulus[stim][area_name] /= aggregate_weights

        for from_area in self.area_by_name:
            self.connectomes[from_area][area_name] /= aggregate_weights

    def update_plasticity(self, from_area, to_area, new_beta):
        self.area_by_name[to_area].beta_by_area[from_area] = new_beta

    def update_stimulus_plasticity(self, stim, new_beta):
        self.stimuli[stim].beta = new_beta

    def update_plasticities(
        self,
        area_update_map: Dict[str, Dict[str, float]] = EMPTY_MAPPING,
        stim_update_map: Dict[str, float] = EMPTY_MAPPING,
    ):
        """
        Update plasticity values for areas and stimuli.

        area_update_map: Dict[str, Dict[str, float]]
            from_area -> {to_area: new_beta}
        stim_update_map: Dict[str, float]
            stim -> new_beta

        """
        # area_update_map consists of area1: list[ (area2, new_beta) ]
        # represents new plasticity FROM area2 INTO area1
        for to_area, update_rules in area_update_map.items():
            for from_area, new_beta in update_rules:
                self.update_plasticity(from_area, to_area, new_beta)

        # stim_update_map consists of area: list[ (stim, new_beta) ]f
        # represents new plasticity FROM stim INTO area
        for stim, new_beta in stim_update_map.items():
            self.update_stimulus_plasticity(stim, new_beta)

    def create_fixed_assembly(self, area_name, index):
        """
        create a fixed assembly in area_name starting at index of size k
        """
        area = self.area_by_name[area_name]
        area.winners = list(range(index, index + k))
        area.fix_assembly()

    def project(self, areas_by_stim, dst_areas_by_src_area, verbose=0):
        """
        Validate stim_area, area_area well defined
        areas_by_stim: {"stim1":["A"], "stim2":["C","A"]}
        dst_areas_by_src_area: {"A":["A","B"],"C":["C","A"]}

        Then run project_into for each area in areas_by_stim and dst_areas_by_src_area
        """

        stim_in = collections.defaultdict(list)
        area_in = collections.defaultdict(list)

        for stim, areas in areas_by_stim.items():
            if stim not in self.stimuli:
                raise IndexError(f"Not in brain.stimuli: {stim}")
            for area_name in areas:
                if area_name not in self.area_by_name:
                    raise IndexError(f"Not in brain.areas: {area_name}")
                stim_in[area_name].append(stim)
        for from_area_name, to_area_names in dst_areas_by_src_area.items():
            if from_area_name not in self.area_by_name:
                raise IndexError(from_area_name + " not in brain.areas")
            for to_area_name in to_area_names:
                if to_area_name not in self.area_by_name:
                    raise IndexError(f"Not in brain.areas: {to_area_name}")
                area_in[to_area_name].append(from_area_name)

        to_update_area_names = stim_in.keys() | area_in.keys()

        winners = {}
        for area_name in to_update_area_names:
            area = self.area_by_name[area_name]
            winners[area_name] = self.project_into(
                area, stim_in[area_name], area_in[area_name], verbose
            )

        # after all areas have been projected into, update the winners
        # need to wait so that new winners don't affect other areas in the same round
        for area_name in to_update_area_names:
            area = self.area_by_name[area_name]
            area.winners = winners[area_name]
            area.saved_winners.append(area.winners)

    def get_input(
        self,
        target_area: Area,
        from_stimuli: List[str],
        from_areas: List[str],
        verbose=0,
    ) -> List[float]:
        """
        Project stimuli and areas into target_area. Aggregate.

        Returns the sum of inputs from stimuli and areas projected into target_area.
        """
        if target_area.fixed_assembly:
            return target_area.winners

        target_area_name = target_area.name
        inputs = np.zeros(target_area.n, dtype=np.float32)
        for stim in from_stimuli:
            stim_inputs = self.connectomes_by_stimulus[stim][target_area_name]
            inputs += stim_inputs
        for from_area_name in from_areas:
            connectome = self.connectomes[from_area_name][target_area_name]
            for w in self.area_by_name[from_area_name].winners:
                inputs += connectome[w]

        if verbose >= 2:
            print("inputs:", inputs)

        return inputs

    def update_weights_by_winner(
        self,
        winners: np.ndarray,
        target_area: Area,
        from_stimuli: List[str],
        from_areas: List[str],
        verbose=0,
    ):
        """
        Old version of updating weights, where we multiply by a constant factor
        """
        # for i in repeat_winners, stimulus_inputs[i] *= (1+beta)
        num_inputs_processed = 0
        for stim in from_stimuli:
            connectomes = self.connectomes_by_stimulus[stim]
            target_connectome = connectomes[target_area.name]
            stim_to_area_beta = self.stimuli[stim].beta
            if self.disable_plasticity:
                stim_to_area_beta = 0.0
            for i in winners:
                target_connectome[i] *= 1 + stim_to_area_beta
            if verbose >= 2:
                print(f"{stim} now looks like: ")
                print(self.connectomes_by_stimulus[stim][target_area.name])
            num_inputs_processed += 1

        # connectome for each in_area->area
        # for each i in repeat_winners, for j in in_area.winners, connectome[j][i] *= (1+beta)
        for from_area_name in from_areas:
            from_area_winners = self.area_by_name[from_area_name].winners
            from_area_connectomes = self.connectomes[from_area_name]
            the_connectome = from_area_connectomes[target_area.name]
            area_to_area_beta = (
                0
                if self.disable_plasticity
                else target_area.beta_by_area[from_area_name]
            )
            for i in winners:
                for j in from_area_winners:
                    the_connectome[j, i] *= 1.0 + area_to_area_beta

    def update_weights_by_potential(
        self,
        target_area: Area,
        new_input: np.ndarray,
        winners: List[int],
        from_stimuli: List[str],
        from_areas: List[str],
    ):
        """
        Update weights by potential

        new_input: np.ndarray (n,)
            aggregate incoming potential for each neuron
        winners: list of length k
            idx of new winners for target_area
        from_stimuli: list of str
            stimuli projected into target_area
        from_areas: list of str
            areas projected into target_area


        for each weight from neuron i to neuron j
            if j is not a winner, do nothing
            if j is a winner:
                if j fired, weight = weight * (1 + beta * (1 - potential))
                if j didn't fire, weight = weight * (1 - beta * potential)
        """
        multihot_winners = np.zeros(target_area.n, dtype=np.float32)
        multihot_winners[winners] = 1.0
        potential_difference = multihot_winners - new_input

        for stim_name, stim in self.stimuli.items():
            if stim_name not in from_stimuli:
                stim_winners = np.zeros(target_area.n, dtype=np.float32)
            else:
                stim_winners = np.ones(target_area.n, dtype=np.float32)
            the_connectome = self.connectomes_by_stimulus[stim_name][target_area.name]
            stim_to_area_beta = stim.beta
            if self.disable_plasticity:
                stim_to_area_beta = 0.0

            the_connectome *= (
                1.0 + stim_to_area_beta * stim_winners * potential_difference
            )

        for from_area_name, from_area in self.area_by_name.items():
            from_area_winners = np.zeros(target_area.n, dtype=np.float32)
            if from_area_name in from_areas:
                from_area_winners[from_area.winners] = 1.0
            the_connectome = self.connectomes[from_area_name][target_area.name]
            area_to_area_beta = (
                0
                if self.disable_plasticity
                else target_area.beta_by_area[from_area_name]
            )
            the_connectome *= 1.0 + area_to_area_beta * np.outer(
                from_area_winners, potential_difference
            )

    def project_into(
        self,
        target_area: Area,
        from_stimuli: List[str],
        from_areas: List[str],
        verbose=0,
    ) -> List[int]:
        """
        Project stimuli and areas into target_area.
        - Generate new winners for target_area.
        - Update connectomes for target_area with plasticity.

        return: List of new winners for target_area.
        """
        if verbose >= 1:
            print(
                f"Projecting {', '.join(from_stimuli)} "
                f" and {', '.join(from_areas)} into {target_area.name}"
            )

        new_input = self.get_input(target_area, from_stimuli, from_areas, verbose)
        winners = heapq.nlargest(
            target_area.k, range(target_area.n), new_input.__getitem__
        )

        if verbose >= 2:
            print(f"new_winners: {winners}")

        if self.fully_simulate:
            self.update_weights_by_potential(
                target_area, new_input, winners, from_stimuli, from_areas
            )
        else:
            self.update_weights_by_winner(
                winners, target_area, from_stimuli, from_areas, verbose
            )

        return winners
