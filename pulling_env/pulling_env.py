# -*- coding: utf-8 -*-

"""
Implements the 2D Lattice Environment
"""
# Import gym modules
import sys
from math import floor
from collections import OrderedDict

import gym
from gym import (spaces, utils, logger)
import numpy as np
from six import StringIO

# Human-readable
ACTION_TO_STR = {
    0 : 'UL', 1 : 'UR',
    2 : 'DL', 3 : 'DR'}

STR_TO_ACTION = {
    'UL': 0, 'UR': 1,
    'DL': 2, 'DR': 3 }

POLY_TO_INT = {
    'H' : 1, 'P' : -1
}

# grid buffer
BUFFER = 2

class Pulling2DEnv(gym.Env):
    """A 2-dimensional lattice environment from Dill and Lau, 1989
    [dill1989lattice]_.

    TODO

    .. [dill1989lattice] Lau, K.F., Dill, K.A.: A lattice statistical
    mechanics model of the conformational and se quence spaces of proteins.
    Marcromolecules 22(10), 3986–3997 (1989)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, seq, collision_penalty=-2, trap_penalty=0.5):
        """Initializes the lattice

        Parameters
        ----------
        seq : str, must only consist of 'H' or 'P'
            Sequence containing the polymer chain.
        collision_penalty : int, must be a negative value
            Penalty incurred when the agent made an invalid action.
            Default is -2.
        trap_penalty : float, must be between 0 and 1
            Penalty incurred when the agent is trapped. Actual value is
            computed as :code:`floor(length_of_sequence * trap_penalty)`
            Default is -2.

        Raises
        ------
        AssertionError
            If a certain polymer is not 'H' or 'P'
        """
        try:
            if not set(seq.upper()) <= set('HP'):
                raise ValueError("%r (%s) is an invalid sequence" % (seq, type(seq)))
            self.seq = seq.upper()
        except AttributeError:
            logger.error("%r (%s) must be of type 'str'" % (seq, type(seq)))
            raise

        try:
            if collision_penalty >= 0:
                raise ValueError("%r (%s) must be negative" %
                                 (collision_penalty, type(collision_penalty)))
            if not isinstance(collision_penalty, int):
                raise ValueError("%r (%s) must be of type 'int'" %
                                 (collision_penalty, type(collision_penalty)))
            self.collision_penalty = collision_penalty
        except TypeError:
            logger.error("%r (%s) must be of type 'int'" %
                         (collision_penalty, type(collision_penalty)))
            raise

        try:
            if not 0 < trap_penalty < 1:
                raise ValueError("%r (%s) must be between 0 and 1" %
                                 (trap_penalty, type(trap_penalty)))
            self.trap_penalty = trap_penalty
        except TypeError:
            logger.error("%r (%s) must be of type 'float'" %
                         (trap_penalty, type(trap_penalty)))
            raise

        self.grid_length = len( seq ) + 2 * BUFFER
        self.midpoint = (int((self.grid_length - 1) / 2), int((self.grid_length - 1) / 2))

        #[node, action]
        self.action_space = spaces.MultiDiscrete([ len(seq), 4])
        self.observation_space = spaces.Box(low=-2, high=1,
                                            shape=(self.grid_length, self.grid_length),
                                            dtype=int)
        # Initialize values
        self.reset()


    def reset(self):
        """Resets the environment"""
        self.actions = []
        self.collisions = 0
        self.trapped = 0
        # self.done = len(self.seq) == 1

        self.grid = np.zeros(shape=(self.grid_length, self.grid_length), dtype=int)
        self.mid_row = self.grid_length // 2
        self.chain = []

        # place entire chain on grid 
        for i in range( len( self.seq ) ):
            self.grid[ self.mid_row, BUFFER + i ] = POLY_TO_INT[ self.seq[i] ]
            self.chain.append( (self.mid_row, BUFFER+i) )

        self.last_action = None
        return self.grid

    def render(self, mode='human'):
        """Renders the environment"""

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        # Flip so highest y-value row is printed first
        # desc = np.flipud(self.grid).astype(str)
        desc = self.grid.astype( str )

        # Convert everything to human-readable symbols
        desc[desc == '0'] = '*'
        desc[desc == '1'] = 'H'
        desc[desc == '-1'] = 'P'

        # Obtain all x-y indices of elements
        x_free, y_free = np.where(desc == '*')
        x_h, y_h = np.where(desc == 'H')
        x_p, y_p = np.where(desc == 'P')

        # Decode if possible
        desc.tolist()
        try:
            desc = [[c.decode('utf-8') for c in line] for line in desc]
        except AttributeError:
            pass

        # All unfilled spaces are gray
        for unfilled_coords in zip(x_free, y_free):
            desc[unfilled_coords] = utils.colorize(desc[unfilled_coords], "gray")

        # All hydrophobic molecules are bold-green
        for hmol_coords in zip(x_h, y_h):
            desc[hmol_coords] = utils.colorize(desc[hmol_coords], "green", bold=True)

        # All polar molecules are cyan
        for pmol_coords in zip(x_p, y_p):
            desc[pmol_coords] = utils.colorize(desc[pmol_coords], "cyan")

        # Provide prompt for last action
        if self.last_action is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Up", "Right"][self.last_action]))
        else:
            outfile.write("\n")

        # Draw desc
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile

    def _get_adjacent_coords(self, coords):
        """Obtains all adjacent coordinates of the current position

        Parameters
        ----------
        coords : 2-tuple
            Coordinates (X-y) of the current position

        Returns
        -------
        dictionary
            All adjacent coordinates
        """
        x, y = coords
        adjacent_coords = {
            0 : (x - 1, y),
            1 : (x, y - 1),
            2 : (x, y + 1),
            3 : (x + 1, y),
        }

        return adjacent_coords

    def set_chain( self, chain ):
        '''
        Construct a grid from chain

        :param chain: List of (row, col) coordinates. The i^th coord corresponds
        to i^th node in sequence
        '''
        self.grid = np.zeros( ( self.grid_length, self.grid_length ), dtype=int )
        self.chain = []
        for index in range(len(chain)):
            self.chain.append([chain[index], self.seq[index]])
        for i, (( row, col ), _) in enumerate( self.chain ):
            self.grid[ row, col ] = POLY_TO_INT[ self.seq[i] ]

    def step( self, action ):
        '''
        New step function for pulling environment
        '''

        node, pull_dir = action

        diag_coords = self._get_diag_coords(self.chain[node][0])
        next_move = diag_coords[pull_dir]

        # if the new location already has a node there
        collision = self.get_collision(next_move) # Collision signal

        # if the new location doesn't have any adjacent nodes
        if collision is False and (node != 0 and node != len(self.chain) - 1):
            collision = self.get_invalid_move(next_move, self.chain[node - 1][0], self.chain[node + 1][0])

        #update chain, go to next state
        if collision is False:

            #apply update
            #want to store the previous timesteps nodes as this will be used to update the other nodes
            old_locations_left = [self.chain[node][0]]
            old_locations_right = [self.chain[node][0]]
            self.chain[node][0] = next_move

            #update left
            
            current_node = next_move
            pointer = node - 1
            if pointer >= 0:
                left_node = self.chain[pointer][0]
                closest_node = True
                #update left side of chain
                while(self.node_update(current_node, left_node)):

                    #update the left node with the node two spots closer to the other node
                    #this means we do not have an already looked at spot for the next iteration
                    if closest_node is True:
                        new_pos = self.get_intermediate(current_node, pull_dir)
                        self.chain[pointer][0] = new_pos
                        closest_node = False
                        old_locations_left.append(left_node)
                    else:
                        old_location = old_locations_left.pop(0)
                        self.chain[pointer][0] = old_location
                        old_locations_left.append(left_node)

                    
                    #use the newly updated left node as the next check in the next loop
                    current_node = self.chain[pointer][0]
                    pointer -= 1
                    if pointer < 0:
                        break
                    left_node = self.chain[pointer][0]

            current_node = next_move
            pointer = node + 1
            if pointer < len(self.chain):
                right_node = self.chain[pointer][0]
                closest_node = True
                #update right side of chain
                while(self.node_update(current_node, right_node)):

                    if closest_node is True:
                        new_pos = self.get_intermediate(current_node, pull_dir)
                        self.chain[pointer][0] = new_pos
                        closest_node = False
                        old_locations_right.append(right_node)
                    else:
                        old_location = old_locations_right.pop(0)
                        self.chain[pointer][0] = old_location
                        old_locations_right.append(right_node)
                    
                    #use the newly updated right node as the next check in the next loop
                    current_node = self.chain[pointer][0]
                    pointer += 1
                    if pointer == len(self.chain):
                        break
                    right_node = self.chain[pointer][0]

        grid = self._draw_grid_new(self.chain)
        #TODO: what do we do with self.done?
        self.done = False
        # self.done = True if (len(self.chain) == len(self.seq) or is_trapped) else False
        reward = self._compute_reward(False, collision)
        info = {
            'chain_length' : len(self.chain),
            'seq_length'   : len(self.seq),
            'collisions'   : self.collisions,
            'actions'      : [ACTION_TO_STR[i] for i in self.actions],
            'chain'  : self.chain
        }

        return (grid, reward, self.done, info)   
    
    def verify_chain( self, exp_chain ):
        '''
        Verify current chain matches exp_chain

        :param exp_chain: List of expected coordinates ( row, col )
        :return: True if chain matches False o.w.
        '''

        if len( exp_chain ) != len( self.chain ): return False

        for i in range( len( exp_chain ) ):
            if exp_chain[i] != self.chain[i]:
                return False
        
        chain_map = { (row, col):POLY_TO_INT[self.seq[i]] for i, (row, col) 
                in enumerate( exp_chain ) }
       
        # check to ensure chain is placed on grid correctly
        for i in range( len( self.grid ) ):
            for j in range( len( self.grid ) ):
                if (i, j) in chain_map:
                    if self.grid[i, j] != chain_map[i, j]: return False
                else:
                    if self.grid[i, j] != 0: return False

        return True

    def _get_diag_coords(self, coords):
        """Obtains all adjacent coordinates of the current position

        Parameters
        ----------
        coords : 2-tuple
            Coordinates (X-y) of the current position

        Returns
        -------
        dictionary
            All adjacent coordinates
        """
        y, x = coords
        diag_coords = {
            0 : (y-1, x-1),
            1 : (y-1, x+1),
            2 : (y+1, x-1),
            3 : (y+1, x+1),
        }

        return diag_coords

    def get_collision(self, next_move):
        #trans_x, trans_y = tuple(sum(x) for x in zip(self.midpoint, next_move))
        
        y, x = next_move

        #out of bounds
        if x >= self.grid_length or x < 0 or y < 0 or y >= self.grid_length:
            logger.warn('Your agent was out of bounds! Ending the episode.')
            self.collisions += 1
            return True
        else:
            #pair = ((0,0), 'H')
            for pair in self.chain:
                chain_coord = pair[0]
                if chain_coord == next_move:
                    self.collisions += 1
                    return True

        return False

    def check_collision(self, next_move):
        y, x = next_move

        #out of bounds
        if x >= self.grid_length or x < 0 or y < 0 or y >= self.grid_length:
            #logger.warn('Your agent was out of bounds! Ending the episode.')
            #self.collisions += 1
            return True
        else:
            #pair = ((0,0), 'H')
            for pair in self.chain:
                chain_coord = pair[0]
                if chain_coord == next_move:
                    #self.collisions += 1
                    return True

        return False

    def get_invalid_move(self, current_node, left_node, right_node):

        cn_y, cn_x = current_node
        ln_y, ln_x = left_node
        rn_y, rn_x = right_node

        if (cn_x == ln_x and (cn_y-1 == ln_y or cn_y+1 == ln_y)) or (cn_y == ln_y and (cn_x-1 == ln_x or cn_x+1 == ln_x)):
            return False
        elif (cn_x == rn_x and (cn_y-1 == rn_y or cn_y+1 == rn_y)) or (cn_y == rn_y and (cn_x-1 == rn_x or cn_x+1 == rn_x)):
            return False

        return True

    def node_update(self, current_node, next_node):
        cn_y, cn_x = current_node
        nn_y, nn_x = next_node

        if (cn_x == nn_x and (cn_y-1 == nn_y or cn_y+1 == nn_y)) or (cn_y == nn_y and (cn_x-1 == nn_x or cn_x+1 == nn_x)):
            return False

        return True

    def get_intermediate(self, current_node, action):
        #TODO: need to check for collision?

        y, x = current_node
        new_node = None

        #UL
        if action == 0:
            new_node = (y+1,x)
            if self.check_collision(new_node):
                new_node = (y, x+1)

        #UR
        elif action == 1:
            new_node = (y+1,x)
            if self.check_collision(new_node):
                new_node = (y, x-1)

        #DL
        elif action == 2:
            new_node = (y-1,x)
            if self.check_collision(new_node):
                new_node = (y, x+1)

        #DR
        elif action == 3:
            new_node = (y-1,x)
            if self.check_collision(new_node):
                new_node = (y, x-1)
        
        return new_node

    def _draw_grid_new(self, chain):
        """Constructs a grid with the current chain

        Parameters
        ----------
        chain : OrderedDict
            Current chain/state

        Returns
        -------
        numpy.ndarray
            Grid of shape :code:`(n, n)` with the chain inside
        """
        self.grid = np.zeros( ( self.grid_length, self.grid_length ), dtype=int )
        self.chain = []
        for coord, poly in chain:
            #trans_x, trans_y = tuple(sum(x) for x in zip(self.midpoint, coord))
            y, x = coord
            # Recall that a numpy array works by indexing the rows first
            # before the columns, that's why we interchange.
            self.grid[(y, x)] = POLY_TO_INT[poly]
            self.chain.append( (y, x) )
    
    def _compute_reward(self, is_trapped, collision):
        """Computes the reward for a given time step

        For every timestep, we compute the reward using the following function:

        .. code-block:: python

            reward_t = state_reward 
                       + collision_penalty
                       + actual_trap_penalty

        The :code:`state_reward` is only computed at the end of the episode
        (Gibbs free energy) and its value is :code:`0` for every timestep
        before that.

        The :code:`collision_penalty` is given when the agent makes an invalid
        move, i.e. going to a space that is already occupied.

        The :code:`actual_trap_penalty` is computed whenever the agent
        completely traps itself and has no more moves available. Overall, we
        still compute for the :code:`state_reward` of the current chain but
        subtract that with the following equation:
        :code:`floor(length_of_sequence * trap_penalty)`
        try:

        Parameters
        ----------
        is_trapped : bool
            Signal indicating if the agent is trapped.
        collision : bool
            Collision signal

        Returns
        -------
        int
            Reward function
        """
        state_reward = self._compute_free_energy(self.chain) if self.done else 0
        collision_penalty = self.collision_penalty if collision else 0
        actual_trap_penalty = -floor(len(self.seq) * self.trap_penalty) if is_trapped else 0

        # Compute reward at timestep, the state_reward is originally
        # negative (Gibbs), so we invert its sign.
        reward = - state_reward + collision_penalty + actual_trap_penalty

        return reward

    def _compute_free_energy(self, chain):
        """Computes the Gibbs free energy given the lattice's state

        The free energy is only computed at the end of each episode. This
        follow the same energy function given by Dill et. al.
        [dill1989lattice]_

        Recall that the goal is to find the configuration with the lowest
        energy.

        .. [dill1989lattice] Lau, K.F., Dill, K.A.: A lattice statistical
        mechanics model of the conformational and se quence spaces of proteins.
        Marcromolecules 22(10), 3986–3997 (1989)

        Parameters
        ----------
        chain : OrderedDict
            Current chain in the lattice

        Returns
        -------
        int
            Computed free energy
        """
        h_polymers = [x for x in chain if chain[x] == 'H']
        h_pairs = [(x, y) for x in h_polymers for y in h_polymers]

        # Compute distance between all hydrophobic pairs
        h_adjacent = []
        for pair in h_pairs:
            dist = np.linalg.norm(np.subtract(pair[0], pair[1]))
            if dist == 1.0: # adjacent pairs have a unit distance
                h_adjacent.append(pair)

        # Get the number of consecutive H-pairs in the string,
        # these are not included in computing the energy
        h_consecutive = 0
        for i in range(1, len(self.chain)):
            if (self.seq[i] == 'H') and (self.seq[i] == self.seq[i-1]):
                h_consecutive += 1

        # Remove duplicate pairs of pairs and subtract the
        # consecutive pairs
        nb_h_adjacent = len(h_adjacent) / 2
        gibbs_energy = nb_h_adjacent - h_consecutive
        reward = - gibbs_energy
        return int(reward)

