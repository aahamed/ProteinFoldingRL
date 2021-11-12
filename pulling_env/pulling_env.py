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
    2 : 'BL', 3 : 'BR',
    4: 'STOP' }

STR_TO_ACTION = {
    'UL': 0, 'UR': 1,
    'BL': 2, 'BR': 3 ,
    'STOP': 4 }

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

    def __init__(self, seq, collision_penalty=-2, trap_penalty=0.5, early_stopping=-1):
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
            # if not isinstance(collision_penalty, int):
            #     raise ValueError("%r (%s) must be of type 'int'" %
            #                      (collision_penalty, type(collision_penalty)))
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
        self.early_stopping = early_stopping

        #[node, action]
        self.action_space = spaces.MultiDiscrete([ len(seq), 5])
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
        self.done = False
        # self.done = len(self.seq) == 1

        self.grid = np.zeros(shape=(self.grid_length, self.grid_length), dtype=int)
        self.mid_row = self.grid_length // 2
        self.chain = []

        # place entire chain on grid 
        for i in range( len( self.seq ) ):
            self.grid[ self.mid_row, BUFFER + i ] = POLY_TO_INT[ self.seq[i] ]
            self.chain.append( [(self.mid_row, BUFFER+i), self.seq[i]] )

        self.last_action = None
        self.timestep = 0
        self.max_timesteps = 2 * len( self.seq )
        self.old_energy = 0
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

    def set_chain( self, chain ):
        '''
        Construct a grid from chain

        :param chain: List of (row, col) coordinates. The i^th coord corresponds
        to i^th node in sequence
        '''
        assert len( chain ) == len( self.seq )
        self.grid = np.zeros( ( self.grid_length, self.grid_length ), dtype=int )
        self.chain = []
        for index in range(len(chain)):
            self.chain.append([chain[index], self.seq[index]])
        for i, (( row, col ), _) in enumerate( self.chain ):
            self.grid[ row, col ] = POLY_TO_INT[ self.seq[i] ]

    def is_end_node( self, node ):
        if node == 0 or node == (len( self.seq ) -1):
            return True
        return False

    def pull_chain( self, node, old_locations, pull_dir, flag='forward' ):
        '''
        pull chain
        :param node: node to pull
        :param old_locations: stack containing old location of node
        :param pull_dir: direction to pull
        :param flag: either forward or backward. If forward
        pull nodes > node o.w. pull nodes < node
        '''
        assert flag in ['forward', 'backward']

        def get_next_node( node ):
            next_node = node+1 if flag=='forward' else node-1
            if next_node >= 0 and next_node < len( self.seq ):
                return next_node
            return None
 
        curr_node_coord = self.chain[node][0]
        next_node = get_next_node( node )

        # if this is end node, nothing to do
        if next_node is None: return
   
        # pull nodes in direction specified by flag until valid configuration is reached
        next_node_coord = self.chain[next_node][0]
        closest_node = True
        while not self.is_adjacent( curr_node_coord, next_node_coord ):
            if closest_node:
                # if closest node to pulled node; move to free corner
                new_pos = self.get_intermediate( curr_node_coord, pull_dir )
                self.chain[next_node][0] = new_pos
                closest_node = False
                old_locations.append( next_node_coord )
            else:
                # for other nodes just move the node up by 2 positions
                new_pos = old_locations.pop(0)
                self.chain[next_node][0] = new_pos
                old_locations.append( next_node_coord )

            curr_node_coord = self.chain[next_node][0]
            next_node = get_next_node( next_node )
            if next_node is None: break
            next_node_coord = self.chain[next_node][0]


    def step( self, action ):
        '''
        New step function for pulling environment
        '''
        
        self.timestep += 1
        node, pull_dir = action

        # action is stop; end episode
        if ACTION_TO_STR[ pull_dir ] == 'STOP':
            self.done = True
            reward = self._compute_reward(False, False, True)
            info = {
                'chain_length' : len(self.chain),
                'seq_length'   : len(self.seq),
                'collisions'   : self.collisions,
                'actions'      : [ACTION_TO_STR[i] for i in self.actions],
                'chain'  : self.chain
            }
            return (self.grid, reward, self.done, info)   

        diag_coords = self._get_diag_coords(self.chain[node][0])
        next_move = diag_coords[pull_dir]

        # if the new location already has a node there
        collision = self.get_collision(next_move, count=True) # Collision signal

        # if the new location doesn't have any adjacent nodes
        if collision is False and not self.is_end_node(node):
            collision = self.is_invalid_move(next_move, node)
        
        # perform pull operation
        if not collision:
            old_locations_left = [self.chain[node][0]]
            old_locations_right = [self.chain[node][0]]
            self.chain[node][0] = next_move
            self.pull_chain( node, old_locations_left, pull_dir, flag='backward' )
            self.pull_chain( node, old_locations_right, pull_dir, flag='forward' )

        grid = self._draw_grid_new(self.chain)
        #TODO: what do we do with self.done?
        self.done = self.timestep == self.max_timesteps
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
            if exp_chain[i] != self.chain[i][0]:
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
            Coordinates (y,x) of the current position

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

    def get_collision(self, next_move, count=False):
        y, x = next_move

        #out of bounds
        if x >= self.grid_length or x < 0 or y < 0 or y >= self.grid_length:
            logger.warn('Your agent was out of bounds! Ending the episode.')
            if count: self.collisions += 1
            return True
        else:
            #pair = ((0,0), 'H')
            for pair in self.chain:
                chain_coord = pair[0]
                if chain_coord == next_move:
                    if count: self.collisions += 1
                    return True

        return False

    
    def get_adjacent_coords(self, coords):
        """
        Obtains all adjacent coordinates of the current position
        """
        y, x = coords
        adjacent_coords = [
            (y, x-1),
            (y-1, x),
            (y, x+1),
            (y+1, x),
        ]

        return adjacent_coords

    def is_adjacent( self, c1, c2 ):
        '''
        check if c1 and c2 are adjacent
        :param c1: coordinate 1 in ( y, x )
        :param c2: coordinate 2 in ( y, x )
        returns True if c1 and c2 are adjacent; false o.w.
        '''
        return c1 in self.get_adjacent_coords( c2 )
        

    def is_invalid_move( self, next_move, current_node ):
        prev_node_coord = self.chain[current_node-1][0]
        next_node_coord = self.chain[current_node+1][0]
        
        # For move to be valid; next_move must be adjacent to a neighboring node
        if self.is_adjacent( next_move, prev_node_coord ) or \
                self.is_adjacent( next_move, next_node_coord ):
            return False

        return True

    def get_intermediate(self, current_node, action):
        #TODO: need to check for collision?

        y, x = current_node
        new_node = None

        #UL
        if action == 0:
            new_node = (y+1,x)
            if self.get_collision(new_node):
                new_node = (y, x+1)

        #UR
        elif action == 1:
            new_node = (y+1,x)
            if self.get_collision(new_node):
                new_node = (y, x-1)

        #DL
        elif action == 2:
            new_node = (y-1,x)
            if self.get_collision(new_node):
                new_node = (y, x+1)

        #DR
        elif action == 3:
            new_node = (y-1,x)
            if self.get_collision(new_node):
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
        self.chain = chain
        for coord, poly in chain:
            y, x = coord
            self.grid[(y, x)] = POLY_TO_INT[poly]

        return self.grid
    
    def _compute_reward(self, is_trapped, collision, is_early_stopping=False):
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
        # new_energy = self._compute_free_energy(self.chain)
        # state_reward = max(new_energy - self.old_energy,0)
        collision_penalty = self.collision_penalty if collision else 0
        actual_trap_penalty = -floor(len(self.seq) * self.trap_penalty) if is_trapped else 0
        early_stop_penalty = self.early_stopping if is_early_stopping else 0

        # Compute reward at timestep, the state_reward is originally
        # negative (Gibbs), so we invert its sign.
        reward = - state_reward + collision_penalty + actual_trap_penalty + early_stop_penalty
        #self.old_energy = new_energy

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
        h_polymers = [c for i, (c, p) in enumerate( chain ) if chain[i][1] == 'H']
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

