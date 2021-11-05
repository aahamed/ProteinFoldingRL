from pulling_env import Pulling2DEnv, POLY_TO_INT, BUFFER
from gym import spaces
import numpy as np


def test_reset():
    print( 'Test reset' )
    seq = 'HHPHHPPHPHPHHHPHPHHH'
    env = Pulling2DEnv( seq )
    state = env.reset()
    for i in range( len( state ) ):
        for j in range( len( state[0] ) ):
            if i == env.mid_row and j >= BUFFER and j < BUFFER + len( seq ):
                assert state[ i, j ] == POLY_TO_INT[ seq[j-BUFFER] ]
            else:
                assert state[ i, j ] == 0
    print( 'Test passed!' )

def test_render():
    print( 'Test render' )
    seq = 'HHPHHP'
    env = Pulling2DEnv( seq )
    state = env.reset()
    env.render()
    print( 'Test passed!' )

def main():
    test_reset()
    test_render()

if __name__ == '__main__':
    main()
