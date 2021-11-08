from pulling_env import Pulling2DEnv, POLY_TO_INT, BUFFER, \
        STR_TO_ACTION
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
                assert env.chain[j-BUFFER] == (i, j)
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

def test_set_chain():
    print( 'Test set chain' )
    seq = 'HPH'
    env = Pulling2DEnv( seq )
    env.reset()
    chain = [ (2,2), (2,3), (2,4) ]
    env.set_chain( chain, seq)
    env.render()
    chain = [ (1,3), (2,3), (2, 4) ]
    env.set_chain( chain, seq)
    env.render()
    seq = 'HPHPP'
    env = Pulling2DEnv( seq )
    chain = [ (1, 1), (2, 1), (2, 2), (2,3), (2, 4) ]
    env.set_chain( chain, seq)
    env.render()
    print( 'Test passed!' )

def test_pull_ur():
    print( 'Test pull upper right node 1' )
    print( 'Case1:' )
    seq = 'HPH'
    env = Pulling2DEnv( seq )
    env.reset()
    chain = [ (2,2), (2,3), (2,4) ]
    env.set_chain( chain, seq)
    env.render()
    # pull node 1 up and then right (UR = 1)
    action = ( 1, 1 )
    env.step( action )
    env.render()
    exp_chain = [ (1, 3), (1, 4), (2, 4) ]
    assert env.verify_chain( exp_chain )
    print( 'Case1 Passed!' )
    print( '\nCase2:' )
    env.reset()
    chain = [ (1,3), (2,3), (2,4) ]
    env.set_chain( chain, seq)
    env.render()
    # pull node 1 up and then right (UR = 1)
    action = ( 1, 1 )
    env.step( action )
    env.render()
    exp_chain = [ (1, 3), (1, 4), (2, 4) ]
    assert env.verify_chain( exp_chain )
    print( 'Test passed!' )

def test_pull_long():
    print( 'Test pull upper right node 3' )
    seq = 'HPHHH'
    env = Pulling2DEnv( seq )
    env.reset()
    chain = [ (2,1), (2,2), (2,3), (2,4), (2,5) ]
    env.set_chain( chain, seq)
    env.render()
    # pull node 3 up and then right (UR = 1)
    action = ( 3, 1 )
    # TODO
    env.step( action )
    env.render()
    exp_chain = [ (2,3),(2, 4), (1, 4), (1, 5) , (2,5)]
    assert env.verify_chain( exp_chain )
    print( 'Test passed!' )

def test_collision():
    pass

def invalid_pull():
    print( 'Test pull bottom left node 2' )
    seq = 'HPHHH'
    env = Pulling2DEnv( seq )
    env.reset()
    chain = [ (1,1), (2,1), (3,1), (3,2), (3,3) ]
    env.set_chain( chain, seq)
    env.render()
    # pull node 3 up and then bottom left (BL = 2)
    action = ( 2, 2 )
    # TODO
    _, reward, _, _ = env.step( action )
    env.render()
    exp_chain = [ (1,1), (2,1), (3,1), (3,2), (3,3)]
    assert reward == -2
    assert env.verify_chain( exp_chain )
    print( 'Test passed!' )

def special_pull():

    print( 'Test pull bottom left node 0' )
    seq = 'HPHHHH'
    env = Pulling2DEnv( seq )
    env.reset()
    chain = [ (1,1), (1,2), (1,3), (0,3), (0,4), (1,4) ]
    env.set_chain(chain, seq)
    env.render()

    #pull node 0 bottom left (BL = 2)
    action = (0, 2)

    env.step( action )
    env.render()
    exp_chain = [ (2,0), (1,0), (1,1), (1,2), (1,3), (1,4) ]
    assert env.verify_chain( exp_chain )
    print( 'Test passed!' )



def main():
    test_reset()
    test_render()
    test_set_chain()
    test_pull_ur()
    test_pull_long()
    invalid_pull()
    special_pull()

if __name__ == '__main__':
    main()
