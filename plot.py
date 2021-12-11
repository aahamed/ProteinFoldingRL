import csv
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set_style("darkgrid")
plt.style.use('seaborn')

def read_csv( filename ):
    data = []
    with open( filename, newline='' ) as csvfile:
        datareader = csv.reader( csvfile, delimiter=',' )
        for i, row in enumerate(datareader):
            if i == 0: continue
            _, step, rew = row
            data.append( [ int(step), float(rew) ] )
            # print(row)
    return np.array( data ).T

def aggregate_seed_data( filenames ):
    '''
    aggregate data over multiple seeds
    '''
    # import pdb; pdb.set_trace()
    agg_x = []
    agg_y = []
    for filename in filenames:
        data = read_csv( filename )
        agg_x.append( data[0] )
        agg_y.append( data[1] )
    agg_y = np.stack( agg_y )
    mean_y = agg_y.mean( axis=0 )
    std_y = agg_y.std( axis=0 )
    x = agg_x[0]
    return x, mean_y, std_y

def smooth_rewards( rewards, lookback=10 ):
    N = len( rewards )
    smooth = np.zeros( N, dtype=np.float32 )
    for i in range( N ):
        start = max( 0, i - lookback )
        smooth[i] = rewards[ start: i+1 ].mean()
    return smooth

def extend( arr, N ):
    '''
    extend arr to len N
    '''
    if len( arr ) >= N: return arr
    new_arr = np.zeros( N )
    new_arr[:len(arr)] = arr
    new_arr[len(arr):] = arr[-1]
    return new_arr

def merge( x1, x2 ):
    if len( x1 ) >= len( x2 ): return x1
    new_x1 = np.zeros( len(x2) )
    new_x1[ : len(x1) ] = x1
    new_x1[ len(x1): ] = max( x1.max(), x2.max() )
    return new_x1

def plot( ax, x, y, std_y, label ):
    y = smooth_rewards( y )
    std_y = smooth_rewards( std_y )
    # sns.lineplot( x, y )
    ax.plot( x, y, label=label )
    ax.fill_between( x, y+std_y, y-std_y, alpha=0.4 )

def main():
    fig, ax = plt.subplots( 1, 1 )
    ax.set( xlabel='Timestep', ylabel='Reward',
            title='Reward vs. Timestep for DQN, A2C and PPO' )

    # DQN
    dqn_filenames = [
        'run-HHPPHH_DQN_1-tag-rollout_ep_rew_mean.csv',
        # 'run-HHPPHHHH_DQN_2-tag-rollout_ep_rew_mean.csv',
        # 'run-HHPPHHPPHH_DQN_LEN10_SEED1_1-tag-rollout_ep_rew_mean.csv',
    ]
    dqn_x, dqn_y, dqn_std_y = aggregate_seed_data( dqn_filenames ) 
    
    # A2C
    a2c_filenames = [
        'run-HHPPHH_A2C_HHPPHH-tag-rollout_ep_rew_mean.csv',
        # 'run-HHPPHHHH_A2C_HHPPHHHH-tag-rollout_ep_rew_mean.csv',
        # 'run-HHPPHHPPHH_A2C_HHPPHHPPHH-tag-rollout_ep_rew_mean.csv'
    ]
    a2c_x, a2c_y, a2c_std_y = aggregate_seed_data( a2c_filenames ) 

    # PPO
    ppo_filenames = [
        'run-HHPPHH_PPO_HHPPHH-tag-rollout_ep_rew_mean.csv',
        # 'run-HHPPHHHH_PPO_HHPPHHHH-tag-rollout_ep_rew_mean.csv',
        # 'run-HHPPHHPPHH_PPO_6-tag-rollout_ep_rew_mean.csv',
        # 'run-phpphphhhphhphhhhh_PPO_phpphphhhphhphhhhh-tag-rollout_ep_rew_mean.csv',
    ]
    ppo_x, ppo_y, ppo_std_y = aggregate_seed_data( ppo_filenames )

    # process
    max_N = max( [ len(dqn_y), len(a2c_y), len(ppo_y) ] )
    dqn_y, a2c_y, ppo_y = extend( dqn_y, max_N ), extend( a2c_y, max_N ), \
            extend( ppo_y, max_N )
    dqn_std_y, a2c_std_y, ppo_std_y = extend( dqn_std_y, max_N ), \
            extend( a2c_std_y, max_N ), extend( ppo_std_y, max_N )
    x = sorted( [ dqn_x, a2c_x, ppo_x ], key=lambda x: len(x) )[-1]
    dqn_x, a2c_x, ppo_x = merge( dqn_x, x ), merge( a2c_x, x ), \
            merge( ppo_x, x )

    # plot
    plot( ax, dqn_x, dqn_y, dqn_std_y, 'DQN' )
    plot( ax, a2c_x, a2c_y, a2c_std_y, 'A2C' )
    plot( ax, ppo_x, ppo_y, ppo_std_y, 'PPO' )
    ax.legend()
    plt.show()

    fig.savefig( 'results-profold.png', bbox_inches='tight' )

if __name__ == '__main__':
    main()
