import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import sys

import dqn
import dqn2
from dqn_utils import *
from atari_wrappers import *
from PNN import PNN
import pickle

train_time_scale = 3

tf.flags.DEFINE_string("model_dir", "/tmp/dqn", 
                       "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_string("env", "PongNoFrameskip-v3", 
                       "Name of gym Atari environment, e.g. Breakout-v0")
tf.flags.DEFINE_string("checkpoint_dir", None, 
                       "if not None, restore from checkpoint_dir")
tf.flags.DEFINE_integer("branch_layer", 1, 
                       "if not None, restore from checkpoint_dir")
tf.flags.DEFINE_string("change", None, 
                       "if not None, restore from checkpoint_dir")
tf.flags.DEFINE_string("save_dir", "/tmp", 
                       "if not None, restore from checkpoint_dir")
flags.DEFINE_boolean('from_scratch', False, 'If true, train from scratch.')
flags.DEFINE_boolean('replay', False, 'If true, replay.')

FLAGS = tf.flags.FLAGS

POLICY_DIR = "saved_policy.pkl"

BRANCH_LAYER = FLAGS.branch_layer
SAVE_DIR = FLAGS.save_dir
CHANGE = eval(FLAGS.change) if FLAGS.change is not None else None
ENV = FLAGS.env

def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def atari_learn(env, session, num_timesteps, filename, save_dir):
    # This is just a rough estimate
    num_iterations = float(num_timesteps)/4.0

    lr_multiplier = 9
    lr_schedule = PiecewiseSchedule(
        [
            (0,                     1e-4 * lr_multiplier),
            (num_iterations / (10), 1e-4 * lr_multiplier),
            (num_iterations / (2),  5e-5 * lr_multiplier),
        ],
        outside_value=5e-5 * lr_multiplier
    )
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        curr_env = get_wrapper_by_name(env, "Monitor")
        return curr_env.get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], 
        outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        filename = filename,
        save_dir = save_dir
    )
    env.close()

def atari_learn_v2(env, session, num_timesteps, filename, save_dir):
    # This is just a rough estimate
    num_iterations = float(num_timesteps)/4.0

    lr_multiplier = 1

    lr_schedule = PiecewiseSchedule(
        [
            (0,                     1e-4 * lr_multiplier),
            (num_iterations / (10), 1e-4 * lr_multiplier),
            (num_iterations / (2),  5e-5 * lr_multiplier),
        ],
        outside_value=5e-5 * lr_multiplier
    )

    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        curr_env = get_wrapper_by_name(env, "Monitor")
        return curr_env.get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (2e6, 0.01),
        ], 
        outside_value=0.01
    )

    dqn2.learn(
        env,
        q_func=pnn_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        filename = filename,
        save_dir = save_dir
    )
    env.close()



def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc 
            for x in local_device_protos 
            if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed, change = None):
    env_id = task.env_id
    env = gym.make(env_id)
    set_global_seeds(seed)
    env.seed(seed)
    expt_dir = FLAGS.model_dir
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)
    if change:
        return change(env)
    else:
        return env

def train():
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task_dict = {"BeamRiderNoFrameskip-v3":0, 
                "BreakoutNoFrameskip-v3":1,
                "EnduroNoFrameskip-v3":2,
                "PongNoFrameskip-v3":3,
                "QbertNoFrameskip-v3":4,
                "SeaquestNoFrameskip-v3":5,
                "SpaceInvadersNoFrameskip-v3":6}
    task = benchmark.tasks[task_dict[FLAGS.env]]
    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    session = get_session()
    atari_learn(env, session, train_time_scale*task.max_timesteps, FLAGS.checkpoint_dir, SAVE_DIR)

def pnn_model(img_in, num_actions, session, scope, reuse=False, branch_layer=BRANCH_LAYER):
    pnn = create_pnn_pong(session, scope, img_in, branch_layer, reuse)
    pnn.add_col(num_actions)
    output_action = pnn.get_q_val_of_col(1)
    return output_action

def create_pnn_pong(session, nn_name, x, branch_layer=1, reuse = False):
    structure = [(84, 84, 4), (8, 8, 4, 32), (4, 4, 2, 64), (3, 3, 1, 64), (512,)]
    pnn = PNN(
        nn_name, 
        structure, 
        x, 
        branch_layer = branch_layer, 
        reuse=reuse
    )
    pnn.add_col(6)
    with open(POLICY_DIR, 'r+') as f:
        v_w = pickle.load(f)
    pnn.assign_values_to_col_temp(v_w, session)
    return pnn

def train_variation():
    benchmark = gym.benchmark_spec('Atari40M')
    task_dict = {"BeamRiderNoFrameskip-v3":0, 
                "BreakoutNoFrameskip-v3":1,
                "EnduroNoFrameskip-v3":2,
                "PongNoFrameskip-v3":3,
                "QbertNoFrameskip-v3":4,
                "SeaquestNoFrameskip-v3":5,
                "SpaceInvadersNoFrameskip-v3":6}
    task = benchmark.tasks[task_dict[ENV]]
    env = get_env(task, 0, change = CHANGE)
    session = get_session()
    atari_learn_v2(env, session, task.max_timesteps, FLAGS.checkpoint_dir, SAVE_DIR)





def replay(env, session, policy_dir):
    structure = [(84, 84, 4), (8, 8, 4, 32), (4, 4, 2, 64), (3, 3, 1, 64), (512,)]
    obs_for_task = tf.placeholder(tf.uint8, [None, 84, 84, 4])
    obs_for_task_float = tf.cast(obs_for_task, tf.float32) / 255.0
    pnn = PNN("q_func", structure, obs_for_task_float)
    pnn.add_col(6)
    pnn.add_col(6)
    loader = tf.train.Saver()
    loader.restore(session, policy_dir)
    action_for_task = tf.argmax(pnn.get_q_val_of_col(0), 1)
    replay_buffer = ReplayBuffer(10, 4)
    obs = env.reset()
    done = False
    total_rewards = 0
    while not done:
        next_index = replay_buffer.store_frame(obs)
        action = session.run(
            action_for_task,
            feed_dict={
                obs_for_task:np.array([replay_buffer.encode_recent_observation()])
            }
        )
        last_reward = 0
        for i in range(4):
            obs, reward_inc, done, _ = env.step(action)
            last_reward += reward_inc
            if done:
                break
        replay_buffer.store_effect(next_index, action, last_reward, done)
        total_rewards += last_reward
    print total_rewards


def play():
    benchmark = gym.benchmark_spec('Atari40M')
    task = benchmark.tasks[3]
    env = get_env(task, 0, change = None)
    session = get_session()
    replay(env, session, "saved/TNN_layer_1/nn_20170430-172122-1450004")


if __name__ == "__main__":
    if FLAGS.replay:
        play()
    elif FLAGS.from_scratch:
        train()
    else:
        train_variation()






