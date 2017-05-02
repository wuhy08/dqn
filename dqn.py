import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
from time import gmtime, strftime
import cv2

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

def compress(img):
    img_h, img_w, img_c = img.shape
    img_h = img_h/2
    img_w = img_w/2
    new_img = cv2.resize(img, (img_h, img_w))
    if len(new_img.shape) == 2:
        return np.expand_dims(new_img, axis=2)
    else:
        return new_img.astype(uint8)

def learn(env,
          q_func,
          optimizer_spec,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10, 
          filename = None,
          save_dir = '/tmp'):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    downsample = False
    skip = True

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
        using_ram = True
    else:
        img_h, img_w, img_c = env.observation_space.shape
        if downsample:
            input_shape = (img_h/2, img_w/2, frame_history_len * img_c)
        else:
            input_shape = (img_h, img_w, frame_history_len * img_c)
        using_ram = False
    num_actions = env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    rew_t_ph = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    # Here, you should fill in your own code to compute the Bellman error. This requires
    # evaluating the current and next Q-values and constructing the corresponding error.
    # TensorFlow will differentiate this error for you, you just need to pass it to the
    # optimizer. See assignment text for details.
    # Your code should produce one scalar-valued tensor: total_error
    # This will be passed to the optimizer in the provided code below.
    # Your code should also produce two collections of variables:
    # q_func_vars
    # target_q_func_vars
    # These should hold all of the variables of the Q-function network and target network,
    # respectively. A convenient way to get these is to make use of TF's "scope" feature.
    # For example, you can create your Q-function network with the scope "q_func" like this:
    # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    # And then you can obtain the variables like this:
    # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
    ######
    
    # YOUR CODE HERE

    q_val_obs_t = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    q_val_obs_tp1 = q_func(obs_tp1_float, num_actions, scope="target_q_func", reuse=False)
    max_q_val_obs_tp1 = tf.reduce_max(q_val_obs_tp1, 1)
    q_val_estimate = rew_t_ph + gamma*(1-done_mask_ph)*max_q_val_obs_tp1
    total_error = tf.reduce_mean(
        tf.square(
            q_val_estimate - tf.reduce_sum(
                q_val_obs_t * tf.one_hot(
                    act_t_ph, num_actions, dtype = tf.float32
                ), 1
            )
        )
    )

    optimal_action = tf.argmax(q_val_obs_t, 1)
    q_func_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, 
        scope='q_func'
    )
    target_q_func_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, 
        scope='target_q_func'
    )
    

    ######

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    nn_saver = tf.train.Saver(q_func_vars, max_to_keep=1)
    nn_saver_freq = 50000
    time_str = strftime("%Y%m%d-%H%M%S", gmtime())

    # evaluation_steps = 50000
    # sample_size = 4000
    # eval_buffer = ReplayBuffer(evaluation_steps, frame_history_len)
    # last_obs_eval = env.reset()

    # for t in range(evaluation_steps):
    #     if not using_ram and downsample:
    #         last_obs_eval = compress(last_obs_eval)
    #     next_index_eval = eval_buffer.store_frame(last_obs_eval)
    #     action_eval = env.action_space.sample()
    #     next_obs_eval, last_reward_eval, done_eval, _ = env.step(action_eval)
    #     eval_buffer.store_effect(next_index_eval, action_eval, last_reward_eval, done_eval)
    #     if done_eval:
    #         last_obs_eval = env.reset()
    #     else:
    #         last_obs_eval = next_obs_eval
    # obs_t_batch_eval, _, _, _, _ = eval_buffer.sample(sample_size)






    # construct the replay buffer

    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
    

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000
    logging_iter = 0
    action_repeat = 4

    if filename is not None:
        nn_saver.restore(session, filename)
        print "Successfully load NN model file "+filename
        model_initialized = True
        # filename_pkl = filename[:filename.rfind('-')] 
        # with open(filename_pkl + '.pkl', 'rb') as input_file:  
        #     try:
        #         replay_buffer = pickle.load(input_file)
        #         print "Successfully load pickle file "+filename_pkl
        #     except EOFError:  
        #         replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
        #         print "Failed to load pickle file "+filename_pkl+", use new replay buffer"

        
        



    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        # At this point, "last_obs" contains the latest observation that was
        # recorded from the simulator. Here, your code needs to store this
        # observation and its outcome (reward, next observation, etc.) into
        # the replay buffer while stepping the simulator forward one step.
        # At the end of this block of code, the simulator should have been
        # advanced one step, and the replay buffer should contain one more
        # transition.
        # Specifically, last_obs must point to the new latest observation.
        # Useful functions you'll need to call:
        # obs, reward, done, info = env.step(action)
        # this steps the environment forward one step
        # obs = env.reset()
        # this resets the environment if you reached an episode boundary.
        # Don't forget to call env.reset() to get a new observation if done
        # is true!!
        # Note that you cannot use "last_obs" directly as input
        # into your network, since it needs to be processed to include context
        # from previous frames. You should check out the replay buffer
        # implementation in dqn_utils.py to see what functionality the replay
        # buffer exposes. The replay buffer has a function called
        # encode_recent_observation that will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        # Don't forget to include epsilon greedy exploration!
        # And remember that the first time you enter this loop, the model
        # may not yet have been initialized (but of course, the first step
        # might as well be random, since you haven't trained your net...)

        #####
        
        # YOUR CODE HERE
        epsilon = exploration.value(t)
        if not using_ram and downsample:
            last_obs = compress(last_obs)
        next_index = replay_buffer.store_frame(last_obs)
        if (random.random < epsilon or not model_initialized):
            action = env.action_space.sample()
        else:
            action = session.run(optimal_action,
                feed_dict={
                    obs_t_ph:np.array([replay_buffer.encode_recent_observation()])
                }
            )
        if not skip:
            next_obs, last_reward, done, _ = env.step(action)
        else:
            last_reward = 0
            for i in range(frame_history_len):
                next_obs, reward_inc, done, _ = env.step(action)
                last_reward += reward_inc
                if done:
                    break
        replay_buffer.store_effect(next_index, action, last_reward, done)
        if done:
            last_obs = env.reset()
        else:
            last_obs = next_obs


        #####

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Here, you should perform training. Training consists of four steps:
            # 3.a: use the replay buffer to sample a batch of transitions (see the
            # replay buffer code for function definition, each batch that you sample
            # should consist of current observations, current actions, rewards,
            # next observations, and done indicator).
            # 3.b: initialize the model if it has not been initialized yet; to do
            # that, call
            #    initialize_interdependent_variables(session, tf.global_variables(), {
            #        obs_t_ph: obs_t_batch,
            #        obs_tp1_ph: obs_tp1_batch,
            #    })
            # where obs_t_batch and obs_tp1_batch are the batches of observations at
            # the current and next time step. The boolean variable model_initialized
            # indicates whether or not the model has been initialized.
            # Remember that you have to update the target network too (see 3.d)!
            # 3.c: train the model. To do this, you'll need to use the train_fn and
            # total_error ops that were created earlier: total_error is what you
            # created to compute the total Bellman error in a batch, and train_fn
            # will actually perform a gradient step and update the network parameters
            # to reduce total_error. When calling session.run on these you'll need to
            # populate the following placeholders:
            # obs_t_ph
            # act_t_ph
            # rew_t_ph
            # obs_tp1_ph
            # done_mask_ph
            # (this is needed for computing total_error)
            # learning_rate -- you can get this from optimizer_spec.lr_schedule.value(t)
            # (this is needed by the optimizer to choose the learning rate)
            # 3.d: periodically update the target network by calling
            # session.run(update_target_fn)
            # you should update every target_update_freq steps, and you may find the
            # variable num_param_updates useful for this (it was initialized to 0)
            #####
            
            # YOUR CODE HERE
            obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = replay_buffer.sample(batch_size)
            if not model_initialized:
                if tf.__version__ == '0.12.0' or tf.__version__ == '0.12.1':
                    global_variables = tf.global_variables()
                else:
                    global_variables = tf.global_variables()
                initialize_interdependent_variables(session, global_variables, {
                    obs_t_ph: obs_t_batch,
                    obs_tp1_ph: obs_tp1_batch
                    }
                )

                model_initialized = True
            session.run(train_fn, feed_dict={
                obs_t_ph:obs_t_batch,
                act_t_ph:act_batch,
                rew_t_ph:rew_batch,
                obs_tp1_ph:obs_tp1_batch,
                done_mask_ph:done_mask,
                learning_rate:optimizer_spec.lr_schedule.value(t)
                })
            if num_param_updates%nn_saver_freq == 0:
                filename_nn_saver = save_dir + "/nn_" + time_str
                nn_saver.save(session, filename_nn_saver, global_step=t)
                print "Successfully save file "+filename_nn_saver
            num_param_updates += 1
            if num_param_updates%target_update_freq == 0:
                session.run(update_target_fn)
            #####

        ### 4. Log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            sys.stdout.flush()
            if logging_iter == 0: 
                filename_log = "train_log_" + time_str + ".txt"
                filename_hyper = "hyper_" + time_str + ".txt"
                file_h = open(filename_hyper, 'w+')
                file_h.write("replay_buffer_size: %d\n"\
                             "batch_size: %d\n"\
                             "gamma: %f\n"\
                             "learning_starts: %d\n"\
                             "learning_freq: %d\n"\
                             "frame_history_len: %d\n"\
                             "target_update_freq: %d\n"\
                             "grad_norm_clipping: %d" % (
                                replay_buffer_size,
                                batch_size,
                                gamma,
                                learning_starts,
                                learning_freq,
                                frame_history_len,
                                target_update_freq,
                                grad_norm_clipping
                                )
                             )
                file_h.close()
                file_l = open(filename_log, 'w+')
                file_l.write("time, mean reward 100 episode, best mean reward, episodes, exploration, learning rate\n")
                logging_iter += 1
                file_l.close()
            file_l = open(filename_log, 'a')
            file_l.write("%d, %f, %f, %d, %f, %f\n" % (
                t, 
                mean_episode_reward, 
                best_mean_episode_reward, 
                len(episode_rewards), 
                exploration.value(t), 
                optimizer_spec.lr_schedule.value(t)
                )
            )
            file_l.close()



