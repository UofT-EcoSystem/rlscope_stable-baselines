"""
Load hyperparameters used in rl-baselines-zoo.

Refactored from rl-baselines-zoo/train.py
"""
import os
import yaml
import copy
import importlib
import time
import pprint
from collections import OrderedDict

import numpy as np
import gym

from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.ppo2.ppo2 import constfn

from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, DDPG, TRPO, SAC

ALGOS = {
  'a2c': A2C,
  'acer': ACER,
  'acktr': ACKTR,
  'dqn': DQN,
  'ddpg': DDPG,
  'her': HER,
  'sac': SAC,
  'ppo2': PPO2,
  'trpo': TRPO
}

def linear_schedule(initial_value):
  """
  Linear learning rate schedule.

  :param initial_value: (float or str)
  :return: (function)
  """
  if isinstance(initial_value, str):
    initial_value = float(initial_value)

  def func(progress):
    """
    Progress will decrease from 1 (beginning) to 0
    :param progress: (float)
    :return: (float)
    """
    return progress * initial_value

  return func

def load_rl_baselines_zoo_hyperparams(rl_baselines_zoo_dir, algo, env_id,
                                      # Build tensorflow neural-network?
                                      build_model_layers=False,
                                      verbose=False,
                                      log_interval=-1,
                                      # tensorboard_log=None,
                                      seed=0,
                                      # n_jobs=1,
                                      # sampler='random',
                                      # pruner='median',
                                      # n_trials=10,
                                      # n_timesteps=-1,
                                      # optimize_hyperparameters=False,
                                      # log_folder='logs',
                                      # trained_agent='',
                                      ):
  """
  Refactored from rl-baselines-zoo/train.py
  """
  yml_path = os.path.join(rl_baselines_zoo_dir, 'hyperparams/{}.yml'.format(algo))

  is_atari = False
  if 'NoFrameskip' in env_id:
    is_atari = True

  # Load hyperparameters from yaml file
  with open(yml_path, 'r') as f:
    hyperparams_dict = yaml.load(f)
    if env_id in list(hyperparams_dict.keys()):
      hyperparams = hyperparams_dict[env_id]
    elif is_atari:
      hyperparams = hyperparams_dict['atari']
    else:
      raise ValueError("Hyperparameters not found for {}-{}".format(algo, env_id))

  # Sort hyperparams that will be saved
  saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
  # all_hyperparams = copy.deepcopy(hyperparams)

  algo_ = algo
  # HER is only a wrapper around an algo
  if algo == 'her':
    algo_ = saved_hyperparams['model_class']
    assert algo_ in {'sac', 'ddpg', 'dqn'}, "{} is not compatible with HER".format(algo_)
    # Retrieve the model class
    hyperparams['model_class'] = ALGOS[saved_hyperparams['model_class']]

  if verbose > 0:
    pprint.pprint(saved_hyperparams)

  n_envs = hyperparams.get('n_envs', 1)

  if verbose > 0:
    print("Using {} environments".format(n_envs))

  # Create learning rate schedules for ppo2 and sac
  if algo_ in ["ppo2", "sac"]:
    for key in ['learning_rate', 'cliprange']:
      if key not in hyperparams:
        continue
      if isinstance(hyperparams[key], str):
        schedule, initial_value = hyperparams[key].split('_')
        initial_value = float(initial_value)
        hyperparams[key] = linear_schedule(initial_value)
      elif isinstance(hyperparams[key], float):
        hyperparams[key] = constfn(hyperparams[key])
      else:
        raise ValueError('Invalid valid for {}: {}'.format(key, hyperparams[key]))

  # # Should we overwrite the number of timesteps?
  # if n_timesteps > 0:
  #   if verbose:
  #     print("Overwriting n_timesteps with n={}".format(n_timesteps))
  # else:
  #   n_timesteps = int(hyperparams['n_timesteps'])

  normalize = False
  normalize_kwargs = {}
  if 'normalize' in hyperparams.keys():
    normalize = hyperparams['normalize']
    if isinstance(normalize, str):
      normalize_kwargs = eval(normalize)
      normalize = True
    del hyperparams['normalize']

  if 'policy_kwargs' in hyperparams.keys():
    hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])

  # Delete keys so the dict can be pass to the model constructor
  if 'n_envs' in hyperparams.keys():
    del hyperparams['n_envs']
  del hyperparams['n_timesteps']

  # obtain a class object from a wrapper name string in hyperparams
  # and delete the entry
  env_wrapper = get_wrapper_class(hyperparams)
  if 'env_wrapper' in hyperparams.keys():
    del hyperparams['env_wrapper']

  def create_env(n_envs):
    """
    Create the environment and wrap it if necessary
    :param n_envs: (int)
    :return: (gym.Env)
    """
    # global hyperparams

    if is_atari:
      if verbose > 0:
        print("Using Atari wrapper")
      env = make_atari_env(env_id, num_env=n_envs, seed=seed)
      # Frame-stacking with 4 frames
      env = VecFrameStack(env, n_stack=4)
    elif algo_ in ['dqn', 'ddpg']:
      if hyperparams.get('normalize', False):
        print("WARNING: normalization not supported yet for DDPG/DQN")
      env = gym.make(env_id)
      env.seed(seed)
      if env_wrapper is not None:
        env = env_wrapper(env)
    else:
      if n_envs == 1:
        env = DummyVecEnv([make_env(env_id, 0, seed, wrapper_class=env_wrapper)])
      else:
        # env = SubprocVecEnv([make_env(env_id, i, seed) for i in range(n_envs)])
        # On most env, SubprocVecEnv does not help and is quite memory hungry
        env = DummyVecEnv([make_env(env_id, i, seed, wrapper_class=env_wrapper) for i in range(n_envs)])
      if normalize:
        if verbose > 0:
          if len(normalize_kwargs) > 0:
            print("Normalization activated: {}".format(normalize_kwargs))
          else:
            print("Normalizing input and reward")
        env = VecNormalize(env, **normalize_kwargs)
    # Optional Frame-stacking
    if hyperparams.get('frame_stack', False):
      n_stack = hyperparams['frame_stack']
      env = VecFrameStack(env, n_stack)
      print("Stacking {} frames".format(n_stack))
      del hyperparams['frame_stack']
    return env


  env = create_env(n_envs)
  # Stop env processes to free memory
  # if optimize_hyperparameters and n_envs > 1:
  #   env.close()

  # Parse noise string for DDPG and SAC
  if algo_ in ['ddpg', 'sac'] and hyperparams.get('noise_type') is not None:
    noise_type = hyperparams['noise_type'].strip()
    noise_std = hyperparams['noise_std']
    n_actions = env.action_space.shape[0]
    if 'adaptive-param' in noise_type:
      assert algo_ == 'ddpg', 'Parameter is not supported by SAC'
      hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                          desired_action_stddev=noise_std)
    elif 'normal' in noise_type:
      hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                      sigma=noise_std * np.ones(n_actions))
    elif 'ornstein-uhlenbeck' in noise_type:
      hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                 sigma=noise_std * np.ones(n_actions))
    else:
      raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
    print("Applying {} noise with std {}".format(noise_type, noise_std))
    del hyperparams['noise_type']
    del hyperparams['noise_std']

  # if trained_agent.endswith('.pkl') and os.path.isfile(trained_agent):
  #   # Continue training
  #   print("Loading pretrained agent")
  #   # Policy should not be changed
  #   del hyperparams['policy']
  #
  #   # model = ALGOS[algo].load(trained_agent, env=env,
  #   #                          tensorboard_log=tensorboard_log, verbose=verbose, **hyperparams)
  #
  #   exp_folder = trained_agent.split('.pkl')[0]
  #   if normalize:
  #     print("Loading saved running average")
  #     env.load_running_average(exp_folder)

  # elif optimize_hyperparameters:
  #
  #   if verbose > 0:
  #     print("Optimizing hyperparameters")
  #
  #
  #   def create_model(*_args, **kwargs):
  #     """
  #     Helper to create a model with different hyperparameters
  #     """
  #     return ALGOS[algo](env=create_env(n_envs), tensorboard_log=tensorboard_log,
  #                        verbose=0, **kwargs)
  #
  #
  #   data_frame = hyperparam_optimization(algo, create_model, create_env, n_trials=n_trials,
  #                                        n_timesteps=n_timesteps, hyperparams=hyperparams,
  #                                        n_jobs=n_jobs, seed=seed,
  #                                        sampler_method=sampler, pruner_method=pruner,
  #                                        verbose=verbose)
  #
  #   report_name = "report_{}_{}-trials-{}-{}-{}.csv".format(env_id, n_trials, n_timesteps,
  #                                                           sampler, pruner)
  #
  #   log_path = os.path.join(log_folder, algo, report_name)
  #
  #   if verbose:
  #     print("Writing report to {}".format(log_path))
  #
  #   os.makedirs(os.path.dirname(log_path), exist_ok=True)
  #   data_frame.to_csv(log_path)
  #   exit()
  # else:
  # Train an agent from scratch
  model = ALGOS[algo](env=env,
                      # tensorboard_log=tensorboard_log,
                      verbose=verbose, _init_setup_model=build_model_layers, **hyperparams)

  kwargs = {}
  if log_interval > -1:
    kwargs = {'log_interval': log_interval}

  env.close()

  return {
    # Plain old yml parameters without any interpretation.
    'hyperparams': saved_hyperparams,
    # yml parameters with some processing (e.g., eval python expression) but don't delete hyperparams.
    # 'yml_hyperparams': saved_hyperparams,
    # Processed hyperparameters suitable for kwargs for the stable-baseliens algorithm class (e.g, DDPG, SAC, etc.); a.k.a. "model".
    'model_kwargs': hyperparams,
    'model': model,
    # 'env': env,
    'learn_kwargs': kwargs,
  }

def get_wrapper_class(hyperparams):
  """
  Get a Gym environment wrapper class specified as a hyper parameter
  "env_wrapper".
  e.g.
  env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

  :param hyperparams: (dict)
  :return: a subclass of gym.Wrapper (class object) you can use to
           create another Gym env giving an original env.
  """

  def get_module_name(fullname):
    return '.'.join(wrapper_name.split('.')[:-1])

  def get_class_name(fullname):
    return wrapper_name.split('.')[-1]

  if 'env_wrapper' in hyperparams.keys():
    wrapper_name = hyperparams.get('env_wrapper')
    wrapper_module = importlib.import_module(get_module_name(wrapper_name))
    return getattr(wrapper_module, get_class_name(wrapper_name))
  else:
    return None

def make_env(env_id, rank=0, seed=0, log_dir=None, wrapper_class=None):
  """
  Helper function to multiprocess training
  and log the progress.

  :param env_id: (str)
  :param rank: (int)
  :param seed: (int)
  :param log_dir: (str)
  :param wrapper: (type) a subclass of gym.Wrapper to wrap the original
                  env with
  """
  if log_dir is None and log_dir != '':
    log_dir = "/tmp/gym/{}/".format(int(time.time()))
  os.makedirs(log_dir, exist_ok=True)

  def _init():
    set_global_seeds(seed + rank)
    env = gym.make(env_id)

    # Dict observation space is currently not supported.
    # https://github.com/hill-a/stable-baselines/issues/321
    # We allow a Gym env wrapper (a subclass of gym.Wrapper)
    if wrapper_class:
      env = wrapper_class(env)

    env.seed(seed + rank)
    env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=True)
    return env

  return _init
