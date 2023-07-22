import sys
sys.path.append("..")

from stable_baselines3.dqn import DQN
import tqdm as tq
from network_layers.cnn_layer import CNNStateLayer


class DQN_Agent:
    '''
        DQN based agent which can be used with our without feature extraction module
    '''
    def __init__(
                    self, model_name, policy, env, learning_rate=0.0001,
                    buffer_size=1000000, learning_starts=50000, batch_size=32,
                    tau=0.9, gamma=0.99, train_freq=4, gradient_steps=1,
                    replay_buffer_class=None, replay_buffer_kwargs=None,
                    optimize_memory_usage=False, target_update_interval=10000,
                    exploration_fraction=0.1, exploration_initial_eps=1.0,
                    exploration_final_eps=0.05, max_grad_norm=10,
                    tensorboard_log=None, policy_kwargs=None, verbose=0,
                    seed=None, device='auto', _init_setup_model=True
                ):
        super(DQN_Agent, self).__init__()
        
        self.model_name = model_name
        self.policy = policy
        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.replay_buffer_class = replay_buffer_class
        self.replay_buffer_kwargs = replay_buffer_kwargs
        self.optimize_memory_usage = optimize_memory_usage
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm
        self.tensorboard_log = tensorboard_log
        self.policy_kwargs = policy_kwargs
        self.verbose = verbose
        self.seed = seed
        self.device = device
        self._init_setup_model = _init_setup_model
        if self.policy == "MultiInputPolicy":
            # Used for feature extraction
            self.policy_kwargs = dict(
                                        features_extractor_class=CNNStateLayer,
                                        features_extractor_kwargs=dict(feature_dim=14)
                                    )
        if self.policy == "MlpPolicy":
            self.policy_kwargs = None
        self.model = DQN(
                            policy=self.policy, env=self.env,
                            learning_rate=self.learning_rate,
                            buffer_size=self.buffer_size,
                            learning_starts=self.learning_starts,
                            batch_size=self.batch_size,
                            tau=self.tau, gamma=self.gamma,
                            train_freq=self.train_freq,
                            gradient_steps=self.gradient_steps,
                            replay_buffer_class=self.replay_buffer_class,
                            replay_buffer_kwargs=self.replay_buffer_kwargs,
                            optimize_memory_usage=self.optimize_memory_usage,
                            target_update_interval=self.target_update_interval,
                            exploration_fraction=self.exploration_fraction,
                            exploration_initial_eps=self.exploration_initial_eps,
                            exploration_final_eps=self.exploration_final_eps,
                            max_grad_norm=self.max_grad_norm,
                            tensorboard_log=self.tensorboard_log,
                            policy_kwargs=self.policy_kwargs,
                            verbose=self.verbose,
                            seed=self.seed,
                            device=self.device,
                            _init_setup_model=self._init_setup_model,
                        )
    
    def train_model(self, total_timesteps):
        '''
            Train the model
        '''
        self.model.learn(total_timesteps, progress_bar=True)

    def test_model(self, env):
        '''
            Run trained model on test data
        '''
        state = env.reset()
        for i in tq.tqdm(range(len(env.nifty_df))):
            action, _ = self.model.predict(state, deterministic=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done == True:
                state = env.reset()

    def save_model(self, path, file_name):
        '''
            Save trained model
        '''
        self.model.save(path + file_name)
    
    def load_model(self, path, file_name):
        '''
            Load trained model
        '''
        self.model = self.model.load(path + file_name)
