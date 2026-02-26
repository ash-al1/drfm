import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

env = make_vec_env("Walker2d-v4", n_envs=4)

model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        verbose=1,
        tensorboard_log="./sac_walker_tensorboard/",
)

eval_env = gym.make("Walker2d-v4", render_mode="human")

eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=50_000,
        deterministic=True,
        render=False,
)

model.learn(
        total_timesteps=1_000_000,
        callback=eval_callback,
        progress_bar=True,
)

model.save("sac_walker2d")
obs, _ = eval_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    eval_env.render()
    if terminated or truncated:
        obs, _ = eval_env.reset()
