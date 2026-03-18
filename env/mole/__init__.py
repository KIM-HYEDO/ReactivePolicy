from gymnasium.envs.registration import register

register(
    id="mole/WhackAMole-v0",
    entry_point="env.mole.mole_env:WhackAMoleReactiveEnv",
    max_episode_steps=300,
)
