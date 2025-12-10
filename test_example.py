import argparse
import os
import time
import numpy as np
import gymnasium as gym
from ppo_modules.ppo import PPO

def run_eval(env_name: str,
             model_path: str,
             episodes: int,
             max_timesteps: int,
             render: bool,
             render_delay: float,
             continuous: bool):
    # Gymnasium requires render_mode at creation time for on-screen rendering.
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if continuous else env.action_space.n

    # Same hyperparams as train_example.py (only lr/clip/K/gamma matter for loading)
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        K_epochs=80,
        eps_clip=0.2,
        has_continuous_action_space=continuous,
        action_std_init=0.6 if continuous else 0.0,
    )
    agent.deterministic = True  # use greedy actions for evaluation
    agent.load(model_path)

    scores = []
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        state = np.asarray(state, dtype=np.float32)
        ep_reward = 0.0

        for t in range(1, max_timesteps + 1):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            state = np.asarray(next_state, dtype=np.float32)

            if render:
                env.render()
                if render_delay > 0:
                    time.sleep(render_delay)

            if done:
                break

        agent.buffer.clear()  # clear rollout data after each eval episode
        scores.append(ep_reward)
        print(f"Episode {ep:3d} | Reward: {ep_reward:.2f}")

    env.close()
    avg = float(np.mean(scores)) if scores else 0.0
    print("=" * 80)
    print(f"Evaluated {episodes} episodes. Average reward: {avg:.2f}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO evaluation on a saved model")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--model", type=str,
                        default=None,
                        help="Path to saved .pth file (defaults to PPO_models/<env>/ppo_final.pth)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--max-timesteps", type=int, default=400, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--render-delay", type=float, default=0.0, help="Sleep seconds between rendered frames")
    parser.add_argument("--continuous", action="store_true", help="Use continuous action space")
    args = parser.parse_args()

    model_path = args.model or os.path.join("PPO_models", args.env, "ppo_final.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    run_eval(
        env_name=args.env,
        model_path=model_path,
        episodes=args.episodes,
        max_timesteps=args.max_timesteps,
        render=args.render,
        render_delay=args.render_delay,
        continuous=args.continuous,
    )
