"""
PPO Training with Weights & Biases (wandb) integration for experiment tracking.

Install wandb: pip install wandb

Usage:
    python train/example_wandb_training.py --env CartPole-v1
    python train/example_wandb_training.py --env Pendulum-v1 --continuous
"""

import argparse

import gymnasium as gym

import numpy as np
import os
import sys
# Add parent directory to path to import train package

from ppo_modules.ppo import PPO

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO with wandb")
    
    # Environment
    parser.add_argument("--env", type=str, default="CartPole-v1",
                       help="Environment name")
    parser.add_argument("--continuous", action="store_true",
                       help="Use continuous action space")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=500,
                       help="Maximum number of episodes")
    parser.add_argument("--max-timesteps", type=int, default=400,
                       help="Maximum timesteps per episode")
    parser.add_argument("--update-timestep", type=int, default=None,
                       help="Update policy every n timesteps (default: max_timesteps * 4)")
    
    # Hyperparameters
    parser.add_argument("--lr-actor", type=float, default=0.0003,
                       help="Learning rate for actor")
    parser.add_argument("--lr-critic", type=float, default=0.001,
                       help="Learning rate for critic")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--K-epochs", type=int, default=80,
                       help="Update policy for K epochs")
    parser.add_argument("--eps-clip", type=float, default=0.2,
                       help="Clip parameter for PPO")
    
    # Continuous action space parameters
    parser.add_argument("--action-std", type=float, default=0.6,
                       help="Initial action std for continuous actions")
    parser.add_argument("--action-std-decay-rate", type=float, default=0.05,
                       help="Action std decay rate for continuous actions")
    parser.add_argument("--min-action-std", type=float, default=0.1,
                       help="Minimum action std for continuous actions")
    parser.add_argument("--action-std-decay-freq", type=int, default=int(2.5e5),
                       help="Action std decay frequency (timesteps)")
    
    # Saving
    parser.add_argument("--save-freq", type=int, default=50,
                       help="Save model every n episodes")
    
    # Wandb
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Check wandb availability
    use_wandb = not args.no_wandb and WANDB_AVAILABLE
    
    if not WANDB_AVAILABLE and not args.no_wandb:
        print("\n" + "!" * 80)
        print("Wandb not available. Please install: pip install wandb")
        print("Training will continue without wandb logging...")
        print("!" * 80 + "\n")
    
    # Create environment
    env_name = args.env
    has_continuous_action_space = args.continuous
    env = gym.make(env_name)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    # Set update_timestep proportional to episode length (like original)
    if args.update_timestep is None:
        update_timestep = args.max_timesteps * 4
    else:
        update_timestep = args.update_timestep
    
    # Configuration dictionary
    config = {
        "env_name": env_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "has_continuous_action_space": has_continuous_action_space,
        "lr_actor": args.lr_actor,
        "lr_critic": args.lr_critic,
        "gamma": args.gamma,
        "K_epochs": args.K_epochs,
        "eps_clip": args.eps_clip,
        "action_std_init": args.action_std if has_continuous_action_space else None,
        "max_episodes": args.episodes,
        "max_timesteps": args.max_timesteps,
        "update_timestep": update_timestep,
    }
    
    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="ppo-pytorch",
            name=f"ppo_{env_name}",
            config=config,
            tags=["continuous" if has_continuous_action_space else "discrete"]
        )
        print("\n✓ Wandb initialized successfully!")
    else:
        print("\n✗ Training without wandb logging")
        use_wandb = False
    
    print("\n" + "=" * 80)
    print("PPO Training with Wandb")
    print("=" * 80)
    print(f"\nEnvironment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space: {'Continuous' if has_continuous_action_space else 'Discrete'}")
    
    # Initialize PPO agent
    ppo_agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        K_epochs=args.K_epochs,
        eps_clip=args.eps_clip,
        has_continuous_action_space=has_continuous_action_space,
        action_std_init=args.action_std if has_continuous_action_space else 0.6
    )
    
    # Create save directory
    save_dir = f"PPO_models/{env_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "-" * 80)
    print("Hyperparameters:")
    for key, value in config.items():
        if value is not None:
            print(f"  {key}: {value}")
    print("-" * 80)
    
    # Training loop
    print("\nStarting training...\n")
    
    time_step = 0
    episode_rewards = []
    running_reward = 0
    
    for episode in range(1, args.episodes + 1):
        # Handle both gym and gymnasium API
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        episode_reward = 0
        episode_length = 0
        
        for t in range(1, args.max_timesteps + 1):
            # Select action
            action = ppo_agent.select_action(state)
            
            # Take action in environment
            # Handle both gym (4 values) and gymnasium (5 values) API
            step_result = env.step(action)
            if len(step_result) == 5:
                state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                state, reward, done, _ = step_result
            
            # Save reward and terminal flag
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            
            time_step += 1
            episode_reward += reward
            episode_length += 1
            
            # Update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()
                
                # Log update
                if use_wandb:
                    wandb.log({
                        "time_step": time_step,
                        "policy_update": time_step // update_timestep
                    })
            
            # Decay action std for continuous action spaces
            if has_continuous_action_space and time_step % args.action_std_decay_freq == 0:
                ppo_agent.decay_action_std(args.action_std_decay_rate, args.min_action_std)
                
                if use_wandb:
                    wandb.log({
                        "action_std": ppo_agent.action_std,
                        "time_step": time_step
                    })
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        running_reward = 0.05 * episode_reward + 0.95 * running_reward
        
        # Log episode metrics
        episode_metrics = {
            "episode": episode,
            "time_step": time_step,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "running_reward": running_reward,
        }
        
        if episode >= 10:
            episode_metrics["avg_reward_10"] = np.mean(episode_rewards[-10:])
        
        if episode >= 50:
            episode_metrics["avg_reward_50"] = np.mean(episode_rewards[-50:])
        
        if use_wandb:
            wandb.log(episode_metrics)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode:4d} | Timestep {time_step:7d} | "
                  f"Avg Reward (10): {avg_reward:7.2f} | "
                  f"Episode Reward: {episode_reward:7.2f} | "
                  f"Length: {episode_length:4d}")
        
        # Save model
        if episode % args.save_freq == 0:
            model_path = f"{save_dir}/ppo_episode_{episode}.pth"
            ppo_agent.save(model_path)
            print(f"  → Model saved: {model_path}")
    
    env.close()
    
    # Save final model
    final_model_path = f"{save_dir}/ppo_final.pth"
    ppo_agent.save(final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")
    
    # Final statistics
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"Total episodes: {args.episodes}")
    print(f"Total timesteps: {time_step}")
    print(f"Final average reward (last 50 episodes): {np.mean(episode_rewards[-50:]):.2f}")
    print(f"Best episode reward: {np.max(episode_rewards):.2f}")
    print("=" * 80)
    
    if use_wandb:
        # Log final summary
        wandb.summary["final_avg_reward"] = np.mean(episode_rewards[-50:])
        wandb.summary["best_episode_reward"] = np.max(episode_rewards)
        wandb.summary["total_timesteps"] = time_step
        
        wandb.finish()
        print("\n✓ Wandb run completed!")

