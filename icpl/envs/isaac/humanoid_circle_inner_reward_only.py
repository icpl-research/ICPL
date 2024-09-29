@torch.jit.script
def compute_inner_reward(obs_buf, 
reset_buf, 
progress_buf, 
actions, 
energy_cost_scale, 
joints_at_limit_cost_scale, 
termination_height,
death_cost, 
max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, float, float, float) -> Tensor
    # Reward is based on forward velocity
    reward = obs_buf[:, 4]
    # Penalize energy usage
    energy_cost = energy_cost_scale * torch.sum(actions**2, dim=-1)
    reward -= energy_cost
    # Penalize joints at limit
    joints_at_limit_cost = joints_at_limit_cost_scale * torch.sum(obs_buf[:, 12:32] > 0.99,
    dim=-1)
    reward -= joints_at_limit_cost
    # Reward for alternating leg movements
    leg_difference = torch.abs(obs_buf[:, 12] - obs_buf[:, 13])
    reward += leg_difference
    # Penalize for not using both legs equally
    leg_usage_difference = torch.abs(torch.sum(obs_buf[:, 12:14], dim=-1) - 1.0)
    reward -= leg_usage_difference
    # Penalize for jumping (both feet off the ground)
    feet_off_ground = torch.sum(obs_buf[:, 54:56] == 0)
    reward -= feet_off_ground
    # Reward for maintaining an upright torso
    upright_torso = 1 - torch.abs(1 - obs_buf[:, 11])
    reward += 2 * upright_torso # Increase reward for upright torso
    # Penalize for low torso position
    low_torso = obs_buf[:, 2] < termination_height
    reward = torch.where(low_torso, torch.ones_like(reward) * -1.0, reward)
    # Penalize for high angular velocity
    high_ang_velocity = torch.abs(obs_buf[:, 6:9]).sum(dim=-1) > 1.0
    reward = torch.where(high_ang_velocity, torch.ones_like(reward) * -1.0, reward)
    # Penalize for falling
    not_upright = torch.abs(1 - obs_buf[:, 11]) > 0.25
    reward = torch.where(not_upright, torch.ones_like(reward) * death_cost, reward)
    # Reset if fallen
    reset = torch.where(not_upright, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf),
    reset)
    return reward
