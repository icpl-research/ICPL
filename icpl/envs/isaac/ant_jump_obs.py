class AntJump(VecTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.obs_buf[:], self.torso_position[:], self.prev_torso_position[:], self.up_vec[:], self.heading_vec[:], self.velocity[:], self.ang_velocity[:] = compute_ant_jump_observations(
            self.obs_buf, self.root_states, self.torso_position,
            self.inv_start_rot, self.dof_pos, self.dof_vel,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)

def compute_ant_jump_observations(obs_buf, root_states, torso_position,
                             inv_start_rot, dof_pos, dof_vel,
                             dof_limits_lower, dof_limits_upper, dof_vel_scale,
                             sensor_force_torques, actions, dt, contact_force_scale,
                             basis_vec0, basis_vec1, up_axis_idx):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    prev_torso_position = torso_position.clone()

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    torso_quat, up_proj, up_vec, heading_vec = compute_heading_and_up_vec(
        torso_rotation, inv_start_rot, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw = compute_rot_new(
        torso_quat, velocity, ang_velocity)

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    obs = torch.cat((root_states.view(-1, 13), vel_loc, angvel_loc,
                     yaw.unsqueeze(-1), roll.unsqueeze(-1),
                     up_proj.unsqueeze(-1), dof_pos_scaled,
                     dof_vel * dof_vel_scale, sensor_force_torques.view(-1, 24) * contact_force_scale,
                     actions), dim=-1)

    hip_4 = actions[:, 0]
    ankle_4 = actions[:, 1]
    hip_1 = actions[:, 2]
    ankle_1 = actions[:, 3]
    hip_2 = actions[:, 4]
    ankle_2 = actions[:, 5]
    hip_3 = actions[:, 6]
    ankle_3 = actions[:, 7]
    
    return obs, torso_position, prev_torso_position, up_vec, heading_vec, velocity, ang_velocity



