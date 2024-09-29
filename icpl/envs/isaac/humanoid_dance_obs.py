class HumanoidDance(VecTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        
        self.obs_buf[:], self.torso_position[:], self.prev_torso_position[:], self.velocity_world[:], self.angular_velocity_world[:], self.velocity_local[:], self.angular_velocity_local[:], self.up_vec[:], self.heading_vec[:] = compute_humanoid_dance_observations(
            self.obs_buf, self.root_states, self.torso_position,
            self.inv_start_rot, self.dof_pos, self.dof_vel, self.dof_force_tensor,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale, self.angular_velocity_scale,
            self.basis_vec0, self.basis_vec1)

def compute_humanoid_dance_observations(obs_buf, root_states, torso_position, inv_start_rot, dof_pos, dof_vel,
                                  dof_force, dof_limits_lower, dof_limits_upper, dof_vel_scale,
                                  sensor_force_torques, actions, dt, contact_force_scale, angular_velocity_scale,
                                  basis_vec0, basis_vec1):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, float, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    prev_torso_position_new = torso_position.clone()

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity_world = root_states[:, 7:10]
    angular_velocity_world = root_states[:, 10:13]

    torso_quat, up_proj, up_vec, heading_vec = compute_heading_and_up_vec(
        torso_rotation, inv_start_rot, basis_vec0, basis_vec1, 2)

    velocity_local, angular_velocity_local, roll, pitch, yaw = compute_rot_new(
        torso_quat, velocity_world, angular_velocity_world)

    roll = normalize_angle(roll).unsqueeze(-1)
    yaw = normalize_angle(yaw).unsqueeze(-1)
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)
    scale_angular_velocity_local = angular_velocity_local * angular_velocity_scale

    obs = torch.cat((root_states[:, 0:3].view(-1, 3), velocity_local, scale_angular_velocity_local,
                     yaw, roll, up_proj.unsqueeze(-1),
                     dof_pos_scaled, dof_vel * dof_vel_scale, dof_force * contact_force_scale,
                     sensor_force_torques.view(-1, 12) * contact_force_scale, actions), dim=-1)

    return obs, torso_position, prev_torso_position_new, velocity_world, angular_velocity_world, velocity_local, scale_angular_velocity_local, up_vec, heading_vec