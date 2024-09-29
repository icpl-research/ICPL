class HumanoidJump(VecTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        
        self.obs_buf[:], self.torso_position[:], self.prev_torso_position[:], self.velocity_world[:], self.angular_velocity_world[:], self.velocity_local[:], self.angular_velocity_local[:], self.up_vec[:], self.heading_vec[:], self.right_leg_contact_force[:], self.left_leg_contact_force[:] = compute_humanoid_jump_observations(
            self.obs_buf, self.root_states, self.torso_position,
            self.inv_start_rot, self.dof_pos, self.dof_vel, self.dof_force_tensor,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale, self.angular_velocity_scale,
            self.basis_vec0, self.basis_vec1)

def compute_humanoid_jump_observations(obs_buf, root_states, torso_position, inv_start_rot, dof_pos, dof_vel,
                                  dof_force, dof_limits_lower, dof_limits_upper, dof_vel_scale,
                                  sensor_force_torques, actions, dt, contact_force_scale, angular_velocity_scale,
                                  basis_vec0, basis_vec1):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, float, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

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

    right_leg_contact_force = sensor_force_torques[:, 0:3]
    left_leg_contact_force = sensor_force_torques[:, 6:9]

    abdomen_y_pos = dof_pos[:, 0]
    abdomen_z_pos = dof_pos[:, 1]
    abdomen_x_pos = dof_pos[:, 2]
    right_hip_x_pos = dof_pos[:, 3]
    right_hip_z_pos = dof_pos[:, 4]
    right_hip_y_pos = dof_pos[:, 5]
    right_knee_pos = dof_pos[:, 6]
    right_ankle_x_pos = dof_pos[:, 7]
    right_ankle_y_pos = dof_pos[:, 8]
    left_hip_x_pos = dof_pos[:, 9]
    left_hip_z_pos = dof_pos[:, 10]
    left_hip_y_pos = dof_pos[:, 11]
    left_knee_pos = dof_pos[:, 12]
    left_ankle_x_pos = dof_pos[:, 13]
    left_ankle_y_pos = dof_pos[:, 14]
    right_shoulder1_pos = dof_pos[:, 15]
    right_shoulder2_pos = dof_pos[:, 16]
    right_elbow_pos = dof_pos[:, 17]
    left_shoulder1_pos = dof_pos[:, 18]
    left_shoulder2_pos = dof_pos[:, 19]
    left_elbow_pos = dof_pos[:, 20]

    right_shoulder1_action = actions[:, 15]
    right_shoulder2_action = actions[:, 16]
    right_elbow_action = actions[:, 17]
    left_shoulder1_action = actions[:, 18]
    left_shoulder2_action = actions[:, 19]
    left_elbow_action = actions[:, 20]

    return obs, torso_position, prev_torso_position_new, velocity_world, angular_velocity_world, velocity_local, scale_angular_velocity_local, up_vec, heading_vec, right_leg_contact_force, left_leg_contact_force