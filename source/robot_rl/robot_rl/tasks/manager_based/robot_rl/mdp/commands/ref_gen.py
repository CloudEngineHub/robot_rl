import math
import torch
from torch import Tensor
from typing import Tuple

# Combination formula for Bezier coefficients
def _ncr(n: int, r: int) -> int:
    return math.comb(n, r)


def bezier_deg(
    order: int,
    tau: torch.Tensor,
    step_dur: torch.Tensor,
    control_points: torch.Tensor,
    degree: int,
) -> torch.Tensor:
    """
    Computes the Bezier curve (order=0) or its time-derivative (order=1).
    Args:
        order: 0 for position, 1 for derivative
        tau: Tensor of shape [batch], clipped to [0,1]
        step_dur: Tensor of shape [batch]
        control_points: Tensor of shape [batch, degree+1]
        degree: polynomial degree
    Returns:
        Tensor of shape [batch]
    """
    # Ensure tau and step_dur are [batch]
    tau = torch.clamp(tau, 0.0, 1.0)
    batch = tau.size(0)

    if order == 1:
        # derivative of Bezier
        cp_diff = control_points[:, 1:] - control_points[:, :-1]  # [batch, degree]
        coefs = torch.tensor([_ncr(degree - 1, i) for i in range(degree)],
                             dtype=control_points.dtype,
                             device=control_points.device)  # [degree]
        i = torch.arange(degree, device=control_points.device)
        tau_pow = tau.unsqueeze(1) ** i.unsqueeze(0)                # [batch, degree]
        one_minus_pow = (1 - tau).unsqueeze(1) ** (degree - 1 - i).unsqueeze(0)  # [batch, degree]
        terms = degree * cp_diff * coefs.unsqueeze(0) * one_minus_pow * tau_pow
        dB = terms.sum(dim=1) / step_dur                              # [batch]
        return dB
    else:
        # position of Bezier
        coefs = torch.tensor([_ncr(degree, i) for i in range(degree + 1)],
                             dtype=control_points.dtype,
                             device=control_points.device)  # [degree+1]
        i = torch.arange(degree + 1, device=control_points.device)
        tau_pow = tau.unsqueeze(1) ** i.unsqueeze(0)                 # [batch, degree+1]
        one_minus_pow = (1 - tau).unsqueeze(1) ** (degree - i).unsqueeze(0)  # [batch, degree+1]
        terms = control_points * coefs.unsqueeze(0) * one_minus_pow * tau_pow  # [batch, degree+1]
        B = terms.sum(dim=1)                                          # [batch]
        return B


def calculate_cur_swing_foot_pos(
    bht: torch.Tensor,
    z_init: torch.Tensor,
    z_sw_max: torch.Tensor,
    tau: torch.Tensor,
    step_x_init: torch.Tensor,
    step_y_init: torch.Tensor,
    T_gait: torch.Tensor,
    zsw_neg: torch.Tensor,
    clipped_step_x: torch.Tensor,
    clipped_step_y: torch.Tensor,
) -> torch.Tensor:
    """
    Batch-friendly swing foot position calculation.
    Args:
        bht: [batch]
        p_sw0: [batch,3]
        z_sw_max: [batch]
        tau: [batch]
        T_gait: [batch]
        zsw_neg: [batch]
        clipped_step_x: [batch]
        clipped_step_y: [batch]
    Returns:
        p_swing: [batch,3]
    """
    # Vertical Bezier control points (degree 5)
    degree_v = 6
    control_v = torch.stack([
        z_init,                      # Start
        z_init + 0.2 * (z_sw_max - z_init),
        z_init + 0.6 * (z_sw_max - z_init),
        z_sw_max,                    # Peak at mid-swing
        zsw_neg + 0.5 * (z_sw_max - zsw_neg),
        zsw_neg + 0.05 * (z_sw_max - zsw_neg),
        zsw_neg                      # End
    ], dim=1)

    # Horizontal X and Y (linear interpolation)
    p_swing_x = ((1 - bht) * step_x_init + bht * clipped_step_x).unsqueeze(1)
    p_swing_y = ((1 - bht) * step_y_init + bht * clipped_step_y).unsqueeze(1)

    # Z via 5th-degree Bezier
    p_swing_z = bezier_deg(
        0, tau, T_gait, control_v, degree_v
    ).unsqueeze(1)

    v_swing_z = bezier_deg(
        1, tau, T_gait, control_v, degree_v
    ).unsqueeze(1)

    return torch.cat([p_swing_x, p_swing_y, p_swing_z], dim=1), v_swing_z  # [batch,3]


def calculate_cur_swing_foot_pos_stair(
    bht: torch.Tensor,
    z_init: torch.Tensor,
    z_sw_max: torch.Tensor,
    tau: torch.Tensor,
    step_x_init: torch.Tensor,
    step_y_init: torch.Tensor,
    T_gait: torch.Tensor,
    zsw_neg: torch.Tensor,
    clipped_step_x: torch.Tensor,
    clipped_step_y: torch.Tensor,
) -> torch.Tensor:
    """
    Batch-friendly swing foot position calculation.
    Args:
        bht: [batch]
        p_sw0: [batch,3]
        z_sw_max: [batch]
        tau: [batch]
        T_gait: [batch]
        zsw_neg: [batch]
        clipped_step_x: [batch]
        clipped_step_y: [batch]
    Returns:
        p_swing: [batch,3]
    """
    # Vertical Bezier control points (degree 5)
    degree_v = 6
    control_v = torch.stack([
        z_init,                      # Start
        z_init + 0.6 * (z_sw_max - z_init),
        z_sw_max,
        z_sw_max,                    # Peak at mid-swing
        zsw_neg + 0.5 * (z_sw_max - zsw_neg),
        zsw_neg + 0.05 * (z_sw_max - zsw_neg),
        zsw_neg                      # End
    ], dim=1)

    # Horizontal X and Y (linear interpolation)
    p_swing_x = ((1 - bht) * step_x_init + bht * clipped_step_x).unsqueeze(1)
    p_swing_y = ((1 - bht) * step_y_init + bht * clipped_step_y).unsqueeze(1)

    # Z via 5th-degree Bezier
    p_swing_z = bezier_deg(
        0, tau, T_gait, control_v, degree_v
    ).unsqueeze(1)

    v_swing_z = bezier_deg(
        1, tau, T_gait, control_v, degree_v
    ).unsqueeze(1)

    return torch.cat([p_swing_x, p_swing_y, p_swing_z], dim=1), v_swing_z  # [batch,3]



def coth(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.tanh(x)




class HLIP(torch.nn.Module):
    """Hybrid Linear Inverted Pendulum implementation using PyTorch."""
    def __init__(self, grav: torch.Tensor, z0: float, T_ds: float, T: torch.Tensor, y_nom: float):
        """
        Initializes the Hybrid Linear Inverted Pendulum (HLIP) controller.
        """
        super().__init__()
        # Store properties
        self.grav = grav
        self.z0 = z0
        self.y_nom = y_nom
        self.T_ds = T_ds
        self.T = T
        self.device = grav.device
        self.num_envs = grav.shape[0]

        self.lambda_ = torch.sqrt(self.grav / self.z0)

        # -- START OF FIX --
        # Build state-space matrices as BATCHED tensors from the start.
        # Shape: (num_envs, 2, 2)
        # Build state-space matrices as BATCHED tensors from the start.
        # Shape: (num_envs, 2, 2)
        self.A_ss = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self.A_ss[:, 0, 1] = 1.0
        self.A_ss[:, 1, 0] = self.lambda_**2

        # A_ds is CONSTANT across all envs, so it should be a single 2D matrix.
        self.A_ds = torch.tensor([[0.0, 1.0], [0.0, 0.0]], device=self.device)

        # B_usw depends on lambda_, so it MUST be batched.
        # Shape: (num_envs, 2, 1)
        self.B_usw = torch.zeros(self.num_envs, 2, 1, device=self.device)
        self.B_usw[:, 1, 0] = -self.lambda_**2
        # -- END OF FIX --
        
        # This can now be called safely
        self._compute_s2s_matrices()
    def _compute_s2s_matrices(self) -> None:
        """Compute and store step-to-step A and B matrices."""
        # Matrices are already on the correct device from __init__

        # Reshape the scalar part of the expression to allow for broadcasting.
        scalar_term = (self.T - self.T_ds).view(-1, 1, 1)

        # Create a batched A_ss matrix of shape (num_envs, 2, 2).
        batched_A_ss = self.A_ss * scalar_term
        
        # exp_ss now has shape (num_envs, 2, 2).
        exp_ss = torch.matrix_exp(batched_A_ss)

        # exp_ds is a single (2, 2) matrix.
        exp_ds = torch.matrix_exp(self.A_ds * self.T_ds)
        
        num_envs = exp_ss.shape[0]
        
        # To multiply with the batched exp_ss, we must expand the single exp_ds matrix
        # Shape: (2, 2) -> (1, 2, 2) -> (num_envs, 2, 2)
        batched_exp_ds = exp_ds.unsqueeze(0).expand(num_envs, -1, -1)
        
        # bmm requires both tensors to be batched
        self.A_s2s = torch.bmm(exp_ss, batched_exp_ds)
        
        # B_usw is already batched from __init__, so we can use it directly
        self.B_s2s = torch.bmm(exp_ss, self.B_usw)
        # -- END OF FIX --
        
        # Note: The redundant lines at the end have been removed.

    def _remap_for_init_stance_state(self, Xdes, Ydes, Ux, Uy):
        """
        Remaps the desired states for the next step to be the initial states for the
        two feet, which is used for legged robots.

        This function calculates the state [pos, vel] for the next stance leg and next swing leg.
        """
        # -- START OF THE FINAL FIX --

        # 1. Calculate the state [pos, vel] for the next stance leg (current swing leg)
        # pos = current swing foot pos relative to new stance foot = X_des - u_sw
        # vel = current swing foot vel relative to new stance foot = V_des
        x_pos_stance_next = Xdes[:, 0, 0] - Ux
        x_vel_stance_next = Xdes[:, 1, 0]
        # Stack them into a (batch, 2) tensor
        x_init_stance_next = torch.stack((x_pos_stance_next, x_vel_stance_next), dim=1)

        y_pos_stance_next = Ydes[:, 0, 0] - Uy
        y_vel_stance_next = Ydes[:, 1, 0]
        y_init_stance_next = torch.stack((y_pos_stance_next, y_vel_stance_next), dim=1)


        # 2. Calculate the state [pos, vel] for the next swing leg (current stance leg)
        # pos = current stance foot pos relative to new stance foot = -u_sw
        # vel = current stance foot vel relative to new stance foot = -V_des
        x_pos_swing_next = -Xdes[:, 0, 0] + Ux
        x_vel_swing_next = -Xdes[:, 1, 0]
        # Stack them into a (batch, 2) tensor
        x_init_swing_next = torch.stack((x_pos_swing_next, x_vel_swing_next), dim=1)

        y_pos_swing_next = -Ydes[:, 0, 0] + Uy
        y_vel_swing_next = -Ydes[:, 1, 0]
        y_init_swing_next = torch.stack((y_pos_swing_next, y_vel_swing_next), dim=1)


        # 3. Combine the two legs into a single state tensor.
        # The result is shape (batch, 2, 2), where dims are (batch, [pos, vel], [stance, swing])
        x_init = torch.stack((x_init_stance_next, x_init_swing_next), dim=2)
        y_init = torch.stack((y_init_stance_next, y_init_swing_next), dim=2)
        
        # -- END OF THE FINAL FIX --
        
        return x_init, y_init
    def _compute_desire_com_trajectory(self, cur_time, Xdesire):
        """
        Computes the desired com trajectory at the current time.

        Args:
            cur_time: The current time in the swing phase.
            Xdesire: The desired state [x, x_dot] as a tensor of shape (batch, 2).
        """
        # -- START OF FIX --
        # Xdesire is a 2D tensor of shape (batch, 2).
        # We index it accordingly to get the position (column 0) and velocity (column 1).
        x0 = Xdesire[:, 0]
        v0 = Xdesire[:, 1]
        # -- END OF FIX --
        
        c = torch.cosh(self.lambda_ * cur_time)
        s = torch.sinh(self.lambda_ * cur_time)

        # desired com position and velocity
        xd = x0 * c + v0 * s / self.lambda_
        vd = x0 * self.lambda_ * s + v0 * c

        return xd, vd

    def _solve_deadbeat_gain(self, A: Tensor, B: Tensor) -> Tensor:
        """Solve for deadbeat gains."""
        A_tmp = torch.stack([
            torch.tensor([-B[0], -B[1]]),
            torch.tensor([
                A[1, 1] * B[0] - A[0, 1] * B[1],
                A[0, 0] * B[1] - A[1, 0] * B[0]
            ])
        ])
        B_tmp = torch.tensor([A[0, 0] + A[1, 1], A[0, 1] * A[1, 0] - A[0, 0] * A[1, 1]])
        return torch.linalg.solve(A_tmp, B_tmp)

    def compute_desired_orbit(self, cmd: torch.Tensor, T: torch.Tensor):
        """
        Computes the desired final state of the LIP model based on the given command.

        Args:
            cmd: Tensor of shape (batch, 2) containing the desired velocity command [vx, vy].
            T: Tensor of shape (batch,) containing the step duration.

        Returns:
            A tuple of tensors for the desired final states and foot placements.
        """
        V_des = cmd
        c = torch.cosh(self.lambda_ * T)
        s = torch.sinh(self.lambda_ * T)

        U_des_p1 = (V_des[:, 0] / self.lambda_) * s / (c - 1)
        # Apply a small offset to the sign calculation to prevent zero-crossings from being exactly zero
        sign_vy = torch.sign(V_des[:, 1] + 1e-6)
        U_des_p2 = (V_des[:, 1] / self.lambda_) * s / (c - 1) - self.y_nom * sign_vy

        eye_expanded = torch.eye(2, device=self.device).expand(self.num_envs, -1, -1)
        
        # --- START OF FIX ---
        # Solve for the X-axis dynamics and name the result 'Xdes'
        Bu_x = self.B_s2s * U_des_p1.view(-1, 1, 1)
        Xdes = torch.linalg.solve(eye_expanded - self.A_s2s, Bu_x)

        # Solve for the Y-axis dynamics and name the result 'Ydes'
        Bu_y = self.B_s2s * U_des_p2.view(-1, 1, 1)
        Ydes = torch.linalg.solve(eye_expanded - self.A_s2s, Bu_y)
        
        # Combine them for the subsequent calculations that use the mixed states
        Combined_des = torch.cat([Xdes, Ydes], dim=-1)
        # --- END OF FIX ---

        c_Tds = torch.cosh(self.lambda_ * self.T_ds)
        s_Tds = torch.sinh(self.lambda_ * self.T_ds)
        c_Tss = torch.cosh(self.lambda_ * (T - self.T_ds))
        s_Tss = torch.sinh(self.lambda_ * (T - self.T_ds))

        # Note: The indexing on Combined_des is correct.
        # [:,:,0] gets the X states, [:,:,1] gets the Y states.
        Ux = (
            V_des[:, 0] * (c_Tds * c_Tss + s_Tds * s_Tss)
            - Combined_des[:, 0, 0] * c_Tss
            - Combined_des[:, 1, 0] * s_Tss / self.lambda_
        )
        Uy = (
            V_des[:, 1] * (c_Tds * c_Tss + s_Tds * s_Tss)
            - Combined_des[:, 0, 1] * c_Tss
            - Combined_des[:, 1, 1] * s_Tss / self.lambda_
        )

        # This will now work because Xdes, Ux, Ydes, and Uy all exist.
        return Xdes, Ux, Ydes, Uy
    def compute_orbit(
        self, T: float, cmd: Tensor
    ) ->  None:
        """Compute desired orbit."""
        # Get desired orbit params
        Xdes, Ux, Ydes, Uy = self.compute_desired_orbit(cmd[:,:2], T)
        # Remap for initial stance
        self.x_init, self.y_init = self._remap_for_init_stance_state(Xdes, Ydes, Ux, Uy)
     
        return Xdes, Ux, Ydes, Uy
   
