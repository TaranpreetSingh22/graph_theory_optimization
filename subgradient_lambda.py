import numpy as np

# Initial step-size numerator
theta_0 = 5  

def update_lambda(lambda_n, f_ij, y_ij, Z_min_UB, LD_lambda, n):
    """
    Update lambda using the sub-gradient method with step-size constraint.

    Parameters:
    lambda_n  : np.array, current lambda values (vector)
    f_ij      : np.array, current flow solution (vector)
    y_ij      : np.array, target values (vector)
    Z_min_UB  : float, best known upper bound
    LD_lambda : float, optimal solution to LR at lambda_n
    n         : int, iteration number

    Returns:
    Updated lambda values (vector)
    """

    # Compute sub-gradient s(lambda)
    s_lambda = np.array(f_ij) - np.array(y_ij)  # This is a vector

    # Compute step size θ^n
    theta_n = theta_0 / np.sqrt(n)  # theta = 5 / sqrt(n)

    # Ensure step size is within the range 0 ≤ θ_n ≤ 2
    if (0<= theta_n <= 2):
      # Compute the denominator ||s(lambda)||^2
      denominator = np.linalg.norm(s_lambda) ** 2  # ||s(lambda)||^2
      if denominator == 0:
          print("Warning: Zero denominator in lambda update. Keeping lambda unchanged.")
          return lambda_n, _, _  # No update if denominator is zero

      # Compute lambda update
      lambda_new = np.array(lambda_n) + (theta_n * ((Z_min_UB - LD_lambda) / denominator) * s_lambda)

      # Ensure non-negativity using max element-wise
      lambda_new = np.maximum(lambda_new, 0)

      print(f"\n------------------in update lambda function----------------------")
      print(f"Lambda_n: {lambda_n}")
      print(f"s_lambda: {s_lambda}")
      print(f"theta_n: {theta_n}")
      print(f"Denominator: {denominator}")
      print(f"Lambda_new: {lambda_new}")
      print("---------------------------------------------------------------------\n")

      return lambda_new, s_lambda, theta_n
    else:
      return lambda_n, s_lambda, theta_n