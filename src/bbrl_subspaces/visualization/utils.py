import torch
import numpy as np

from cvxopt import matrix, solvers

from bbrl.workspace import Workspace



def evaluate_agent(eval_agent, alpha):
    workspace = Workspace()
    eval_agent(workspace, t=0, stop_variable="env/done", alphas=alpha)
    rewards = workspace["env/cumulated_reward"][-1]
    mean_reward = rewards.mean()
    return mean_reward



def intersection_point(m, b, x_coord):
    "Calculates the intersection point of an y = mx'+b axis with the x = x_coord axis"
    # Calculate y-coordinate of intersection point
    y = m * (x_coord) + b
    return x_coord, y



def find_axis_through_point(point, slope):
    # Point coordinates
    x_h, y_h = point
    # Equation of the axis passing through H: y = mx + b
    # We know that the slope is the same as the given slope, so b = y - mx
    b = y_h - slope * x_h
    return slope, b



def generate_left_edge_points(num_points):
    # Generate x values linearly spaced between 0 and 1/2
    x_values = np.linspace(0, 0.5, num_points)
    # Calculate corresponding y values based on the equation sqrt(3) * x
    y_values = np.sqrt(3) * x_values
    # Combine x and y values into a list of points
    points = list(zip(x_values, y_values))
    return points



def generate_lower_edge_points(num_points):
    # Generate x values linearly spaced between 0 and 1
    x_values = np.linspace(0, 1, num_points)
    # Set y values to zero for all points
    y_values = np.zeros_like(x_values)
    # Combine x and y values into a list of points
    points = list(zip(x_values, y_values))
    return points



def get_alphas_from_point(x, y):
    triangle_vertices = [[0, 0],[1, 0], [0.5, np.sqrt(3)/2]]
    A = np.vstack([triangle_vertices[0], triangle_vertices[1], triangle_vertices[2]]).T
    A = np.vstack([A, np.ones(3)])
    b = np.array([x, y, 1])  

    # Solving the equations system
    alphas = np.linalg.lstsq(A, b, rcond=None)[0]    
    alphas = np.maximum(alphas, 0)  # Ensure that all values are positive
    alphas /= np.sum(alphas)    # Normalize
    return torch.Tensor(alphas)



def is_inside_triangle(point, A, B, C):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(point, A, B) < 0.0
    b2 = sign(point, B, C) < 0.0
    b3 = sign(point, C, A) < 0.0
    return ((b1 == b2) and (b2 == b3))



def projection_convex_hull(p, p1, p2, p3):
    """
    Find the projection of a point onto the convex hull defined by three points p1, p2 and p3.

    Parameters:
        p (numpy.array): The point to be projected.
        p1 (numpy.array): First vector defining the convex hull.
        p2 (numpy.array): Second vector defining the convex hull.
        p3 (numpy.array): Third vector defining the convex hull.

    Returns:
        numpy.array: The projected points coefficients.
    """
    # Ensure all vectors have the same dimension
    assert p.shape == p1.shape == p2.shape == p3.shape, "All vectors must have the same dimension"

    n = 3
    # Construct the P and q matrices for the QP problem
    p_mat = 2 * np.array(
        [[np.dot(p1, p1.T), np.dot(p1, p2.T), np.dot(p1, p3.T)],
         [np.dot(p1, p2.T), np.dot(p2, p2.T), np.dot(p2, p3.T)],
         [np.dot(p1, p3.T), np.dot(p2, p3.T), np.dot(p3, p3.T)]])
    q_mat = -2 * np.array([np.dot(p, p1),
                           np.dot(p, p2),
                           np.dot(p, p3)])

    g_mat = np.array([[-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]])
    
    h_mat = np.array([0, 0, 0])
    a_mat = np.array([[1, 1, 1]])
    b_mat = np.array([1.])

    p_mat = p_mat.astype(np.double)
    q_mat = q_mat.astype(np.double)
    g_mat = g_mat.astype(np.double) 
    h_mat = h_mat.astype(np.double)
    a_mat = a_mat.astype(np.double)
    b_mat = b_mat.astype(np.double)

    # Convert matrices to cvxopt format
    p_mat = matrix(p_mat, (n, n), 'd')
    q_mat = matrix(q_mat, (n, 1), 'd')
    g_mat = matrix(g_mat, (n, n), 'd')
    h_mat = matrix(h_mat, (n, 1), 'd')
    a_mat = matrix(a_mat, (1, n), 'd')
    b_mat = matrix(b_mat, (1, 1), 'd')

    sol = solvers.qp(p_mat, q_mat, G=g_mat, h=h_mat, A=a_mat, b=b_mat)

    # Convert solution to tuple (x, y, z) and return the projected point
    a1, a2, a3 = np.array(sol['x'])
    return a1, a2, a3