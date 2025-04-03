from solution_1a_Zub import solution_1a
from solution_Zlb import Solution_Zlb
from steiner_solution import minimiseCostWithAnnotations

def get_zub_zlb(points, stc, stic, ec, lambda_n, speed, capacity, alpha, beta, result_of_cost, connected_edges, commodities):
    # print(points)
    zub = solution_1a(points, stc, stic, ec, result_of_cost, connected_edges, speed, capacity, alpha, beta, commodities)
    zlb, fij, yij = Solution_Zlb(points, connected_edges, stc, stic, ec, lambda_n, speed, capacity, alpha, beta, commodities)

    return zub, zlb, fij, yij