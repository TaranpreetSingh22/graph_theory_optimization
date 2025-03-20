from solution_1a_Zub import solution_1a
from solution_Zlb import Solution_Zlb
from steiner_solution import minimiseCostWithAnnotations

def get_zub_zlb(points, stc, stic, ec, lambda_n, speed, capacity, b, alpha, beta, source, destination, result_of_cost, connected_edges, src_dest):
    print(points)
    zub = solution_1a(points, stc, stic, ec, result_of_cost, connected_edges, speed, capacity, b, alpha, beta, source, destination, src_dest)
    zlb, fij, yij = Solution_Zlb(points, connected_edges, stc, stic, ec, lambda_n, speed, capacity, b, alpha, beta, source, destination, src_dest)

    return zub, zlb, fij, yij