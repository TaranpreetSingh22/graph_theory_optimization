import numpy as np
import matplotlib.pyplot as plt
import random
import pulp

def LR_1(points, selected_edges, stc, stic, ec, lamda, alphaa):
  print("\n---------------------------------LR1 Print statements------------------------")
  nodes = [i for i in range(len(points))]
  arcs = selected_edges
  lambda_n = {(i, j): k for (i, j), k in zip(arcs, lamda)}
  alpha = alphaa
  m1 = len(nodes) - 1
  m2 = len(arcs)

  print("nodes:",nodes)
  print("arcs:",arcs)
  print("lambda list:", lamda)
  print("lambdas:",lambda_n)
  print("alpha:",alpha)
  print("m1:",m1)
  print("m2:",m2)

  # Costs
  station_cost = stc  # in lakhs
  steiner_cost = stic  # in lakhs
  edge_cost_per_unit = ec  # in lakhs per km
  stations_with_steiner_cost = (station_cost * (len(nodes)-1)) + steiner_cost

  cost = {(i, j): edge_cost_per_unit * ((((points[i][0] - points[j][0])**2) + ((points[i][1] - points[j][1])**2))**0.5) for (i,j) in arcs}

  print("cost:",cost)
  print()

  # Define the LP problem
  lp = pulp.LpProblem("LR1_Optimization", pulp.LpMinimize)

  # Binary decision variables
  y = pulp.LpVariable.dicts("y", arcs, cat="Binary")

  # Objective function: Minimize total cost
  lp += pulp.lpSum((alpha * cost[i, j] - lambda_n[i, j]) * y[i, j] for (i, j) in arcs)

  # Constraints
  lp += pulp.lpSum(y[i, j] for (i, j) in arcs) >= m1  # Lower bound
  lp += pulp.lpSum(y[i, j] for (i, j) in arcs) <= m2  # Upper bound # Constraint (6b)

  # Print the Objective Function
  print("Objective Function:")
  print(lp.objective)  # Inbuilt method
  print("-" * 40)

  # Print All Constraints
  print("Constraints:")
  for name, constraint in lp.constraints.items():
      print(f"{name}: {constraint}")
  print("-" * 40)

  # Solve the problem
  lp.solve()

  # Output the results
  print("Status:", pulp.LpStatus[lp.status])
  print("Objective Value:", pulp.value(lp.objective))
  for (i, j) in arcs:
      print(f"y({i}, {j}):", y[i, j].varValue)

  print("-----------------------------------------------------------------------------\n")
  
  return pulp.value(lp.objective), [y[i, j].varValue for (i,j) in arcs]

from pulp import LpProblem, LpMinimize, LpVariable, lpSum

def LR_2(points, selected_edges, stc, stic, ec, lamda, beeta, speed, capacity, demand, source, destination, src_dest):
    print("\n---------------------------------LR2 Print statements------------------------")
    # Define the parameters
    print(len(points))
    nodes = [i for i in range(len(points))]
    arcs = selected_edges
    lambda_n = {(i, j): k for (i, j), k in zip(arcs, lamda)}
    beta = beeta
    b = demand
    source = source
    sink = destination

    t_ij = {(i, j): ((((points[i][0] - points[j][0])**2) + ((points[i][1] - points[j][1])**2))**0.5)/speed for (i, j) in arcs}  # t_ij
    u_ij = {arc: capacity for arc in arcs} # u_ij

    print("nodes:",nodes)
    print("arcs:",arcs)
    print("lambda list:", lamda)
    print("lambdas:",lambda_n)
    print("beta:",beta)
    print("time:",t_ij)
    print("capacity:",u_ij)

    # Create problem
    prob = LpProblem("Flow_Optimization", LpMinimize)

    # Define decision variables
    print([f"edge:{edge[0]}-{edge[1]}" for edge in arcs])
    f = pulp.LpVariable.dicts("Flow", arcs, lowBound = 0, upBound = 1, cat = "Continuous")

    # Objective function
    prob += lpSum((beta * t_ij[i, j] * b + lambda_n[i, j]) * f[i, j] for (i, j) in arcs)

    # Flow conservation constraints
    # for node in nodes:
    #   if node == source:
    #       prob += (
    #           pulp.lpSum(f[i, j] for (i, j) in arcs if i == node)
    #           - pulp.lpSum(f[j, i] for (j, i) in arcs if i == node)
    #           == 1
    #     )
    #   elif node == sink:
    #       prob += (
    #           pulp.lpSum(f[i, j] for (i, j) in arcs if i == node)
    #           - pulp.lpSum(f[j, i] for (j, i) in arcs if i == node)
    #           == -1
    #       )
    #   else:
    #       prob += (
    #           pulp.lpSum(f[i, j] for (i, j) in arcs if i == node)
    #           - pulp.lpSum(f[j, i] for (j, i) in arcs if i == node)
    #           == 0
    #       )

    for src, sink in src_dest:
      for node in nodes:
        if node == src:
            prob += (
                pulp.lpSum(f[i, j] for (i, j) in arcs if i == node)
                - pulp.lpSum(f[j, i] for (j, i) in arcs if i == node)
                == 1
          )
        elif node == sink:
            prob += (
                pulp.lpSum(f[i, j] for (i, j) in arcs if i == node)
                - pulp.lpSum(f[j, i] for (j, i) in arcs if i == node)
                == -1
            )
        else:
            prob += (
                pulp.lpSum(f[i, j] for (i, j) in arcs if i == node )
                - pulp.lpSum(f[j, i] for (j, i) in arcs if i == node)
                == 0
            )

    # for (i, j) in arcs:
    #   prob += (
    #            pulp.lpSum(f[i, j] for (i, j) in arcs)
    #            - pulp.lpSum(f[j, i] for (j, i) in arcs)
    #            >= -1
    #      )

    #   prob += (
    #            pulp.lpSum(f[i, j] for (i, j) in arcs)
    #            - pulp.lpSum(f[j, i] for (j, i) in arcs)
    #            <= 1
    #      )

    # Capacity constraints
    for (i, j) in arcs:
        prob += f[i, j] * b <= u_ij[i, j]
        prob += f[i, j] >= 0
        prob += f[i, j] <= 1

    # Solve problem
    prob.solve()

    print("Solver Status:", pulp.LpStatus[prob.status])
    print(prob)

    # Debug if infeasible
    if prob.status == -1:  # Infeasible
        print("\nDebugging Constraints:")

        # Flow conservation constraints
        print("\nFlow Conservation Constraints:")
        # for node in nodes:
        #     inflow = pulp.lpSum(f[i, j].varValue for (i, j) in arcs if i == node)
        #     outflow = pulp.lpSum(f[j, i].varValue for (j, i) in arcs if i == node)
        #     if node == source:
        #         print(f"Node {node} (Source): Outflow - Inflow = {outflow} - {inflow} (should equal {1})")
        #     elif node == sink:
        #         print(f"Node {node} (Sink): Outflow - Inflow = {outflow} - {inflow} (should equal {-1})")
        #     else:
        #         print(f"Node {node}: Outflow - Inflow = {outflow} - {inflow} (should equal 0)")

        for src, sink in src_dest:  # Ensure source and sink are defined
          for node in nodes:
            inflow = pulp.lpSum(f[j, i].varValue for (j, i) in arcs if i == node)
            outflow = pulp.lpSum(f[i, j].varValue for (i, j) in arcs if i == node)
            balance = outflow - inflow  # Flow conservation equation

            if node == src:
                print(f"Node {node} (Source): Outflow - Inflow = {balance} (should equal {1})")
            elif node == sink:
                print(f"Node {node} (Sink): Outflow - Inflow = {balance} (should equal {-1})")
            else:
                print(f"Node {node}: Outflow - Inflow = {balance} (should equal 0)")

        # Capacity constraints
        print("\nCapacity Constraints:")
        for (i, j) in arcs:
            lhs = f[i, j].varValue * b
            rhs = u_ij[i, j]
            if lhs is not None:  # Only check if the variable has a value
                print(f"Arc ({i}, {j}): Flow = {lhs}, Capacity = {rhs} (Flow <= Capacity)")
            else:
                print(f"Arc ({i}, {j}): No value assigned (possible issue)")

        # Variable bounds
        print("\nVariable Bounds:")
        for (i, j) in arcs:
            value = f[i, j].varValue
            if value is not None:  # Only check if the variable has a value
                if value < 0 or value > b:
                    print(f"Variable Flow[{i}, {j}] = {value} (out of bounds [0, {b}])")
                else:
                    print(f"Variable Flow[{i}, {j}] = {value} (within bounds)")
            else:
              print(f"Variable Flow[{i}, {j}] has no value assigned (possible issue)")
        
        print("-----------------------------------------------------------------------------\n")
      
    else:
        # Output results if feasible
        print("Objective Value:", pulp.value(prob.objective))
        for (i, j) in arcs:
            print(f"Flow on arc ({i}, {j}):", f[i, j].varValue)

        print("-----------------------------------------------------------------------------\n")
      
        return pulp.value(prob.objective), [f[i, j].varValue for (i,j) in arcs]

def Solution_Zlb(points, selected_edges, stc, stic, ec, lambda_n, speed, capacity, demand, alpha, beta, source, destination, src_dest):
  print(selected_edges)
  lr1_result, yij = LR_1(points, selected_edges, stc, stic, ec, lambda_n, alpha)
  lr2_result, fij = LR_2(points, selected_edges, stc, stic, ec, lambda_n, beta, speed, capacity, demand, source, destination, src_dest)
  print("LR1 Solution:", lr1_result)
  print("LR2 Solution:", lr2_result)
  print("Fij:", fij)
  print("Yij:", yij)

  result = lr1_result + lr2_result
  print("Total Solution:", result)

  return result, fij, yij