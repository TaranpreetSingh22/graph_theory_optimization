import pulp
import random

def solution_1a(points, stc, stic, ec,  result_of_cost, connected_edges, speed, capacity, b, alpha, beta, source, destination, src_dest):
  print(f"\n----------------------in solution 1a ZUB----------------\n")
  # Define the LP problem
  lp = pulp.LpProblem("Flow_Optimization", pulp.LpMinimize)

  # Define nodes and arcs
  nodes = [i for i in range(len(points))]
  arcs = connected_edges
  m1 = len(nodes) - 1
  m2 = len(arcs)
  print(arcs)

  station_cost = stc
  steiner_cost = stic
  edge_cost = ec

  # Parameters
  time = {(i, j): ((((points[i][0] - points[j][0])**2) + ((points[i][1] - points[j][1])**2))**0.5)/speed for (i, j) in arcs}  # t_ij
  cost = {(i, j): edge_cost * ((((points[i][0] - points[j][0])**2) + ((points[i][1] - points[j][1])**2))**0.5) for (i, j) in arcs}
  capacity = {arc: capacity for arc in arcs} # u_ij
  demand = b  # b^k
  source = source # p(k)
  sink = destination # q(k)
  

  print(f"nodes : {nodes}")
  print(f"arcs : {arcs}")
  print(f"time : {time}")
  print(f"cost : {cost}")
  print(f"demand : {demand}")
  print(f"total_cost : {result_of_cost}")
  print(f"alpha : {alpha}")
  print(f"beta : {beta}")

  # Variables: flow on each arc
  flow = pulp.LpVariable.dicts("Flow", arcs, lowBound=0, upBound=demand, cat="Continuous")

  # Binary decision variables
  y = pulp.LpVariable.dicts("y", arcs, cat="Binary")

  # Objective function: Minimize cost
  lp += (alpha * pulp.lpSum(cost[i, j] * y[i, j] for (i, j) in arcs)) + (beta * pulp.lpSum(flow[i, j] * time[i, j] for (i, j) in arcs))

  # Constraints
  # # Flow conservation
  # for node in nodes:
  #     if node in source:
  #         lp += (
  #             pulp.lpSum(flow[i, j] for (i, j) in arcs if i == node)
  #             - pulp.lpSum(flow[j, i] for (j, i) in arcs if i == node)
  #             == demand
  #       )
  #     elif node in sink:
  #         lp += (
  #             pulp.lpSum(flow[i, j] for (i, j) in arcs if i == node)
  #             - pulp.lpSum(flow[j, i] for (j, i) in arcs if i == node)
  #             == -demand
  #         )
  #     else:
  #         lp += (
  #             pulp.lpSum(flow[i, j] for (i, j) in arcs if i == node)
  #             - pulp.lpSum(flow[j, i] for (j, i) in arcs if i == node)
  #             == 0
  #         )

    # Flow conservation
  for src, sink in src_dest:
      for node in nodes:
        if node == src:
            lp += (
                pulp.lpSum(flow[i, j] for (i, j) in arcs if i == node)
                - pulp.lpSum(flow[j, i] for (j, i) in arcs if i == node)
                == demand
          )
        elif node == sink:
            lp += (
                pulp.lpSum(flow[i, j] for (i, j) in arcs if i == node)
                - pulp.lpSum(flow[j, i] for (j, i) in arcs if i == node)
                == -demand
            )
        else:
            lp += (
                pulp.lpSum(flow[i, j] for (i, j) in arcs if i == node )
                - pulp.lpSum(flow[j, i] for (j, i) in arcs if i == node)
                == 0
            )

  # for (i, j) in arcs:
  #     lp += (
  #              pulp.lpSum(flow[i, j] for (i, j) in arcs)
  #              - pulp.lpSum(flow[j, i] for (j, i) in arcs)
  #              >= -1
  #        )

  #     lp += (
  #              pulp.lpSum(flow[i, j] for (i, j) in arcs)
  #              - pulp.lpSum(flow[j, i] for (j, i) in arcs)
  #              <= 1
  #        )

  # Capacity constraints
  for (i, j) in arcs:
      lp += flow[i, j] <= demand
      lp += flow[i, j] <= capacity[i, j]
      lp += flow[i, j] >= 0

  # Constraints
  lp += pulp.lpSum(y[i, j] for (i, j) in arcs) >= m1  # Lower bound
  lp += pulp.lpSum(y[i, j] for (i, j) in arcs) <= m2  # Upper bound # Constraint (1e)

  # Solve the problem
  lp.solve()
  status = lp.solve()
  print("Status:", pulp.LpStatus[lp.status])

  # Debug if infeasible
  if lp.status == -1:  # Infeasible
      print("\nDebugging Constraints:")

      # Flow conservation constraints
      # print("\nFlow Conservation Constraints:")
      # for node in nodes:
      #     inflow = pulp.lpSum(flow[i, j].varValue for (i, j) in arcs if i == node)
      #     outflow = pulp.lpSum(flow[j, i].varValue for (j, i) in arcs if i == node)
      #     if node == source:
      #         print(f"Node {node} (Source): Outflow - Inflow = {outflow} - {inflow} (should equal {demand})")
      #     elif node == sink:
      #         print(f"Node {node} (Sink): Outflow - Inflow = {outflow} - {inflow} (should equal {-demand})")
      #     else:
      #         print(f"Node {node}: Outflow - Inflow = {outflow} - {inflow} (should equal 0)")

      print("\nFlow Conservation Constraints:")
      for src, sink in src_dest:  # Ensure source and sink are defined
        for node in nodes:
          inflow = pulp.lpSum(flow[j, i].varValue for (j, i) in arcs if i == node)
          outflow = pulp.lpSum(flow[i, j].varValue for (i, j) in arcs if i == node)
          balance = outflow - inflow  # Flow conservation equation

          if node == src:
              print(f"Node {node} (Source): Outflow - Inflow = {balance} (should equal {demand})")
          elif node == sink:
              print(f"Node {node} (Sink): Outflow - Inflow = {balance} (should equal {-demand})")
          else:
              print(f"Node {node}: Outflow - Inflow = {balance} (should equal 0)")

      # Capacity constraints
      print("\nCapacity Constraints:")
      for (i, j) in arcs:
          lhs = flow[i, j].varValue
          rhs = capacity[i, j]
          if lhs is not None:  # Only check if the variable has a value
              print(f"Arc ({i}, {j}): Flow = {lhs}, Capacity = {rhs} (Flow <= Capacity)")
          else:
              print(f"Arc ({i}, {j}): No value assigned (possible issue)")

      # Variable bounds
      print("\nVariable Bounds:")
      for (i, j) in arcs:
          value = flow[i, j].varValue
          if value is not None:  # Only check if the variable has a value
              if value < 0 or value > demand:
                  print(f"Variable Flow[{i}, {j}] = {value} (out of bounds [0, {demand}])")
              else:
                  print(f"Variable Flow[{i}, {j}] = {value} (within bounds)")
          else:
            print(f"Variable Flow[{i}, {j}] has no value assigned (possible issue)")
          
      print("\n-----------------------------------------------------------\n")
    
  else:
      # Output results if feasible
      print("Objective Value:", pulp.value(lp.objective))
      for (i, j) in arcs:
          print(f"Flow on arc ({i}, {j}):", flow[i, j].varValue)
      
      print("\n-----------------------------------------------------------\n")

      return pulp.value(lp.objective)