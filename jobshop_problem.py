from qubots.base_problem import BaseProblem
import random
import os

def read_instance(filename):
    """
    Reads an instance file in the Taillard format.
    
    The file format is as follows:
      - First line: (header information; often ignored)
      - Second line: contains five numbers: number of jobs, number of machines, seed, upper bound, lower bound.
      - Next nb_jobs lines: processing times for each job (given in processing order).
      - Next nb_jobs lines: machine order for each job (an ordered list of visited machines, 1-indexed).
    
    The processing times are re-ordered so that for each job j and each machine m (0 ≤ m < nb_machines),
    processing_time[j][m] is the processing time of job j’s activity that is executed on machine m.
    
    Returns:
      nb_jobs, nb_machines, processing_time, machine_order, max_start
    """

    # Resolve relative path with respect to this module’s directory.
    if not os.path.isabs(filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_dir, filename)

    with open(filename) as f:
        lines = f.readlines()
    
    # Assume the first line is a header; the second line contains key numbers.
    # (Adjust indexing as needed if your instance files differ slightly.)
    first_line = lines[1].split()
    nb_jobs = int(first_line[0])
    nb_machines = int(first_line[1])
    
    # Read processing times (in processing order) for each job.
    processing_times_in_order = []
    for i in range(3, 3 + nb_jobs):
        parts = lines[i].split()
        times = [int(parts[j]) for j in range(nb_machines)]
        processing_times_in_order.append(times)
    
    # Read machine order for each job.
    machine_order = []
    for i in range(4 + nb_jobs, 4 + 2 * nb_jobs):
        parts = lines[i].split()
        # Convert to 0-indexed.
        order = [int(parts[j]) - 1 for j in range(nb_machines)]
        machine_order.append(order)
    
    # Reorder processing times: For each job j and each machine m, find the position
    # of m in machine_order[j] and use that index in processing_times_in_order[j].
    processing_time = []
    for j in range(nb_jobs):
        proc_times = []
        for m in range(nb_machines):
            pos = machine_order[j].index(m)
            proc_times.append(processing_times_in_order[j][pos])
        processing_time.append(proc_times)
    
    # Trivial upper bound: sum of all processing times (across all jobs)
    max_start = sum(sum(processing_time[j]) for j in range(nb_jobs))
    return nb_jobs, nb_machines, processing_time, machine_order, max_start

class JobShopProblem(BaseProblem):
    """
    Job Shop Scheduling Problem (Taillard format) for Qubots.
    
    A set of jobs must be processed on every machine of the shop. Each job consists of an ordered
    sequence of tasks (activities). For each job j, the processing order is given by machine_order[j]
    (a permutation of machine indices). Each task has a specified processing time.
    
    In addition to the precedence constraints (each job’s activities must follow its specified order),
    disjunctive resource constraints enforce that each machine processes only one activity at a time.
    
    **Candidate Solution Representation:**
      A dictionary with a key "jobs_order" mapping to a list of length nb_machines.
      Each element is a permutation (list) of job indices (0-indexed) representing the processing order
      on that machine.
    """
    
    def __init__(self, instance_file: str):
        (self.nb_jobs,
         self.nb_machines,
         self.processing_time,
         self.machine_order,
         self.max_start) = read_instance(instance_file)
    
    def evaluate_solution(self, solution) -> float:
        """
        Evaluates a candidate solution.
        
        Expects:
          solution: a dictionary with key "jobs_order" whose value is a list of length nb_machines.
                    Each element is a permutation of job indices (0-indexed) representing the job sequence
                    on that machine.
        
        The evaluation builds a directed acyclic graph (DAG) whose nodes are pairs (j, m) indicating the
        activity of job j on machine m. Two types of edges are added:
          - For each job j, if machine_order[j] = [m₀, m₁, …, m₍M-1₎], then add an edge from (j, m₀) to (j, m₁),
            from (j, m₁) to (j, m₂), etc.
          - For each machine m, if candidate["jobs_order"][m] = [j₁, j₂, …, jₙ], then add an edge from (j₁, m)
            to (j₂, m), from (j₂, m) to (j₃, m), etc.
        
        Each node’s processing time is given by processing_time[j][m]. Then the finish time of a node is the
        maximum finish time of its predecessors plus its processing time. The makespan is defined as the maximum
        finish time among the last activities of each job (i.e. for each job j, the node corresponding to the
        last machine in machine_order[j]).
        
        If the graph contains a cycle (i.e. the candidate violates constraints), a high penalty is returned.
        """
        penalty = 1e9
        # Check candidate solution structure.
        if not isinstance(solution, dict) or "jobs_order" not in solution:
            return penalty
        candidate = solution["jobs_order"]
        if not isinstance(candidate, list) or len(candidate) != self.nb_machines:
            return penalty
        for m in range(self.nb_machines):
            if sorted(candidate[m]) != list(range(self.nb_jobs)):
                return penalty
        
        # Build the set of nodes.
        nodes = [(j, m) for j in range(self.nb_jobs) for m in range(self.nb_machines)]
        
        # Initialize graph and in-degree.
        graph = {node: [] for node in nodes}
        in_degree = {node: 0 for node in nodes}
        
        # Add job precedence edges.
        for j in range(self.nb_jobs):
            order_j = self.machine_order[j]  # list of machine indices in the order for job j
            for k in range(len(order_j) - 1):
                u = (j, order_j[k])
                v = (j, order_j[k + 1])
                graph[u].append(v)
                in_degree[v] += 1
        
        # Add machine order edges (from candidate solution).
        for m in range(self.nb_machines):
            order_m = candidate[m]  # candidate order for machine m
            for i in range(len(order_m) - 1):
                j1 = order_m[i]
                j2 = order_m[i + 1]
                u = (j1, m)
                v = (j2, m)
                graph[u].append(v)
                in_degree[v] += 1
        
        # Perform topological sort and compute finish times.
        finish_time = {node: 0 for node in nodes}
        # For nodes with in-degree 0, set finish_time = processing_time of that node.
        queue = []
        for node in nodes:
            if in_degree[node] == 0:
                j, m = node
                finish_time[node] = self.processing_time[j][m]
                queue.append(node)
        
        processed = 0
        while queue:
            u = queue.pop(0)
            processed += 1
            for v in graph[u]:
                j_v, m_v = v
                # Update finish_time[v] = max( finish_time[v], finish_time[u] + processing_time of v )
                finish_time[v] = max(finish_time[v], finish_time[u] + self.processing_time[j_v][m_v])
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        # If not all nodes were processed, there is a cycle.
        if processed != len(nodes):
            return penalty
        
        # For each job, determine the finish time of its last activity.
        job_completion = []
        for j in range(self.nb_jobs):
            # The last activity of job j is on machine m_last = last element of machine_order[j]
            m_last = self.machine_order[j][-1]
            job_completion.append(finish_time[(j, m_last)])
        makespan = max(job_completion)
        return makespan
    
    def random_solution(self):
        """
        Generates a random candidate solution.
        
        Returns a dictionary with key "jobs_order" mapping to a list of length nb_machines.
        Each element is a random permutation of job indices.
        """
        candidate = []
        for m in range(self.nb_machines):
            order = list(range(self.nb_jobs))
            random.shuffle(order)
            candidate.append(order)
        return {"jobs_order": candidate}
