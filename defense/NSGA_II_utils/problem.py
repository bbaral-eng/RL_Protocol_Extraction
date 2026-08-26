from pymoo.core.problem import ElementwiseProblem
import defense.NSGA_II_utils.genome as genome
import defense.NSGA_II_utils.rules as rules

# problem.py adjusts genome.py and rules.py into math that NSGA-II from pymoo can run. 

class DefenseProblem(ElementwiseProblem):
    def __init__(self, protocol):
        self.protocol = protocol
        lower, upper = genome.bounds(protocol)
        super().__init__(n_var=len(lower), n_obj=3, n_ieq_constr=0, xl=lower, xu=upper)

    def _evaluate(self, x, out, *args, **kwargs):

        schedule = genome.decode(x, self.protocol)
        count = rules.count_score(schedule)
        spread = rules.spread_score(schedule)
        attack_accuracy = rules.surrogate_attack_accuracy(schedule, self.protocol)

        # fitness score per objective, all already oriented for pymoo's minimize:
        # count minimizes between [0, 1] 
        # spread minimizes between [0, 1]  
        # attack_accuracy minimizes between [0, 1]  
        out['F'] = [count, spread, attack_accuracy]
