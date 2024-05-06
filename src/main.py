import sys

from argparser import create_parser

from task1 import (
    QueuelessSystemSolver,
    # LimitedQueueSystemSolver,
    LimitlessQueueSystemSolver,
    LimitlessQueueSystemWithDisbandSolver,
)
from task2 import MultipleSetupersSystemSolver
from task3 import Simulation

from data import (
    data1,
    data2,
    data3,
)

if __name__ == "__main__":
    
    parser = create_parser()
    namespace = parser.parse_args(sys.argv[1:])
    
    task_num = namespace.task
    
    match(task_num):
        case '1':
            obj11_solver = QueuelessSystemSolver(**data1)
            obj11_solver.solve_and_plot(interp_model_kind='cubic', figsize=(15, 20))
            
            # obj12_solver = LimitedQueueSystemSolver(**data1)
            # obj12_solver.solve_and_plot()
            
            obj13_solver = LimitlessQueueSystemSolver(**data1)
            obj13_solver.solve_and_plot(show_plot=False, fontsize=12)
            
            obj14_solver = LimitlessQueueSystemWithDisbandSolver(**data1)
            obj14_solver.solve_and_plot(show_plot=False, fontsize=12)
        case '2':
            obj21_solver = MultipleSetupersSystemSolver(**data2)
            obj21_solver.solve_and_plot()
        case '3':
            sim = Simulation(
                machine_num=data3['N'],
                worker_num=int(namespace.workers),
                prod_time=data3['prod_time'],
                setup_time=data3['setup_time'],
                failure_on_product=data3['failure_on_product'],
                logfile_path=data3['logfile_path'],
                sim_time=float(namespace.simtime),
            )
            sim()
    