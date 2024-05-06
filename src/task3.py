from __future__ import annotations

from random import expovariate
from queue import Queue, PriorityQueue
from functools import reduce, total_ordering

from typing import Literal, Final, Any, Callable
from io import TextIOWrapper

def collect_statistics(func: Callable[..., None]):

    def decorator(self: Simulation, *args, **kwargs):
        
        # Average statistics
        
        self.avg_broken_machines += self.broken_machines_num * (self.current_time - self.prev_current_time) / self.sim_time
        self.avg_waiting_machines += self.waiting_machines_num * (self.current_time - self.prev_current_time) / self.sim_time
        self.avg_busy_workers += self.busy_workers_num * (self.current_time - self.prev_current_time) / self.sim_time

        func(self, *args, **kwargs)
        
        # Maximum statistics
        
        self.max_broken_machines = max(self.max_broken_machines, self.broken_machines_num)
        self.max_waiting_machines = max(self.max_waiting_machines, self.waiting_machines_num)
        self.max_busy_workers = max(self.max_busy_workers, self.busy_workers_num)
        
    return decorator

def logging_required(type: Literal['CEC', 'FEC']):

    def decorator(func: Callable[..., None]):

        def wrapper(self: Simulation, *args, **kwargs):
            func(self, *args, **kwargs)
            
            if type == "FEC":
                self.log_event(*map(lambda event: f'{type}: {event}', sorted(self.events.queue, key=lambda event: event.time)), '\n')
            else:
                kwarg_machine = kwargs.get('machine', None)
                machine: Machine = args[0] if kwarg_machine is None else kwarg_machine
                
                if machine.state == 'producing' and machine.produced_num % self.num_to_failure == 0:
                    event_type = 'recovery'
                elif machine.state == 'producing' and machine.produced_num % self.num_to_failure != 0:
                    event_type = 'produced'
                elif machine.state == 'setting' or machine.state == 'waiting':
                    event_type = 'failure'
                else:
                    raise NotImplementedError
                    
                self.log_event(f'New event!\n{type}: time = {self.current_time}, type = {event_type}, machine_id = {machine.id}\n\n')

        return wrapper

    return decorator
@total_ordering
class Event:
    
    def __init__(self, time: float, type: Literal['produced', 'recovery', 'failure'], machine: Machine) -> None:
        self.time: float = time
        self.type: Literal['produced', 'recovery', 'failure'] = type
        self.machine: Machine = machine
        
    def __lt__(self, other: Event) -> bool:
        return self.time < other.time
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Event):
            raise NotImplementedError
        
        return self.time == other.time
    
    def __repr__(self) -> str:
        return f'time = {self.time}, type = {self.type}, machine_id = {self.machine.id}\n'
    
class Machine:
    
    def __init__(self, id: int, init_state: Literal['producing', 'waiting', 'setting'] = 'producing') -> None:
        self.id: int = id
        self.state: Literal['producing', 'waiting', 'setting'] = init_state
        self.produced_num: int = 0
        
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: Any) -> bool:
        
        if not isinstance(other, type(self)):
            raise NotImplementedError
        
        return self.id == other.id
    
    def __repr__(self) -> str:
        return f'Machine #{self.id}, {self.state}, completed {self.produced_num} products\n'
        
class Simulation:
    
    def __init__(self, machine_num: int,
                 worker_num: int,
                 prod_time: float,
                 setup_time: float,
                 failure_on_product: int,
                 logfile_path: str,
                 sim_time: float) -> None:
        
        # Constants
        
        self.machine_num: Final[int] = machine_num
        self.worker_num: Final[int] = worker_num
        self.num_to_failure: Final[int] = failure_on_product
        self.setup_time: Final[float] = setup_time
        self.prod_time: Final[float] = prod_time
        self.sim_time: Final[float] = sim_time
        self.logfile_path: Final[str] = logfile_path
        
        # Simulation info
        
        self.current_time: float = 0.0
        self.prev_current_time: float = 0.0
        
        # Containers
        
        self.events: PriorityQueue[Event] = PriorityQueue()
        self.machines: set[Machine] = {Machine(id) for id in range(machine_num)}
        self.recovery_queue: Queue = Queue()
        
        # Statistics
        
        self.broken_machines_num: int = 0
        self.waiting_machines_num: int = 0
        self.busy_workers_num: int = 0
        
        self.avg_broken_machines: float = 0.0
        self.avg_waiting_machines: float = 0.0
        self.avg_busy_workers: float = 0.0
        
        self.max_broken_machines: int = 0
        self.max_waiting_machines: int = 0
        self.max_busy_workers: int = 0
        
        self._init()

    def __del__(self) -> None:
        self.logfile.close()

    def __call__(self, *args, **kwargs) -> None:
        return self._run(*args, **kwargs)
    
    @logging_required('FEC')
    @collect_statistics
    def loop(self) -> None:
        
        event = self.events.get()
        
        self.prev_current_time = self.current_time
        self.current_time = event.time
        
        match(event.type):
            case 'produced':
                self._handle_prod(event.machine)
            case 'recovery':
                self._handle_recovery(event.machine)
            case 'failure':
                self._handle_failure(event.machine)
        

    def _run(self, *args, **kwargs) -> None:
        
        while self.current_time < self.sim_time:
            self.loop()

        self.print_statistics()

    def log_event(self, *args) -> None:
        self.logfile.write(reduce(lambda i, j: str(i) + str(j), args))

    def print_statistics(self) -> None:
        max_products_produced = max(map(lambda machine: machine.produced_num, self.machines))
        
        print(
            f"""Среднее число простаивающих станков: {self.avg_broken_machines}
            \rМаксимальное число простаивающих станков: {self.max_broken_machines}
            \rСреднее число ожидающих обслуживания станков: {self.avg_waiting_machines}
            \rМаксимальное число ожидающих обслуживания станков: {self.max_waiting_machines}
            \rСреднее число занятых наладчиков: {self.avg_busy_workers}
            \rМаксимальное число занятых наладчиков: {self.max_busy_workers}
            \rМаксимальное число произведенных деталей: {max_products_produced}
            \rВремя: {self.current_time}"""
        )
            
    @staticmethod    
    def puasson_distribution(mean: float) -> float:
        return expovariate(1 / mean)

    def _plan_prod(self, machine: Machine) -> None:
        self.events.put(Event(self.current_time + self.puasson_distribution(self.prod_time), 'produced', machine))

    def _plan_recovery(self, machine: Machine) -> None:
        self.events.put(Event(self.current_time + self.puasson_distribution(self.prod_time), 'recovery', machine))
    
    def _plan_failure(self, machine: Machine) -> None:
        self.events.put(Event(self.current_time, 'failure', machine))

    @logging_required('CEC')
    def _handle_prod(self, machine: Machine) -> None:
        machine.produced_num += 1    
        
        if machine.produced_num % self.num_to_failure == 0:
            self._plan_failure(machine)
        else:
            self._plan_prod(machine)

    @logging_required('CEC')
    def _handle_failure(self, machine: Machine) -> None:
        self.broken_machines_num += 1
        machine.state = 'waiting'
        
        if self.busy_workers_num < self.worker_num:
            self._plan_recovery(machine)
            machine.state = 'setting'
            self.busy_workers_num += 1
        else:
            self.recovery_queue.put(machine)
            self.waiting_machines_num += 1
    
    @logging_required('CEC')
    def _handle_recovery(self, machine: Machine) -> None:
        self.busy_workers_num -= 1
        self.broken_machines_num -= 1
        machine.state = 'producing'
        self._plan_prod(machine)
        
        if not self.recovery_queue.empty() and self.busy_workers_num < self.worker_num:
            queued_machine = self.recovery_queue.get()
            self._plan_recovery(queued_machine)
            self.busy_workers_num += 1   
            self.waiting_machines_num -= 1
        
    def _init(self) -> None:
        self.logfile: TextIOWrapper = open(self.logfile_path, 'w')
        
        for machine in self.machines:
            match(machine.state):
                case 'producing':
                    self._plan_prod(machine)
                case 'setting':
                    self.broken_machines_num += 1
                    self.busy_workers_num += 1
                    self._plan_recovery(machine)
                case 'waiting':
                    self.broken_machines_num += 1
                    
                    if self.busy_workers_num < self.worker_num:
                        self._plan_recovery(machine)
                        machine.state = 'setting'
                        self.busy_workers_num += 1
                    else:
                        self.recovery_queue.put(machine)
                        self.waiting_machines_num += 1