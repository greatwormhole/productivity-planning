import argparse

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', choices=['1', '2', '3'], default='1')
    parser.add_argument('-w', '--workers', default=10)
    parser.add_argument('-s', '--simtime', default=10000.0)
    
    return parser