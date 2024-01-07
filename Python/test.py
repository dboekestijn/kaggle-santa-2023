from multiprocessing import Pool


class Ex:

    def __init__(self, processes: int):
        self.processes = processes
        self.v = 2

    def worker(self, x: int) -> int:
        return x ** self.v + 1

    def get_results(self):
        with Pool(self.processes) as pool:
            return pool.map(self.worker, [i for i in range(self.processes)])


if __name__ == "__main__":
    ex = Ex(3)
    print(ex.get_results())
