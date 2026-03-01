##check if your mpi environment is correctly setup
import numpy as np
import unittest
from NEDAS.utils.parallel import Comm, distribute_tasks

class TestParallel(unittest.TestCase):

    def setUp(self):
        self.comm = Comm()
        self.pid = self.comm.Get_rank()
        self.nproc = self.comm.Get_size()

    def test_parallel_start(self):
        self.assertIsInstance(self.pid, int)
        self.assertIsInstance(self.nproc, int)
        self.assertTrue(self.pid < self.nproc)

    def test_bcast(self):
        if self.pid == 0:
            data0 = 100
            data1 = np.array([1, 2, 3])
            data2 = {'x':1, 'y':2}
            data3 = [1, True]
        else:
            data0 = None
            data1 = None
            data2 = None
            data3 = None
        data0 = self.comm.bcast(data0)
        data1 = self.comm.bcast(data1)
        data2 = self.comm.bcast(data2)
        data3 = self.comm.bcast(data3)

        self.assertEqual(data0, 100)
        self.assertTrue((data1==np.array([1,2,3])).all())
        self.assertEqual(data2['x'], 1)
        self.assertEqual(data2['y'], 2)
        self.assertEqual(data3[0], 1)
        self.assertTrue(data3[1])

    def test_send_recv(self):
        if self.pid == 0:
            data = np.array([1, 2, 3])
            self.comm.send(data, dest=self.nproc-1, tag=0)
        if self.pid == self.nproc-1:
            recv_data = self.comm.recv(source=0, tag=0)
            self.assertTrue((recv_data == np.array([1, 2, 3])).all())

    def test_allgather(self):
        data = [self.pid]

        gather_data = []
        for entry in self.comm.allgather(data):
            for value in entry:
                gather_data.append(value)

        if self.pid == 0:
            self.assertTrue((np.array(gather_data) == np.arange(self.nproc)).all())

    def test_distribute_tasks_case1(self):
        ##case1: one task per self.pid
        task_list = distribute_tasks(self.comm, np.arange(self.nproc))
        self.assertEqual(task_list[self.pid][0], self.pid)

    def test_distribute_tasks_case2(self):
        ##case2: more tasks than self.nproc
        ntasks = 10*self.nproc
        task_list = distribute_tasks(self.comm, np.arange(ntasks))
        full_task_list = []
        for p in range(self.nproc):
            for e in task_list[p]:
                full_task_list.append(e)
        self.assertTrue((np.array(full_task_list) == np.arange(ntasks)).all())

    def test_distribute_tasks_case3(self):
        ##case3: fewer tasks than self.nproc
        ntasks = np.max(1, self.nproc//10)
        task_list = distribute_tasks(self.comm, np.arange(ntasks))
        full_task_list = []
        for p in range(self.nproc):
            for e in task_list[p]:
                full_task_list.append(e)
        self.assertTrue((np.array(full_task_list) == np.arange(ntasks)).all())

    def test_distribute_tasks_case4(self):
        ##case4: uneven workload for 100 tasks
        ntasks = 100
        workload = np.random.randint(1, 10, ntasks)
        task_list = distribute_tasks(self.comm, np.arange(ntasks), workload)
        full_task_list = []
        for p in range(self.nproc):
            for e in task_list[p]:
                full_task_list.append(e)
        self.assertTrue((np.array(full_task_list) == np.arange(ntasks)).all())

if __name__ == '__main__':
    unittest.main()

