##check if your mpi environment is correctly setup
import numpy as np
import unittest
import parallel

class TestParallel(unittest.TestCase):

    def test_parallel_start(self):
        comm = parallel.parallel_start()
        self.assertIsInstance(comm.Get_size(), int)
        self.assertIsInstance(comm.Get_rank(), int)
        self.assertTrue(comm.Get_rank() < comm.Get_size())

    def test_bcast(self):
        comm = parallel.parallel_start()

        if comm.Get_rank() == 0:
            data0 = 100
            data1 = np.array([1, 2, 3])
            data2 = {'x':1, 'y':2}
            data3 = [1, True]
        else:
            data0 = None
            data1 = None
            data2 = None
            data3 = None
        data0 = comm.bcast(data0)
        data1 = comm.bcast(data1)
        data2 = comm.bcast(data2)
        data3 = comm.bcast(data3)

        self.assertEqual(data0, 100)
        self.assertTrue((data1==np.array([1,2,3])).all())
        self.assertEqual(data2['x'], 1)
        self.assertEqual(data2['y'], 2)
        self.assertEqual(data3[0], 1)
        self.assertTrue(data3[1])

    def test_send_recv(self):
        comm = parallel.parallel_start()
        pid = comm.Get_rank()
        nproc = comm.Get_size()
        if pid == 0:
            data = np.array([1, 2, 3])
            comm.send(data, dest=nproc-1, tag=0)
        if pid == nproc-1:
            recv_data = comm.recv(source=0, tag=0)
            self.assertTrue((recv_data == np.array([1, 2, 3])).all())

    def test_allgather(self):
        comm = parallel.parallel_start()
        pid = comm.Get_rank()
        nproc = comm.Get_size()
        data = [pid]

        gather_data = []
        for entry in comm.allgather(data):
            for value in entry:
                gather_data.append(value)

        if pid == 0:
            self.assertTrue((np.array(gather_data) == np.arange(nproc)).all())


    def test_distribute_tasks(self):
        comm = parallel.parallel_start()
        pid = comm.Get_rank()
        nproc = comm.Get_size()

        ##case1: one task per pid
        task_list = parallel.distribute_tasks(comm, np.arange(nproc))
        self.assertEqual(task_list[pid][0], pid)

        ##case2: more tasks than nproc
        ntasks = 10*nproc
        task_list = parallel.distribute_tasks(comm, np.arange(ntasks))
        full_task_list = []
        for p in range(nproc):
            for e in task_list[p]:
                full_task_list.append(e)
        self.assertTrue((np.array(full_task_list) == np.arange(ntasks)).all())

        ##case3: fewer tasks than nproc
        ntasks = np.max(1, nproc//10)
        task_list = parallel.distribute_tasks(comm, np.arange(ntasks))
        full_task_list = []
        for p in range(nproc):
            for e in task_list[p]:
                full_task_list.append(e)
        self.assertTrue((np.array(full_task_list) == np.arange(ntasks)).all())

        ##case4: uneven workload for 100 tasks
        ntasks = 100
        workload = np.random.randint(1, 10, ntasks)
        task_list = parallel.distribute_tasks(comm, np.arange(ntasks), workload)
        full_task_list = []
        for p in range(nproc):
            for e in task_list[p]:
                full_task_list.append(e)
        self.assertTrue((np.array(full_task_list) == np.arange(ntasks)).all())


if __name__ == '__main__':
    unittest.main()

