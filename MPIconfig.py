Single_Agent = 1

if Single_Agent:
    rank = 1
else:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    rank = comm.rank
    size = comm.size
    name = MPI.Get_processor_name()
