#include <mpi.h>
#include <iostream>

void mpi_basic_info(int rank, int world_size)
{
    std::cout << "\n# mpi_basic_info" << std::endl;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    std::cout << "[mpi_basic_info()-rank-" << rank << "] world_size: " << world_size << std::endl;
    std::cout << "[mpi_basic_info()-rank-" << rank << "] processor_name: " << processor_name << std::endl;
    std::cout << "[mpi_basic_info()-rank-" << rank << "] name_len: " << name_len << std::endl;
}

void mpi_send_and_receive(int rank, int world_size)
{
    std::cout << "\n# mpi_send_and_receive" << std::endl;
    if (world_size < 2) //need at least 2 processes to continue
    {
        // MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    if (rank == 0)
    {
        int tmp0 = 233;
        std::cout << "[mpi_send_and_receive()-rank-" << rank << "] send " << tmp0 << " to rank-1" << std::endl;
        // data, count, datatype, destination-rank, tag, communicator
        MPI_Send(&tmp0, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1)
    {
        int tmp1;
        // data, count, datatype, source-rank, tag, communicator, status
        MPI_Recv(&tmp1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "[mpi_send_and_receive()-rank-" << rank << "] receive " << tmp1 << " from rank-0" << std::endl;
    }
}

void mpi_ring(int rank, int world_size)
{
    std::cout << "\n# mpi_ring" << std::endl;
    if (world_size < 2) //need at least 2 processes to continue
    {
        return;
    }
    int token = 233;
    if (rank != 0)
    {
        MPI_Recv(&token, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "[mpi_ring()-rank-" << rank << "] receive " << token << " from rank-" << rank-1 << std::endl;
        token = token + 3;
    }
    MPI_Send(&token, 1, MPI_INT, (rank + 1) % world_size, 0, MPI_COMM_WORLD);
    std::cout << "[mpi_ring()-rank-" << rank << "] send " << token << " to rank-" << (rank+1)%world_size << std::endl;
    if (rank == 0)
    {
        MPI_Recv(&token, 1, MPI_INT, world_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "[mpi_ring()-rank-" << rank << "] receive " << token << " from rank-" << world_size-1 << std::endl;
    }
}

// mpicxx -o tbd00.exe draft00.cpp
// mpirun -n 3 ./tbd00.exe
// TODO sync all process after each example
// TODO if sync the process between examples, no need to modify the tag
int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);
    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    mpi_basic_info(rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);

    mpi_send_and_receive(rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);

    mpi_ring(rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
}
