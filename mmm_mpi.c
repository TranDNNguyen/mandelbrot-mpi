
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define idx(N, M, i, j) (M)[(N)*(i) + (j)]
#define NSEC_PER_SEC 1000000000

unsigned int matrix_checksum(int N, void *M, unsigned int size);

static void init_matrices(int N, double *A, double *B)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            idx(N, A, i, j) = i + j;
            idx(N, B, i, j) = i + j * 2;
        }
    }
}


/*
 * Multiply a row of A by a single column of B.
 */
static void mult_row(int N, double *A, double *B, double *C, int row)
{
    for (int j = 0; j < N; j++)
        idx(N, C, row, j) = 0;

    for (int k = 0; k < N; k ++)
        for (int j = 0; j < N; j++)
            idx(N, C, row, j) += idx(N, A, row, k) * idx(N, B, k, j);
}


// Return time passed in seconds.
static double get_timespec_delta(const struct timespec *start,
        const struct timespec *stop)
{
    long long delta_nsec, start_nsec, stop_nsec;

    start_nsec = start->tv_sec * NSEC_PER_SEC + start->tv_nsec;
    stop_nsec = stop->tv_sec * NSEC_PER_SEC + stop->tv_nsec;
    delta_nsec = stop_nsec - start_nsec;

    return (double)delta_nsec / NSEC_PER_SEC;
}


// Row for this process to start on.
static int offset(int N, int processes, int rank)
{
    return rank * (N / processes);
}

// Number of rows in A this process handles.
static int amount(int N, int processes, int rank)
{
    int amnt = N / processes;

    // Handle leftovers.
    if (rank == processes - 1)
        amnt += N % processes;

    return amnt;
}


int main(int argc, char** argv) {
    int processes, rank, N;
    double *A;
    double *B;
    double *C;
    struct timespec start, stop;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        if (argc != 2) {
            printf("Usage: mmm_mpi N\n");
            return 1;
        } else {
            N = atoi(argv[1]);
        }

        if (N <= 0) {
            printf("Error: wrong matrix order (N > 0)\n");
            return 1;
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int amnt = amount(N, processes, rank);


    // Divide work based on rows of A. Send all needed information.
    if (rank == 0) {
        A = malloc(N*N*sizeof(double));
        B = malloc(N*N*sizeof(double));
        C = malloc(N*N*sizeof(double));

        clock_gettime(CLOCK_MONOTONIC, &start);

        init_matrices(N, A, B);

        // All processes need B in full.
        for (int i = 1; i < processes; i++)
            MPI_Send(B, N*N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);

        // Send appropriate A rows over.
        for (int i = 1; i < processes; i++) {
            int off_rem = offset(N, processes, i);
            int amnt_rem = amount(N, processes, i);
            MPI_Send(A + off_rem*N, amnt_rem*N, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        }
    } else {
        // Only allocate needed memory.
        A = malloc(amnt*N*sizeof(double));
        B = malloc(N*N*sizeof(double));
        C = malloc(amnt*N*sizeof(double));

        MPI_Recv(B, N*N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);

        // Load appropriate rows of A.
        MPI_Recv(A, amnt*N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
    }


    // Calculate for rows that this process handles.
    for (int i = 0; i < amnt; i++)
        mult_row(N, A, B, C, i);


    // Send answers back to master.
    if (rank > 0) {
        // Send appropriate rows of C.
        MPI_Send(C, amnt*N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    } else {
        // Receive appropriate C rows.
        for (int i = 1; i < processes; i++) {
            int off_rem = offset(N, processes, i);
            int amnt_rem = amount(N, processes, i);
            MPI_Recv(C + off_rem*N, amnt_rem*N, MPI_DOUBLE, i, 2,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        clock_gettime(CLOCK_MONOTONIC, &stop);

        printf("Running time: %.6f secs\n", get_timespec_delta(&start, &stop));

        printf("A: %u\n", matrix_checksum(N, A, sizeof(double)));
        printf("B: %u\n", matrix_checksum(N, B, sizeof(double)));
        printf("C: %u\n", matrix_checksum(N, C, sizeof(double)));
    }

    free(A);
    free(B);
    free(C);
    MPI_Finalize();
}
