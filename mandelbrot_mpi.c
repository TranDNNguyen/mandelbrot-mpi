
#include <complex.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <linux/limits.h>


#define CENTER_MIN -10.0
#define CENTER_MAX 10.0
#define ZOOM_MIN 0
#define ZOOM_MAX 100
#define CUTOFF_MIN 50
#define CUTOFF_MAX 1000

#define BOUND_MAGN 2.0
#define SIZE 1024

#define idx(N, M, i, j) (M)[(N)*(i) + (j)]
#define NSEC_PER_SEC 1000000000

unsigned int matrix_checksum(int N, void *M, unsigned int size);


/*
 * Compute the number of iterations for a single imaginary number.
 */
static int compute(double real, double imag, double cutoff)
{
    double complex c = real + imag*I;
    double complex z = 0;

    for (int i = 0; i < cutoff; i++) {
        if (cabs(z) > BOUND_MAGN)
            return i;

        z = z*z + c;
    }

    // Assume cutoff value is pure white.
    return cutoff;
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


int main(int argc, char** argv) {
    int processes, rank, zoom, cutoff;
    double xcenter, ycenter;
    int *graph = NULL;
    struct timespec start, stop;
    double step;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        if (processes < 2) {
            printf("Error: not enough tasks\n");
            return 1;
        }


        if (argc != 5) {
            printf("Usage: mandlebrot_mpi xcenter ycenter zoom cutoff.\n");
            return 1;
        } else {
            xcenter = atof(argv[1]);
            ycenter = atof(argv[2]);
            zoom = atoi(argv[3]);
            cutoff = atoi(argv[4]);
        }

        if (xcenter < CENTER_MIN || xcenter > CENTER_MAX) {
            printf("Error: wrong x-center (%f <= N <= %f)\n", CENTER_MIN, CENTER_MAX);
            return 1;
        }

        if (ycenter < CENTER_MIN || ycenter > CENTER_MAX) {
            printf("Error: wrong y-center (%f <= N <= %f)\n", CENTER_MIN, CENTER_MAX);
            return 1;
        }

        if (zoom < ZOOM_MIN || zoom > ZOOM_MAX) {
            printf("Error: wrong zoom (%d <= N <= %d)\n", ZOOM_MIN, ZOOM_MAX);
            return 1;
        }

        if (cutoff < CUTOFF_MIN || cutoff > CUTOFF_MAX) {
            printf("Error: wrong cutoff (%d <= N <= %d)\n", CUTOFF_MIN, CUTOFF_MAX);
            return 1;
        }
    }

    MPI_Bcast(&xcenter, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ycenter, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&zoom, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cutoff, 1, MPI_INT, 0, MPI_COMM_WORLD);
    step = 1.0/((float) (1 << zoom));

    if (rank == 0) {
        // Main process starts timer and allocates entire graph.
        graph = malloc(SIZE*SIZE*sizeof(int));
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Send which rows to use.
        for (int i = 0; i < SIZE; i++) {
            int p = i % (processes - 1) + 1;

            MPI_Send(&i, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
        }

        // Collect results.
        for (int i = 0; i < SIZE; i++) {
            int p = i % (processes - 1) + 1;

            MPI_Recv(graph + i*SIZE, SIZE, MPI_INT, p, 1, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
        }

        // Terminate subprocesses.
        for (int i = 1; i < processes; i++)
            MPI_Send(&i, 1, MPI_INT, i, 2, MPI_COMM_WORLD);

        clock_gettime(CLOCK_MONOTONIC, &stop);

        printf("Running time: %f secs\n", get_timespec_delta(&start, &stop));
        printf("M: %u\n", matrix_checksum(SIZE, graph, sizeof(int)));


        // Print the graph out to a pgm file.
        char file_name[PATH_MAX];
        sprintf(file_name, "mandel_%f_%f_%d_%d.pgm", xcenter, ycenter, zoom, cutoff);
        FILE *file = fopen(file_name, "w");

        fprintf(file, "P2\n%d %d\n%d\n", SIZE, SIZE, cutoff);

        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                fprintf(file, "%d ", idx(SIZE, graph, i, j));
            }

            fprintf(file, "\n");
        }

        fclose(file);
    } else {
        // Other processes allocates their part of the graph.
        int row;
        bool cont = true;
        MPI_Status status;

        graph = malloc(SIZE*sizeof(int));

        while(cont) {
            MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == 0) {
                // Tag 0 indicates we calculate row and continue.
                for (int i = 0; i < SIZE; i++) {
                    double x = xcenter - (SIZE/2 - row) * step;
                    double y = ycenter - (SIZE/2 - i) * step;
                    idx(SIZE, graph, 0, i) = compute(x, y, cutoff);
                }

                // Non-main processes send their values to master.
                MPI_Send(graph, SIZE, MPI_INT, 0, 1, MPI_COMM_WORLD);
            } else {
                // Got termination signal.
                cont = false;
            }
        }
    }

    free(graph);

    MPI_Finalize();
}

