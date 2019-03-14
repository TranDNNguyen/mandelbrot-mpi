CC := mpicc
CFLAGS := -Wall -Wpedantic -Werror -O2 -lm -std=c11

all: mmm_mpi mandelbrot_mpi

common_objs		:= matrix_checksum.o
mmm_mpi_objs	:= mmm_mpi.o
mandelbrot_mpi_objs	:= mandelbrot_mpi.o

DEPFLAGS = -MMD -MF $(@:.o=.d)
deps := $(patsubst %.o,%.d,$(common_objs))
deps += $(patsubst %.o,%.d,$(mmm_mpi_objs))
deps += $(patsubst %.o,%.d,$(mandelbrot_mpi_objs))
-include $(deps)

mmm_mpi: $(common_objs) $(mmm_mpi_objs)
	@echo "LD	$@"
	@$(CC) $^ -o $@ $(CFLAGS)

mandelbrot_mpi: $(common_objs) $(mandelbrot_mpi_objs)
	@echo "LD	$@"
	@$(CC) $^ -o $@ $(CFLAGS)

%.o: %.c
	@echo "CC	$@"
	@$(CC) -c -o $@ $< $(DEPFLAGS) $(CFLAGS)

clean:
	@echo "CLEAN"
	@rm -f $(common_objs) $(deps) $(mmm_mpi_objs) $(mandelbrot_mpi_objs)
	@rm -f mmm_mpi mandelbrot_mpi

.PHONY: all clean

