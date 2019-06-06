PROGS=main

all : $(PROGS)

main : main.cu
	nvcc -o $@ $^
clean:
	rm -f $(PROGS)