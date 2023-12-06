#FC = pgf90
#CC = pgcc
FC = gfortran
CC = gcc
FFLAGS = 
CFLAGS = -D_UNDERSCORE
LDFLAGS = 


all: rd_wr_binary.exe

clean:
	rm -f *.o rd_wr_binary.exe

rd_wr_binary.exe: rd_wr_binary.o read_geogrid.o write_geogrid.o
	$(FC) $(LDFLAGS) -o rd_wr_binary.exe rd_wr_binary.o read_geogrid.o write_geogrid.o

#rd_wr_binary.exe: rd_wr_binary.o write_geogrid.o
#	$(FC) $(LDFLAGS) -o rd_wr_binary.exe rd_wr_binary.o write_geogrid.o

rd_wr_binary.o: rd_wr_binary.f90
	$(FC) -c $(FFLAGS) rd_wr_binary.f90

read_geogrid.o: read_geogrid.c
	$(CC) -c $(CFLAGS) read_geogrid.c

write_geogrid.o: write_geogrid.c
	$(CC) -c $(CFLAGS) write_geogrid.c
