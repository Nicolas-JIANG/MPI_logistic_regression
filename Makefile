CC		= mpic++
TARGET	= mpi_admm
CFLAGS	= -std=c++11
DLIB_FLAGS	= -lpthread -lX11
DLIB	= $(HOME)/opt/dlib
SOURCE	= $(DLIB)/dlib/all/source.cpp

all:	$(TARGEET)

$(TARGET):	mpi_admm.cpp admm.h
	$(CC) $(CFLAGS) -I$(DLIB) $(SOURCE) $(DLIB_FLAGS) -o $(TARGET) admm.h mpi_admm.cpp
