CC = gcc
CFLAGS = -Wall -Werror -Wextra -O2 -march=native -std=c11 -D_GNU_SOURCE -I./include -I./onnxruntime/include
LDFLAGS = -lpthread -lrt -lm -lturbojpeg -lspng -lonnxruntime -L./onnxruntime/lib -Wl,-rpath,./onnxruntime/lib

SRCS = src/master_t.c src/spmc_workqueue.c src/worker_thread.c
OBJS = $(SRCS:.c=.o)
TARGET = face_server

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
