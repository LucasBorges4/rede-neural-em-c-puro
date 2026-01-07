CC = gcc
CFLAGS = -Wall -Wextra -O2 -lm

TARGET = main
SRC = main.c neural-network.c activation-function.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(CFLAGS)

# regra genérica para compilar qualquer .c em .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
