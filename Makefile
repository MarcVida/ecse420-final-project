########################################################################
####################### Makefile Template ##############################
########################################################################

# Compiler settings
CC = nvcc
CFLAGS = -Xlinker /NOIMPLIB -Xlinker /NOEXP # NOIMPLIB removes the .lib file and NOEXP removes the .exp file
LDFLAGS = # Flags for the linker

# Makefile settings
TARGET = out
EXT = .cu
SRCDIR = src
OBJDIR = obj

# Source and object files
SRC = $(wildcard $(SRCDIR)/*.cu)
OBJ = $(SRC:$(SRCDIR)/%$(EXT)=$(OBJDIR)/%.obj)

# UNIX-based OS variables & settings
RM = rm
DELOBJ = $(OBJ)

# Windows OS variables & settings
DEL = del
EXE = .exe
WDELOBJ = $(SRC:$(SRCDIR)/%$(EXT)=$(OBJDIR)\\%.obj)

########################################################################
####################### Targets beginning here #########################
########################################################################

all: $(TARGET)

# Builds the app
$(TARGET): $(OBJ) | $(OBJDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Creating object directory
$(OBJDIR):
	if not exist $@ mkdir $@

# Building rule for .o files and its .cu
$(OBJDIR)/%.obj: $(SRCDIR)/%$(EXT)
	$(CC) $(CFLAGS) -o $@ -c $<

################### Cleaning rules for Unix-based OS ###################
# Cleans complete project
.PHONY: cleanunix
cleanunix:
	$(RM) $(DELOBJ) $(TARGET)

#################### Cleaning rules for Windows OS #####################
# Cleans complete project
.PHONY: clean
clean:
	$(DEL) $(WDELOBJ) $(TARGET)$(EXE)
