# Configuration
F2PY = f2py
PYTHON = python3
MODULE_NAME = probit
WRAPPER_SRC = wrapper.f90

# Paths
DART_DIR = ./src
DART_INC = $(DART_DIR)
DART_LIB = $(DART_DIR)/libdart.a

# Build targets
all: $(MODULE_NAME).so

# Build the Python extension by leveraging DART's Makefile
$(MODULE_NAME).so: $(WRAPPER_SRC) $(DART_LIB)
	$(F2PY) -c -m $(MODULE_NAME) $(WRAPPER_SRC) \
	-I$(DART_INC) \
	-L$(DART_DIR) \
	-ldart \
	--f90flags="-ffree-line-length-none" \
	--fcompiler=gnu95

# Ensure DART library is built using their Makefile
$(DART_LIB):
	$(MAKE) -C $(DART_DIR) libdart.a

# Test the compiled module
test: $(MODULE_NAME).so
	@echo "Testing module import..."
	@$(PYTHON) -c "import $(MODULE_NAME); print('Successfully imported $(MODULE_NAME)')" \
		|| (echo "Import failed"; exit 1)

# Clean targets
clean:
	rm -f *.so *.mod
	$(MAKE) -C $(DART_DIR) clean

distclean: clean
	rm -f *.pyc

.PHONY: all test clean distclean
