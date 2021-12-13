add_mlir_conversion_library(MLIRFooToLinalg
    FooToLinalg.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/FooToLinalg

    LINK_LIBS PUBLIC
    MLIRLinalg
    MLIRFoo
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
