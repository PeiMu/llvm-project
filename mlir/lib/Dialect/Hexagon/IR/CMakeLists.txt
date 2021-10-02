add_mlir_dialect_library(MLIRHexagon
    HexagonDialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Hexagon

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRLLVMIR
    )