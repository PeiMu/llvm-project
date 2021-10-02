add_mlir_dialect(Hexagon hexagon)
add_mlir_doc(Hexagon Hexagon Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Hexagon.td)
mlir_tablegen(HexagonConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRHexagonConversionsIncGen)