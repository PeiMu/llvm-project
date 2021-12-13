add_mlir_dialect(FooOps foo)
add_mlir_doc(FooOps FooDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS FooTransforms.td)
mlir_tablegen(FooTransforms.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRFooTransformIncGen)