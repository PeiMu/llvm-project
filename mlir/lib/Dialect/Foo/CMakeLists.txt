add_mlir_dialect_library(MLIRFoo
  DEPENDS
  MLIRFooOpsIncGen
  MLIRFooTransformsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRLinalg
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)