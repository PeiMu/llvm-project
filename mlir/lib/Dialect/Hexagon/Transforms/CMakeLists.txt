add_mlir_dialect_library(MLIRHexagonTransforms
    LegalizeForLLVMExport.cpp

    DEPENDS
    MLIRHexagonConversionsIncGen

    LINK_LIBS PUBLIC
    MLIRHexagon
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMIR
    )