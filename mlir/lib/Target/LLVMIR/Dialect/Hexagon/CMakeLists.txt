add_mlir_translation_library(MLIRHexagonToLLVMIRTranslation
    HexagonToLLVMIRTranslation.cpp

    DEPENDS
    MLIRHexagonConversionsIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRHexagon
    MLIRLLVMIR
    MLIRSupport
    MLIRTargetLLVMIRExport
    )