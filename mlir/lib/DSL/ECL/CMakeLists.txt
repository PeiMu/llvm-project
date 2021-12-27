add_custom_target(ECL)
set_target_properties(ECL PROPERTIES FOLDER Examples)

set(LLVM_LINK_COMPONENTS
        Support
        )

add_dependencies(ECL ecl)
add_llvm_example(ecl
        eclInterface.cpp
        parser/eclAST.cpp)
