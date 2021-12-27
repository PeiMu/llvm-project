//===- eclAST.h - Lexer for the easy_compiler_language ---------------------==//
//
// This file implements a simple lexer for the easy_compiler_language.
// Following by the toy language from the mlir tutorial.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_ECLLEXER_H
#define LLVM_ECLLEXER_H

#include "llvm/ADT/StringRef.h"

#include <memory>

namespace ecl {

/// Structure definition a location in a file.
struct Location {
	std::shared_ptr<llvm::StringRef> file; 	///< filename
	int64_t line;                           ///< line number
	int64_t col;                            ///< column number
};

} // namespace ecl

#endif //LLVM_ECLLEXER_H
