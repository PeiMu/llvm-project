//===- eclAST.h - Node definition for the AST of easy_compiler_language ---===//
//
// This file implements the AST for the easy_compiler_language. Following by the
// toy language from the mlir tutorial.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ECLAST_H
#define LLVM_ECLAST_H

#include "eclLexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace ecl {

/// A variable type with shape information.
struct VarType {
  llvm::SmallVector<int64_t> shape;
};

/// Base class for all expression nodes.
class ExprAST {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_Var,
    Expr_BinOp,
    Expr_Call,
    Expr_Print,
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const ExprASTKind kind;
  Location location;
};

/// A block-list of expressions.
using ExprASTList = llvm::SmallVector<std::unique_ptr<ExprAST>>;

/// Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
public:
  NumberExprAST(Location loc, double val) : ExprAST(Expr_Num, loc), val(val) {}

  double getValue() { return val; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }

private:
  double val;
};

/// Expression class for a literal value.
class LiteralExprAST : public ExprAST {
public:
  LiteralExprAST(Location loc,
                 llvm::SmallVector<std::unique_ptr<ExprAST>> values,
                 llvm::SmallVector<int64_t> dims)
      : ExprAST(Expr_Literal, loc), values(std::move(values)),
        dims(std::move(dims)) {}

  // todo: - Why need vector and dims here?
  llvm::SmallVector<std::unique_ptr<ExprAST>> getValues() { return values; }
  llvm::SmallVector<int64_t> getDims() { return dims; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal; }

private:
  llvm::SmallVector<std::unique_ptr<ExprAST>> values;
  llvm::SmallVector<int64_t> dims;
};

/// Expression class for *referencing* a variable, like "a".
class VariableExprAST : public ExprAST {
public:
  VariableExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Var, loc), name(name) {}

  llvm::StringRef getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Var; }

private:
  llvm::StringRef name;
};

/// Expression class for *defining* a variable.
class VarDeclExprAST : public ExprAST {
public:
  VarDeclExprAST(Location loc, llvm::StringRef name, VarType type,
                 std::unique_ptr<ExprAST> initVal)
      : ExprAST(Expr_VarDecl, loc), name(name), type(std::move(type)),
        initVal(std::move(initVal)) {}

	llvm::StringRef getName() { return name; }
	ExprAST *getInitVal() { return initVal.get(); }
	const VarType &getType() { return type; }

	/// LLVM style RTTI
	static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarDecl; }

private:
  llvm::StringRef name;
  VarType type;
  std::unique_ptr<ExprAST> initVal;
};



} // namespace ecl

#endif // LLVM_ECLAST_H
