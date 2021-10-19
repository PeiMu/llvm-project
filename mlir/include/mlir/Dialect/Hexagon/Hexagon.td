//===-- Hexagon.td - Hexagon dialect operation definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the Hexagon dialect.
//
//===----------------------------------------------------------------------===//

#ifndef Hexagon_OPS
#define Hexagon_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Hexagon dialect definition
//===----------------------------------------------------------------------===//

def Hexagon_Dialect : Dialect {
  let name = "hexagon";
  let cppNamespace = "::mlir::hexagon";
  let summary = "Basic dialect to target Hexagon Vector extension";
  let description = [{
    // todo: supplement
    This dialect contains the definitions of Hexagon operations and Hexagon
    intrinsic operations.
  }];
}

//===----------------------------------------------------------------------===//
// Hexagon operation definitions
//===----------------------------------------------------------------------===//

// Hexagon dialect op that corresponds (and is convertible to) an LLVM IR
// intrinsic.
class Hexagon_IntrOp<string mnemonic, list<int> overloadedResults,
                     list<int> overloadedOperands, int numResults,
                     list<OpTrait> traits = [], bit requiresAccessGroup = 0,
                     bit requiresAliasScope = 0>
    : LLVM_IntrOpBase</*dialect=*/Hexagon_Dialect,
                      /*opName=*/"intr." # mnemonic,
                      /*enumName=*/"hexagon_V6_" # !subst(".", "_", mnemonic),
                      /*overloadedResults=*/overloadedResults,
                      /*overloadedOperands=*/overloadedOperands,
                      /*traits=*/traits,
                      /*numResults=*/numResults,
                      /*requiresAccessGroup=*/requiresAccessGroup,
                      /*requiresAliasScope=*/requiresAliasScope>;
                      
// Hexagon dialect op that corresponds to an LLVM IR intrinsic with one
// overloaded result.
class Hexagon_OverloadedOneResultIntrOp<string mnemonic,
                                        list<OpTrait> traits = []>
  : Hexagon_IntrOp<mnemonic, [], [], 1, traits>;

// Hexagon dialect op that corresponds to an LLVM IR intrinsic with one
// overloaded result and overloaded operands list.
class Hexagon_OverloadedOperandsWithOneResultIntrOp<string mnemonic,
                                                    list<int> overloadedOperands,
                                                    list<OpTrait> traits = []>
  : Hexagon_IntrOp<mnemonic, [], overloadedOperands, 1, traits>;

class Hexagon_NoOverloadedIntrOp<string mnemonic, list<OpTrait> traits = []>
  : Hexagon_IntrOp<mnemonic, [], [], 1, traits>;

class HexagonIntrBinaryOp<string mnemonic, string op_description> :
    Hexagon_NoOverloadedIntrOp<mnemonic, [
       NoSideEffect,
       AllTypesMatch<["a", "b"]>,
       SameOperandsAndResultType]> {
  let summary = "binary op";
  let description = [{
    Signed Multiply Long (vector). This instruction add corresponding
    signed integer values by elementwise.
  }];

  let arguments = (ins VectorOfLengthAndType<[16], [I32]>:$a,
                         VectorOfLengthAndType<[16], [I32]>:$b);
    let results = (outs VectorOfLengthAndType<[16], [I32]>:$res);
    let assemblyFormat =
      "$a `,` $b attr-dict `:` type($a)";
}

def HexagonIntrAddOp : HexagonIntrBinaryOp<"vaddw", "addition">;

#endif // Hexagon_OPS
