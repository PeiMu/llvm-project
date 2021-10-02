//===- ScalableVectorTypes.td - Scalable vector types definitions --------====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR scalable vector types.
//
//===----------------------------------------------------------------------===//

#ifndef SCALABLE_VECTOR_TYPES
#define SCALABLE_VECTOR_TYPES

class ScalableVector_Type<Dialect dialect,string name>
    : TypeDef<dialect, name> {
  let mnemonic = "vector";

  let summary = "Scalable vector type";
}

#endif // SCALABLE_VECTOR_TYPES