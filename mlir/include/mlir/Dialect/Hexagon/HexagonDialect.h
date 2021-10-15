//===- HexagonDialect.h - MLIR Dialect for RISC-V vector extension --*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_Hexagon_HexagonDIALECT_H
#define MLIR_DIALECT_Hexagon_HexagonDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Hexagon/HexagonDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Hexagon/HexagonTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Hexagon/Hexagon.h.inc"

#endif // MLIR_DIALECT_Hexagon_HexagonDIALECT_H