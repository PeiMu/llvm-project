//===- HexagonDialect.cpp - MLIR Hexagon dialect implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Hexagon dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Hexagon/HexagonDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "mlir/Dialect/Hexagon/HexagonDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Hexagon/Hexagon.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Hexagon/HexagonTypes.cpp.inc"

void hexagon::HexagonDialect::initialize() {
	addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Hexagon/Hexagon.cpp.inc"
	>();
	addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Hexagon/HexagonTypes.cpp.inc"
	>();
}
