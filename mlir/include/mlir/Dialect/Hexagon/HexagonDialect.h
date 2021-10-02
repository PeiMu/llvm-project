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

namespace mlir {

//===----------------------------------------------------------------------===//
// HexagonLMULType
//===----------------------------------------------------------------------===//

/// This HexagonLMULType represents the vector register group multiplier (LMUL)
/// setting. When the LMUL greater than 1, the multiplier (M1, M2, M4, M8)
/// represents the number of vector registers that are combined to form a
/// vector register group. The multiplier can also be fractional values (MF8,
/// MF4, MF2), which reduces the number of bits used in a vector register.
	class HexagonLMULType : public Type {
	public:
		using Type::Type;

		static HexagonLMULType getMF8(MLIRContext *ctx);
		static HexagonLMULType getMF4(MLIRContext *ctx);
		static HexagonLMULType getMF2(MLIRContext *ctx);
		static HexagonLMULType getM1(MLIRContext *ctx);
		static HexagonLMULType getM2(MLIRContext *ctx);
		static HexagonLMULType getM4(MLIRContext *ctx);
		static HexagonLMULType getM8(MLIRContext *ctx);
	};

//===----------------------------------------------------------------------===//
// HexagonMaskType
//===----------------------------------------------------------------------===//

/// This HexagonMaskType represents the mask length setting. The mask length
/// setting is equal to the ratio of SEW and LMUL (n = SEW/LMUL).
	class HexagonMaskType : public Type {
	public:
		using Type::Type;

		static HexagonMaskType getMask1(MLIRContext *ctx);
		static HexagonMaskType getMask2(MLIRContext *ctx);
		static HexagonMaskType getMask4(MLIRContext *ctx);
		static HexagonMaskType getMask8(MLIRContext *ctx);
		static HexagonMaskType getMask16(MLIRContext *ctx);
		static HexagonMaskType getMask32(MLIRContext *ctx);
		static HexagonMaskType getMask64(MLIRContext *ctx);
	};

} // end namespace mlir

#include "mlir/Dialect/Hexagon/HexagonDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Hexagon/HexagonTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Hexagon/Hexagon.h.inc"

#endif // MLIR_DIALECT_Hexagon_HexagonDIALECT_H