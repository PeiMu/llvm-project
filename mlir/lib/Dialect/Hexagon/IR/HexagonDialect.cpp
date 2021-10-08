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

//===----------------------------------------------------------------------===//
// HexagonLMULType
//===----------------------------------------------------------------------===//

HexagonLMULType HexagonLMULType::getMF8(MLIRContext *ctx) {
	return hexagon::MF8Type::get(ctx);
}

HexagonLMULType HexagonLMULType::getMF4(MLIRContext *ctx) {
	return hexagon::MF4Type::get(ctx);
}

HexagonLMULType HexagonLMULType::getMF2(MLIRContext *ctx) {
	return hexagon::MF2Type::get(ctx);
}

HexagonLMULType HexagonLMULType::getM1(MLIRContext *ctx) {
	return hexagon::M1Type::get(ctx);
}

HexagonLMULType HexagonLMULType::getM2(MLIRContext *ctx) {
	return hexagon::M2Type::get(ctx);
}

HexagonLMULType HexagonLMULType::getM4(MLIRContext *ctx) {
	return hexagon::M4Type::get(ctx);
}

HexagonLMULType HexagonLMULType::getM8(MLIRContext *ctx) {
	return hexagon::M8Type::get(ctx);
}

//===----------------------------------------------------------------------===//
// HexagonMaskType
//===----------------------------------------------------------------------===//

HexagonMaskType HexagonMaskType::getMask1(MLIRContext *ctx) {
	return hexagon::Mask1Type::get(ctx);
}

HexagonMaskType HexagonMaskType::getMask2(MLIRContext *ctx) {
	return hexagon::Mask2Type::get(ctx);
}

HexagonMaskType HexagonMaskType::getMask4(MLIRContext *ctx) {
	return hexagon::Mask4Type::get(ctx);
}

HexagonMaskType HexagonMaskType::getMask8(MLIRContext *ctx) {
	return hexagon::Mask8Type::get(ctx);
}

HexagonMaskType HexagonMaskType::getMask16(MLIRContext *ctx) {
	return hexagon::Mask16Type::get(ctx);
}

HexagonMaskType HexagonMaskType::getMask32(MLIRContext *ctx) {
	return hexagon::Mask32Type::get(ctx);
}

HexagonMaskType HexagonMaskType::getMask64(MLIRContext *ctx) {
	return hexagon::Mask64Type::get(ctx);
}

//===----------------------------------------------------------------------===//
// Parser and Printer
//===----------------------------------------------------------------------===//

Type hexagon::HexagonDialect::parseType(DialectAsmParser &parser) const {
	llvm::SMLoc typeLoc = parser.getCurrentLocation();
	StringRef mnemonic;
	parser.parseKeyword(&mnemonic);
	{
		Type genType;
		auto parseResult = generatedTypeParser(parser, mnemonic, genType);
		if (parseResult.hasValue())
			return genType;
	}
	parser.emitError(typeLoc, "unknown type in Hexagon dialect");
	return Type();
}

void hexagon::HexagonDialect::printType(Type type, DialectAsmPrinter &os) const {
	if (failed(generatedTypePrinter(type, os)))
		llvm_unreachable("unexpected 'hexagon' type kind");
}