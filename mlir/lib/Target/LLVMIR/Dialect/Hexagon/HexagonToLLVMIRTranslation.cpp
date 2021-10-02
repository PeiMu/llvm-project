//======- HexagonToLLVMIRTranslation.cpp - Translate Hexagon to LLVM IR ----====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the Hexagon dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/Hexagon/HexagonToLLVMIRTranslation.h"
#include "mlir/Dialect/Hexagon/HexagonDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsHexagon.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the Hexagon dialect to LLVM IR.
	class HexagonDialectLLVMIRTranslationInterface
					: public LLVMTranslationDialectInterface {
	public:
		using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

		/// Translates the given operation to LLVM IR using the provided IR builder
		/// and saving the state in `moduleTranslation`.
		LogicalResult
		convertOperation(Operation *op, llvm::IRBuilderBase &builder,
		                 LLVM::ModuleTranslation &moduleTranslation) const final {
			Operation &opInst = *op;
#include "mlir/Dialect/Hexagon/HexagonConversions.inc"

			return failure();
		}
	};
} // end namespace

void mlir::registerHexagonDialectTranslation(DialectRegistry &registry) {
	registry.insert<hexagon::HexagonDialect>();
	registry.addDialectInterface<hexagon::HexagonDialect,
					HexagonDialectLLVMIRTranslationInterface>();
}

void mlir::registerHexagonDialectTranslation(MLIRContext &context) {
	DialectRegistry registry;
	registerHexagonDialectTranslation(registry);
	context.appendDialectRegistry(registry);
}