//===- LegalizeForLLVMExport.cpp - Prepare Hexagon for LLVM translation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Hexagon/HexagonDialect.h"
#include "mlir/Dialect/Hexagon/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::hexagon;

// Extract an LLVM IR type from the LLVM IR dialect type.
static Type unwrap(Type type) {
	if (!type)
		return nullptr;
	auto *mlirContext = type.getContext();
	if (!LLVM::isCompatibleType(type))
		emitError(UnknownLoc::get(mlirContext),
		          "conversion resulted in a non-LLVM type");
	return type;
}

// Scalable vector type in Hexagon dialect uses LMUL and SEW as parameters to
// provide better semantics. This is the helper function to bridge the gap
// of scalable vector type between the Hexagon dialect and LLVM dialect.
unsigned typeMapping(ScalableVectorType hexagonSVType) {
	auto elementType = hexagonSVType.getElementType();
	auto *elementContext = elementType.getContext();
	auto sizeType = hexagonSVType.getSizeType();
	auto *sizeContext = sizeType.getContext();
	// TODO: support more element type.
	if (elementType.isa<IntegerType>()) {
		// Mapping LMUL and Mask type for different SEW type.
		switch (elementType.cast<IntegerType>().getWidth()) {
			case 64:
				if (sizeType.isa<MF8Type>() || sizeType.isa<MF4Type>() ||
				    sizeType.isa<MF2Type>()) {
					emitError(UnknownLoc::get(sizeContext), "unsupported LMUL Type for ")
									<< elementType << " type.";
				}
				return llvm::TypeSwitch<Type, unsigned>(sizeType)
								.Case<M1Type>([&](Type) { return 1; })
								.Case<M2Type>([&](Type) { return 2; })
								.Case<M4Type>([&](Type) { return 4; })
								.Case<M8Type>([&](Type) { return 8; })
								.Default([](Type) -> unsigned {
									llvm_unreachable("incompatible with RISC-V vector type");
								});
				break;
			case 32:
				if (sizeType.isa<MF8Type>() || sizeType.isa<MF4Type>()) {
					emitError(UnknownLoc::get(sizeContext), "unsupported LMUL Type for ")
									<< elementType << " type.";
				}
				return llvm::TypeSwitch<Type, unsigned>(sizeType)
								.Case<MF2Type>([&](Type) { return 1; })
								.Case<M1Type>([&](Type) { return 2; })
								.Case<M2Type>([&](Type) { return 4; })
								.Case<M4Type>([&](Type) { return 8; })
								.Case<M8Type>([&](Type) { return 16; })
								.Default([](Type) -> unsigned {
									llvm_unreachable("incompatible with RISC-V vector type");
								});
				break;
			case 16:
				if (sizeType.isa<MF8Type>()) {
					emitError(UnknownLoc::get(sizeContext), "unsupported LMUL type for ")
									<< elementType << " type.";
				}
				return llvm::TypeSwitch<Type, unsigned>(sizeType)
								.Case<MF4Type>([&](Type) { return 1; })
								.Case<MF2Type>([&](Type) { return 2; })
								.Case<M1Type>([&](Type) { return 4; })
								.Case<M2Type>([&](Type) { return 8; })
								.Case<M4Type>([&](Type) { return 16; })
								.Case<M8Type>([&](Type) { return 32; })
								.Default([](Type) -> unsigned {
									llvm_unreachable("incompatible with RISC-V vector type");
								});
				break;
			case 8:
				return llvm::TypeSwitch<Type, unsigned>(sizeType)
								.Case<MF8Type>([&](Type) { return 1; })
								.Case<MF4Type>([&](Type) { return 2; })
								.Case<MF2Type>([&](Type) { return 4; })
								.Case<M1Type>([&](Type) { return 8; })
								.Case<M2Type>([&](Type) { return 16; })
								.Case<M4Type>([&](Type) { return 32; })
								.Case<M8Type>([&](Type) { return 64; })
								.Default([](Type) -> unsigned {
									llvm_unreachable("incompatible with RISC-V vector type");
								});
				break;
			case 1:
				return llvm::TypeSwitch<Type, unsigned>(sizeType)
								.Case<Mask64Type>([&](Type) { return 1; })
								.Case<Mask32Type>([&](Type) { return 2; })
								.Case<Mask16Type>([&](Type) { return 4; })
								.Case<Mask8Type>([&](Type) { return 8; })
								.Case<Mask4Type>([&](Type) { return 16; })
								.Case<Mask2Type>([&](Type) { return 32; })
								.Case<Mask1Type>([&](Type) { return 64; })
								.Default([](Type) -> unsigned {
									llvm_unreachable("incompatible with RISC-V vector type");
								});
				break;
			default:
				emitError(UnknownLoc::get(elementContext), "unsupported ")
								<< elementType << " SEW type.";
		}
	} else {
		emitError(UnknownLoc::get(elementContext), "unsupported ")
						<< elementType << " SEW type.";
	}
	return 0;
}

static Optional<Type>
convertScalableVectorTypeToLLVM(ScalableVectorType svType,
                                LLVMTypeConverter &converter) {
	auto elementType = unwrap(converter.convertType(svType.getElementType()));
	if (!elementType)
		return {};
	auto sVectorType =
					LLVM::LLVMScalableVectorType::get(elementType, typeMapping(svType));
	return sVectorType;
}

template <typename OpTy>
class ForwardOperands : public OpConversionPattern<OpTy> {
	using OpConversionPattern<OpTy>::OpConversionPattern;

	LogicalResult
	matchAndRewrite(OpTy op, ArrayRef<Value> operands,
	                ConversionPatternRewriter &rewriter) const final {
		if (ValueRange(operands).getTypes() == op->getOperands().getTypes())
			return rewriter.notifyMatchFailure(op, "operand types already match");

		rewriter.updateRootInPlace(op, [&]() { op->setOperands(operands); });
		return success();
	}
};

struct HexagonLoadOpLowering : public ConvertOpToLLVMPattern<HexagonLoadOp> {
	using ConvertOpToLLVMPattern<HexagonLoadOp>::ConvertOpToLLVMPattern;

	LogicalResult
	matchAndRewrite(HexagonLoadOp loadOp, ArrayRef<Value> operands,
	                ConversionPatternRewriter &rewriter) const override {
		auto type = loadOp.getMemRefType();
		if (!isConvertibleAndHasIdentityMaps(type))
			return failure();

		HexagonLoadOp::Adaptor transformed(operands);
		LLVMTypeConverter converter(loadOp.getContext());

		auto resultType = loadOp.result().getType();
		LLVM::LLVMPointerType llvmDataTypePtr;
		if (resultType.isa<VectorType>()) {
			llvmDataTypePtr =
							LLVM::LLVMPointerType::get(resultType.cast<VectorType>());
		} else if (resultType.isa<ScalableVectorType>()) {
			llvmDataTypePtr = LLVM::LLVMPointerType::get(
							convertScalableVectorTypeToLLVM(resultType.cast<ScalableVectorType>(),
							                                converter)
											.getValue());
		}
		Value dataPtr =
						getStridedElementPtr(loadOp.getLoc(), type, transformed.base(),
						                     transformed.index(), rewriter);
		Value bitCastedPtr = rewriter.create<LLVM::BitcastOp>(
						loadOp.getLoc(), llvmDataTypePtr, dataPtr);
		Value vl = loadOp.getOperand(2);
//		rewriter.replaceOpWithNewOp<HexagonIntrLoadEleOp>(
//						loadOp,
//						convertScalableVectorTypeToLLVM(resultType.cast<ScalableVectorType>(),
//						                                converter)
//										.getValue(),
//						bitCastedPtr, vl);
		return success();
	}
};

struct HexagonStoreOpLowering : public ConvertOpToLLVMPattern<HexagonStoreOp> {
	using ConvertOpToLLVMPattern<HexagonStoreOp>::ConvertOpToLLVMPattern;

	LogicalResult
	matchAndRewrite(HexagonStoreOp storeOp, ArrayRef<Value> operands,
	                ConversionPatternRewriter &rewriter) const override {
		auto type = storeOp.getMemRefType();
		if (!isConvertibleAndHasIdentityMaps(type))
			return failure();

		HexagonStoreOp::Adaptor transformed(operands);
		LLVMTypeConverter converter(storeOp.getContext());

		auto resultType = storeOp.value().getType();
		LLVM::LLVMPointerType llvmDataTypePtr;
		if (resultType.isa<VectorType>()) {
			llvmDataTypePtr =
							LLVM::LLVMPointerType::get(resultType.cast<VectorType>());
		} else if (resultType.isa<ScalableVectorType>()) {
			llvmDataTypePtr = LLVM::LLVMPointerType::get(
							convertScalableVectorTypeToLLVM(resultType.cast<ScalableVectorType>(),
							                                converter)
											.getValue());
		}
		Value dataPtr =
						getStridedElementPtr(storeOp.getLoc(), type, transformed.base(),
						                     transformed.index(), rewriter);
		Value bitCastedPtr = rewriter.create<LLVM::BitcastOp>(
						storeOp.getLoc(), llvmDataTypePtr, dataPtr);
		Value vl = storeOp.getOperand(3);
//		rewriter.replaceOpWithNewOp<HexagonIntrStoreEleOp>(
//						storeOp, transformed.value(), bitCastedPtr, vl);
		return success();
	}
};

using HexagonAddOpLowering =
OneToOneConvertToLLVMPattern<HexagonAddOp, HexagonIntrAddOp>;
using HexagonSubOpLowering =
OneToOneConvertToLLVMPattern<HexagonSubOp, HexagonIntrSubOp>;
using HexagonMulOpLowering =
OneToOneConvertToLLVMPattern<HexagonMulOp, HexagonIntrMulOp>;
//using HexagonDivOpLowering =
//OneToOneConvertToLLVMPattern<HexagonDivOp, HexagonIntrDivOp>;
//using HexagonMaskedAddOpLowering =
//OneToOneConvertToLLVMPattern<HexagonMaskedAddOp, HexagonMaskedIntrAddOp>;
//using HexagonMaskedSubOpLowering =
//OneToOneConvertToLLVMPattern<HexagonMaskedSubOp, HexagonMaskedIntrSubOp>;
//using HexagonMaskedMulOpLowering =
//OneToOneConvertToLLVMPattern<HexagonMaskedMulOp, HexagonMaskedIntrMulOp>;
//using HexagonMaskedDivOpLowering =
//OneToOneConvertToLLVMPattern<HexagonMaskedDivOp, HexagonMaskedIntrDivOp>;

/// Populate the given list with patterns that convert from Hexagon to LLVM.
void mlir::populateHexagonLegalizeForLLVMExportPatterns(
				LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
	// Populate conversion patterns.
	// Remove any Hexagon-specific types from function signatures and results.
	populateFuncOpTypeConversionPattern(patterns, converter);
	converter.addConversion([&converter](ScalableVectorType hexagonSVType) {
		return convertScalableVectorTypeToLLVM(hexagonSVType, converter);
	});

	// clang-format off
	patterns.add<
	        ForwardOperands<CallOp>,
					ForwardOperands<CallIndirectOp>,
					ForwardOperands<ReturnOp>>(converter, &converter.getContext());
	patterns.add<
	        HexagonLoadOpLowering,
					HexagonStoreOpLowering>(converter);
	patterns.add<
	        HexagonAddOpLowering,
					HexagonSubOpLowering,
					HexagonMulOpLowering>(converter);
	// clang-format on
}

void mlir::configureHexagonLegalizeForExportTarget(
				LLVMConversionTarget &target) {
	// clang-format off
	target.addLegalOp<
	        HexagonIntrAddOp,
					HexagonIntrSubOp,
					HexagonIntrMulOp>();
	target.addIllegalOp<
	        HexagonAddOp,
					HexagonSubOp,
					HexagonMulOp>();
	// clang-format on

	auto hasScalableVectorType = [](TypeRange types) {
		for (Type type : types)
			if (type.isa<hexagon::ScalableVectorType>())
				return true;
		return false;
	};
	target.addDynamicallyLegalOp<FuncOp>([hasScalableVectorType](FuncOp op) {
		return !hasScalableVectorType(op.getType().getInputs()) &&
		       !hasScalableVectorType(op.getType().getResults());
	});
	target.addDynamicallyLegalOp<CallOp, CallIndirectOp, ReturnOp>(
					[hasScalableVectorType](Operation *op) {
						return !hasScalableVectorType(op->getOperandTypes()) &&
						       !hasScalableVectorType(op->getResultTypes());
					});
}