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
include "mlir/Dialect/Vector/ScalableVectorTypes.td"

//===----------------------------------------------------------------------===//
// Hexagon dialect definition
//===----------------------------------------------------------------------===//

def Hexagon_Dialect : Dialect {
  let name = "hexagon";
  let cppNamespace = "::mlir::hexagon";
  let summary = "Basic dialect to target RISC-V Vector extension";
  let description = [{
    RISC-V vector extension (Hexagon) is the vector instruction set with scalable
    vector types, and the Hexagon instructions are vector length agnostic (VLA).
    For more details about Hexagon, please see the
    [RISCVV specification](https://github.com/riscv/riscv-v-spec).
    This dialect contains the definitions of RISCVV operations and Hexagon
    intrinsic operations. The former is used to interoperate with higher-level
    dialects, and the latter is responsible for mapping to LLVM IR intrinsic.
  }];
}

//===----------------------------------------------------------------------===//
// Hexagon LMUL type definitions
//===----------------------------------------------------------------------===//

class Hexagon_LMULType<string name, string typeMnemonic, string multiplier,
                      list<Trait> traits = []>
    : TypeDef<Hexagon_Dialect, name, traits, "::mlir::HexagonLMULType"> {
  let mnemonic = typeMnemonic;
  let summary = "The vector register group multiplier (" # multiplier # ").";
  let printer = [{$_printer << "}]# typeMnemonic #[{";}];
  let parser = [{return get}]# name #[{($_ctxt);}];
}

def Hexagon_LMUL_MF8 : Hexagon_LMULType<"MF8", "mf8", "1/8"> {}
def Hexagon_LMUL_MF4 : Hexagon_LMULType<"MF4", "mf4", "1/4"> {}
def Hexagon_LMUL_MF2 : Hexagon_LMULType<"MF2", "mf2", "1/2"> {}
def Hexagon_LMUL_M1 : Hexagon_LMULType<"M1", "m1", "1"> {}
def Hexagon_LMUL_M2 : Hexagon_LMULType<"M2", "m2", "2"> {}
def Hexagon_LMUL_M4 : Hexagon_LMULType<"M4", "m4", "4"> {}
def Hexagon_LMUL_M8 : Hexagon_LMULType<"M8", "m8", "8"> {}

//===----------------------------------------------------------------------===//
// Hexagon mask type definitions
//===----------------------------------------------------------------------===//

class Hexagon_MaskType<string name, string typeMnemonic, string maskLength,
                      list<Trait> traits = []>
    : TypeDef<Hexagon_Dialect, name, traits, "::mlir::HexagonMaskType"> {
  let mnemonic = typeMnemonic;
  let summary = "The mask length (SEW/LMUL = " # maskLength # ") type.";
  let printer = [{$_printer << "}]# typeMnemonic #[{";}];
  let parser = [{return get}]# name #[{($_ctxt);}];
}

def Hexagon_Mask1 : Hexagon_MaskType<"Mask1", "mask1", "1"> {}
def Hexagon_Mask2 : Hexagon_MaskType<"Mask2", "mask2", "2"> {}
def Hexagon_Mask4 : Hexagon_MaskType<"Mask4", "mask4", "4"> {}
def Hexagon_Mask8 : Hexagon_MaskType<"Mask8", "mask8", "8"> {}
def Hexagon_Mask16 : Hexagon_MaskType<"Mask16", "mask16", "16"> {}
def Hexagon_Mask32 : Hexagon_MaskType<"Mask32", "mask32", "32"> {}
def Hexagon_Mask64 : Hexagon_MaskType<"Mask64", "mask64", "64"> {}

//===----------------------------------------------------------------------===//
// Hexagon scalable vector type definitions
//===----------------------------------------------------------------------===//

def ScalableVectorType : ScalableVector_Type<Hexagon_Dialect, "ScalableVector"> {
  let description = [{
    Hexagon scalable vector type takes two parameters. The first one is vector
    register group multiplier (LMUL) type or mask type. The second one is the
    element type, which indicates the selected element width (SEW) setting.
    The LMUL and SEW are used to configure scalable vector length at runtime.
  }];

  let parameters = (ins "Type":$sizeType, "Type":$elementType);

  let printer = [{
    $_printer << "vector<" << getImpl()->sizeType << ',';
    $_printer << getImpl()->elementType << '>';
  }];

  let parser = [{
    if ($_parser.parseLess()) return Type();
    Type sizeType;
    if ($_parser.parseType(sizeType)) return Type();
    if ($_parser.parseComma()) return Type();
    Type elementType;
    if ($_parser.parseType(elementType)) return Type();
    if ($_parser.parseGreater()) return Type();
    return get($_ctxt, sizeType, elementType);
  }];
}
//===----------------------------------------------------------------------===//
// Additional LLVM type constraints
//===----------------------------------------------------------------------===//

def LLVMScalableVectorType :
  Type<CPred<"$_self.isa<::mlir::LLVM::LLVMScalableVectorType>()">,
       "LLVM dialect scalable vector type">;

def LLVMPointerType :
  Type<CPred<"$_self.isa<::mlir::LLVM::LLVMPointerType>()">,
       "LLVM pointer type">;

//===----------------------------------------------------------------------===//
// Hexagon scalable vector type constraints
//===----------------------------------------------------------------------===//

def IsScalableVectorTypePred :
    CPred<"$_self.isa<::mlir::hexagon::ScalableVectorType>()">;

class HexagonScalableVectorOf<list<Type> allowedTypes> :
    ContainerType<AnyTypeOf<allowedTypes>, IsScalableVectorTypePred,
          "$_self.cast<::mlir::hexagon::ScalableVectorType>().getElementType()",
          "Hexagon scalable vector">;

//===----------------------------------------------------------------------===//
// Hexagon operation definitions
//===----------------------------------------------------------------------===//

class Hexagon_Op<string mnemonic, list<OpTrait> traits = []> :
  Op<Hexagon_Dialect, mnemonic, traits> {}

def HexagonLoadOp : Hexagon_Op<"load">,
    Arguments<(ins Arg<AnyMemRef, "", [MemRead]>:$base, Index:$index,
                       AnyInteger:$length)>,
    Results<(outs HexagonScalableVectorOf<[AnyType]>:$result)> {
  let summary = "Load scalable vector from memory";
  let description = [{
    Load a slice of memory into scalable vector with the given element length.
  }];
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
  }];
  let assemblyFormat = "$base `[` $index `]` `,` $length attr-dict `:` "
    "type($base) `,` type($result) `,`  type($length)";
}

def HexagonStoreOp : Hexagon_Op<"store">,
    Arguments<(ins HexagonScalableVectorOf<[AnyType]>:$value, Arg<AnyMemRef, "",
                   [MemWrite]>:$base, Index:$index, AnyInteger:$length)> {
  let summary = "Store scalable vector into memory";
  let description = [{
    Store the given element length of a scalable vector on a slice of memory.
  }];
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
  }];
  let assemblyFormat = "$value `,` $base `[` $index `]` `,` $length attr-dict "
    "`:` type($value) `,` type($base) `,`  type($length)";
}

class Hexagon_BinaryAAXNoMask_Op<string mnemonic, string op_description,
                             list<OpTrait> traits = []> :
  Hexagon_Op<mnemonic, !listconcat(traits, [AllTypesMatch<["src1", "dst"]>])> {
  let summary = op_description # "for Hexagon scalable vectors";
  let description = [{ The `hexagon.}] # mnemonic # [{` operation can be of
    vector-vector form or vector-scalar form. It also takes the vector length
    value and returns a scalable vector with the result of the }]
    # op_description # [{.}];
  let arguments = (ins
          HexagonScalableVectorOf<[AnyInteger]>:$src1,
          AnyType:$src2,
          AnyInteger:$length
  );
  let results = (outs HexagonScalableVectorOf<[AnyInteger]>:$dst);
  let assemblyFormat = "$src1 `,` $src2 `,` $length attr-dict `:` type($src1) "
    "`,` type($src2) `,` type($length)";
}

class Hexagon_BinaryAAXMask_Op<string mnemonic, string op_description,
                              list<OpTrait> traits = []> :
  Hexagon_Op<mnemonic, !listconcat(traits,
                       [AllTypesMatch<["maskedoff", "src1", "dst"]>])> {
  let summary = op_description # "for Hexagon scalable vectors";
  let description = [{ The `hexagon.}] # mnemonic # [{` operation can be of
    vector-vector form or vector-scalar form. It also takes the mask vector,
    maskedoff vector, vector length value and returns a scalable vector with
    the result of the }] # op_description # [{.}];
  let arguments = (ins
          HexagonScalableVectorOf<[AnyInteger]>:$maskedoff,
          HexagonScalableVectorOf<[AnyInteger]>:$src1,
          AnyType:$src2,
          HexagonScalableVectorOf<[I1]>:$mask,
          AnyInteger:$length
  );
  let results = (outs HexagonScalableVectorOf<[AnyInteger]>:$dst);
  let assemblyFormat =
    "$maskedoff `,` $src1 `,` $src2 `,` $mask `,` $length attr-dict `:` "
    "type($maskedoff) `,` type($src2) `,` type($mask) `,` type($length)";
}

def HexagonAddOp : Hexagon_BinaryAAXNoMask_Op<"add", "addition">;
def HexagonSubOp : Hexagon_BinaryAAXNoMask_Op<"sub", "subtraction">;
def HexagonMulOp : Hexagon_BinaryAAXNoMask_Op<"mul", "multiplication">;
def HexagonDivOp : Hexagon_BinaryAAXNoMask_Op<"div", "division">;

def HexagonMaskedAddOp : Hexagon_BinaryAAXMask_Op<"masked.add",
                                                "masked addition">;
def HexagonMaskedSubOp : Hexagon_BinaryAAXMask_Op<"masked.sub",
                                                "masked subtraction">;
def HexagonMaskedMulOp : Hexagon_BinaryAAXMask_Op<"masked.mul",
                                                "masked multiplication">;
def HexagonMaskedDivOp : Hexagon_BinaryAAXMask_Op<"masked.div",
                                                "masked division">;

//===----------------------------------------------------------------------===//
// Hexagon intrinsic operation definitions
//===----------------------------------------------------------------------===//

/*
class Hexagon_USLoad_IntrOp<string mnemonic, list<OpTrait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/Hexagon_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"hexagon_V6_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0],
                  /*list<int> overloadedOperands=*/[1],
                  /*list<OpTrait> traits=*/traits,
                  /*int numResults=*/1>;

class Hexagon_USStore_IntrOp<string mnemonic, list<OpTrait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/Hexagon_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"hexagon_V6_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[],
                  /*list<int> overloadedOperands=*/[0, 2],
                  /*list<OpTrait> traits=*/traits,
                  /*int numResults=*/0>;
*/

class Hexagon_BinaryAAXNoMask_IntrOp<string mnemonic,
                                    list<OpTrait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/Hexagon_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"hexagon_V6_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0],
                  /*list<int> overloadedOperands=*/[1, 2],
                  /*list<OpTrait> traits=*/traits,
                  /*int numResults=*/1>;

/*
class Hexagon_BinaryAAXMask_IntrOp<string mnemonic, list<OpTrait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/Hexagon_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"hexagon_V6_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0],
                  /*list<int> overloadedOperands=*/[2, 4],
                  /*list<OpTrait> traits=*/traits,
                  /*int numResults=*/1>;
*/

/*
def HexagonIntrLoadEleOp : Hexagon_USLoad_IntrOp<"vle">,
  Arguments<(ins LLVMPointerType, AnyInteger)>;
def HexagonIntrStoreEleOp : Hexagon_USStore_IntrOp<"vse">,
  Arguments<(ins LLVMScalableVectorType, LLVMPointerType, AnyInteger)>;
*/

def HexagonIntrAddOp : Hexagon_BinaryAAXNoMask_IntrOp<"vaddh">,
  Arguments<(ins LLVMScalableVectorType, AnyType, AnyInteger)>;
def HexagonIntrSubOp : Hexagon_BinaryAAXNoMask_IntrOp<"vsubh">,
  Arguments<(ins LLVMScalableVectorType, AnyType, AnyInteger)>;
def HexagonIntrMulOp : Hexagon_BinaryAAXNoMask_IntrOp<"vmpyh">,
  Arguments<(ins LLVMScalableVectorType, AnyType, AnyInteger)>;
/*def HexagonIntrDivOp : Hexagon_BinaryAAXNoMask_IntrOp<"vdiv">,
  Arguments<(ins LLVMScalableVectorType, AnyType, AnyInteger)>;

def HexagonMaskedIntrAddOp : Hexagon_BinaryAAXMask_IntrOp<"vadd_mask">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType, AnyType,
                 LLVMScalableVectorType, AnyInteger)>;
def HexagonMaskedIntrSubOp : Hexagon_BinaryAAXMask_IntrOp<"vsub_mask">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType, AnyType,
                 LLVMScalableVectorType, AnyInteger)>;
def HexagonMaskedIntrMulOp : Hexagon_BinaryAAXMask_IntrOp<"vmul_mask">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType, AnyType,
                 LLVMScalableVectorType, AnyInteger)>;
def HexagonMaskedIntrDivOp : Hexagon_BinaryAAXMask_IntrOp<"vdiv_mask">,
  Arguments<(ins LLVMScalableVectorType, LLVMScalableVectorType, AnyType,
                 LLVMScalableVectorType, AnyInteger)>;
*/

#endif // Hexagon_OPS