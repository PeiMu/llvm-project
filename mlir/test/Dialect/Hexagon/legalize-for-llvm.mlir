// RUN: mlir-opt %s -convert-vector-to-llvm="enable-hexagon" -convert-std-to-llvm | mlir-opt | FileCheck %s

// CHECK-LABEL: func @hexagon_arith
func @hexagon_arith(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CHECK: hexagon.intr.vaddw {{.*}} : vector<16xi32>
  %0 = hexagon.intr.vaddw %a, %b : vector<16xi32>
  return %0 : vector<16xi32>
}

