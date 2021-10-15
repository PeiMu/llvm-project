// RUN: mlir-opt %s -convert-vector-to-llvm="enable-hexagon" -convert-std-to-llvm | mlir-opt | FileCheck %s

// CHECK-LABEL: func @hexagon_arith
func @hexagon_arith(%a: vector<4xi16>, %b: vector<4xi16>) -> vector<4xi16> {
  // CHECK: hexagon.intr.vaddhw {{.*}} : vector<4xi16>
  %0 = hexagon.intr.vaddhw %a, %b : vector<4xi16>
  return %0 : vector<4xi16>
}

