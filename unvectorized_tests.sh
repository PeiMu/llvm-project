./build/bin/opt -S -loop-vectorize -enable-interleaved-mem-accesses -enable-masked-interleaved-mem-accesses -disable-output my_test/seperate_masked_interleave.ll
./build/bin/opt -S -loop-vectorize -disable-output my_test/invalid_phi_types.ll
./build/bin/opt -S -loop-vectorize -disable-output my_test/illegal_pre_header.ll
./build/bin/opt -S -loop-vectorize -mtriple=arm64-apple-iphones -force-vector-width=4 -force-vector-interleave=1 -disable-output my_test/nontemp_store.ll
./build/bin/opt -S -loop-vectorize -disable-output my_test/memory_conflicts.ll
./build/bin/opt -S -loop-vectorize -disable-output my_test/non_linear.ll
./build/bin/opt -S -loop-vectorize -enable-vplan-native-path -disable-output my_test/divergen_inner_loop.ll
