define i32 @invalid_phi_types() {
entry:
  br label %for.body

for.body:
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %vec.sum.02 = phi <2 x i32> [ zeroinitializer, %entry ], [ <i32 8, i32 8>, %for.body ]
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, 16
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret i32 0
}

