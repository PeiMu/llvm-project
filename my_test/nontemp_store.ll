; Vectors with i4 elements may not legal with nontemporal stores.
define void @test_i4_store(i4* %ddst) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi i4* [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i4, i4* %ddst.addr, i64 1
  store i4 10, i4* %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

!8 = !{i32 1}

