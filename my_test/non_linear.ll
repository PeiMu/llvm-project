; Can't vectorize this loop because the access to A[X] is non-linear.
;
;  for (i = 0; i < n; ++i) {
;    A[B[i]]++;
;
;CHECK-LABEL: @histogram(
;CHECK-NOT: <4 x i32>
;CHECK: ret i32
define i32 @histogram(i32* nocapture noalias %A, i32* nocapture noalias %B, i32 %n) nounwind uwtable ssp {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %idxprom1 = sext i32 %0 to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %idxprom1
  %1 = load i32, i32* %arrayidx2, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, i32* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret i32 0
}

