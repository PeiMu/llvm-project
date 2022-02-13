; Scenario 3: Check the case where it is illegal to create a masked interleave-
; group because the two accesses are in separate predicated blocks.
; We therefore create a separate interleave-group with gaps for each of the accesses,
; If masked-interleaved-accesses is not enabled we don't create any interleave
; group because all accesses are predicated.
;
; void masked_strided3(const unsigned char* restrict p,
;                     unsigned char* restrict q,
;                     unsigned char guard1,
;                     unsigned char guard2) {
; for(ix=0; ix < 1024; ++ix) {
;     if (ix > guard1) {
;         q[2*ix] = 1;
;     }
;     if (ix > guard2) {
;         q[2*ix+1] = 2;
;     }
; }
;}


; STRIDED_UNMASKED: LV: Checking a loop in "masked_strided3" 
; STRIDED_UNMASKED: LV: Analyzing interleaved accesses...
; STRIDED_UNMASKED-NOT: LV: Creating an interleave group 

; STRIDED_MASKED: LV: Checking a loop in "masked_strided3" 
; STRIDED_MASKED: LV: Analyzing interleaved accesses...
; STRIDED_MASKED-NEXT: LV: Creating an interleave group with:  store i8 2, i8* %{{.*}}, align 1
; STRIDED_MASKED-NEXT: LV: Creating an interleave group with:  store i8 1, i8* %{{.*}}, align 1
; STRIDED_MASKED-NOT: LV: Invalidate candidate interleaved store group due to gaps.

define dso_local void @masked_strided3(i8* noalias nocapture readnone %p, i8* noalias nocapture %q, i8 zeroext %guard1, i8 zeroext %guard2) local_unnamed_addr #0 {
entry:
  %conv = zext i8 %guard1 to i32
  %conv3 = zext i8 %guard2 to i32
  br label %for.body

for.body:
  %ix.018 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %mul = shl nuw nsw i32 %ix.018, 1
  %cmp1 = icmp ugt i32 %ix.018, %conv
  br i1 %cmp1, label %if.then, label %if.end

if.then:
  %arrayidx = getelementptr inbounds i8, i8* %q, i32 %mul
  store i8 1, i8* %arrayidx, align 1
  br label %if.end

if.end:
  %cmp4 = icmp ugt i32 %ix.018, %conv3
  br i1 %cmp4, label %if.then6, label %for.inc

if.then6:
  %add = or i32 %mul, 1
  %arrayidx7 = getelementptr inbounds i8, i8* %q, i32 %add
  store i8 2, i8* %arrayidx7, align 1
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %ix.018, 1
  %exitcond = icmp eq i32 %inc, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
