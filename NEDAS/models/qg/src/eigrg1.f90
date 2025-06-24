! package eigrg1         documentation for individual routines
!                        follows the general package information
!
! latest revision        january 1985
!
! purpose                eigrg1 is a package of two subroutines, eigrg1
!                        and eigvec.  these subroutines, along with a
!                        separate subroutine, eigrg2, compute all the
!                        eigenvalues of a real general matrix.
!                        optionally, all eigenvectors can be computed
!                        or some selected eigenvectors can be computed.
!
! usage                  the user first calls eigrg1 which computes
!                        all eigenvalues.  there are then three
!                        possible courses of action--
!
!                        1.  no eigenvectors are desired.  the process
!                            stops here.
!
!                        2.  all eigenvectors are desired.  eigrg1
!                            computes and returns these in packed
!                            form.  the user can then write his own
!                            fortran to unpack the eigenvectors or
!                            he can call eigvec to extract the
!                            eigenvector corresponding to any given
!                            eigenvalue.
!
!                        3.  only some selected eigenvectors are
!                            desired.  eigrg2 computes and returns
!                            these in packed form.  the user can then
!                            write his own fortran to unpack the
!                            eigenvectors or he can call eigvec to
!                            extract the eigenvector corresponding
!                            to any given eigenvalue.
!
! i/o                    if an error occurs, an error message will
!                        be printed.
!
! precision              single
!
! required library       uliber and q8qst4, which are loaded by
! files                  default on ncar's cray machines.
!
! language               fortran
!
! history                written by members of ncar's scientific
!                        computing division in the early 1970's
!
! algorithm              these subroutines are principally composed
!                        of subroutines obtained from eispack.
!
!                        the matrix is balanced and eigenvalues are
!                        isolated whenever possible.  the matrix is
!                        then reduced to upper hessenberg form (using
!                        stabilized elementary similarity
!                        transformations).
!
!                        if no eigenvectors are desired, the
!                        eigenvalues are found using the qr method.
!                        if all eigenvectors are desired, the
!                        stabilized elementary similarity trans-
!                        formations mentioned above are accumulated,
!                        then the eigenvalues and eigenvectors of the
!                        balanced matrix are found using the qr method.
!                        the eigenvectors of the original matrix are
!                        found from those of the balanced matrix by
!                        applying the back transformations used in
!                        balancing the matrix.  if some eigenvectors
!                        are desired, the eigenvalues are found using
!                        the qr method, and information on the
!                        transformations applied is saved for use
!                        by eigrg2 in finding the eigenvectors.
!
!                        the eigenvectors can be unpacked using eigvec.
!
! portability            american national standards institute
!                        standard fortran.  there are two machine
!                        constants defined and used in subroutines
!                        balanc, hqr, and hqr2.  these are radix
!                        and machep.
!***********************************************************************
!
! subroutine eigrg1 (nm,n,a,iflag,wr,wi,z,work1,work2,ierr)
!
! dimension of           a(nm,n),wr(n),wi(n),z(nm,1),work1(nm,n),
! arguments              work2(n*n+5*n+2)
!
! purpose                eigrg1 computes all the eigenvalues of a
!                        general matrix by the  qr  method.  it also
!                        can compute all the eigenvalues or it
!                        can compute the information necessary for
!                        subroutine eigrg2 to determine selected
!                        eigenvectors.
!
! usage                  call eigrg1 (nm,n,a,iflag,wr,wi,z,work1,work2,
!                                  ierr)
!
! arguments
!
! on input               nm
!                          the row (first) dimension in the calling
!                          program of the two-dimensional arrays a, z,
!                          and work1.
!
!                        n
!                          order of the matrix represented in array a.
!
!                        a
!                          a doubly-dimensioned array with row (first)
!                          dimension nm and column dimension n.  a
!                          contains the matrix of order  n  whose
!                          eigenvalues are to be determined.
!
!                        iflag
!                          an integer variable indicating the number of
!                          eigenvectors to be computed.
!
!                          iflag = 0       no eigenvectors to be
!                                          determined by eigrg1
!                          0 < iflag < n   iflag selected eigenvectors
!                                          are to be found using
!                                          subroutine eigrg2.
!                          iflag = n       all eigenvectors to be
!                                          determined by eigrg1
!
!                        work1
!                          a doubly-dimensioned work array with
!                          row (first) dimension nm and column dimension
!                          at least n.  eigrg1 copies  a  into  work1
!                          so that the original matrix is not
!                          destroyed.  the array  a  is not used
!                          again.  if the user does not mind if his
!                          original matrix is destroyed, he can
!                          call eigrg1 with  a  substituted for
!                          work1.  if eigrg2 is called, work1 must
!                          be input to eigrg2 exactly as it was
!                          output from eigrg1.
!
!                        work2
!                          a work array.  if iflag = 0 (no eigenvectors
!                          desired) or iflag = n (all eigenvectors
!                          desired), then work2 must be dimensioned at
!                          least 2*n+2.  if 0 < iflag < n (some
!                          eigenvectors desired), then work2 must be
!                          dimensioned at least n*n+5*n+2 .  in this
!                          case, work2 must be input to eigrg2
!                          exactly as it was output from eigrg1.
!
! on output              wr, wi
!                          arrays of dimension n containing the real and
!                          imaginary parts, respectively, of the
!                          eigenvalues.
!
!                        z
!                          if  iflag = 0 (no eigenvectors desired),
!                          then  z  is not used.  it may be input
!                          to eigrg1 as a single variable.  if
!                          iflag.gt.0 (some or all eigenvectors
!                          desired), then  z  is a doubly-dimensioned
!                          array.  the row (first) dimension in
!                          the calling program must be  nm  and the
!                          column dimension at least  n.  if
!                          0.lt.iflag.lt.n (some eigenvectors
!                          desired),  z  is used as a work array.
!                          if  iflag = n (all eigenvectors desired),
!                          the columns of  z  contain the real and
!                          imaginary parts of the eigenvectors.
!                          if the i-th eigenvalue is real, the
!                          i-th column of  z  contains its
!                          eigenvector.  since eigenvalues with
!                          nonzero imaginary parts occur in complex
!                          conjugate pairs, if the i-th and (i+1)th
!                          eigenvalues are such a pair, the the
!                          i-th and (i+1)th columns of  z  contain
!                          the real and imaginary parts of the
!                          eigenvector associated with the eigenvalue
!                          having positive imaginary part.
!
!                          the eigenvectors are unnormalized.
!                          subroutine eigvec can be used to extract
!                          the eigenvector corresponding to any
!                          eigenvalue.  if an error exit is made,
!                          none of the eigenvectors has been found.
!
!                        ierr
!                          is set to
!
!                          zero    for normal return.
!                          32+j    if the j-th eigenvalue has not been
!                                  determined after 30 iterations.
!
! special conditions     all eigenvalues are found unless an error
!                        message indicates otherwise.  if this occurs,
!                        no eigenvectors will be found, and
!                        subroutine eigrg2 should not be called to
!                        determine eigenvectors.
!
! algorithm              the matrix is balanced and eigenvalues
!                        are isolated whenever possible.  the matrix
!                        is then reduced to upper hessenberg form
!                        (using stabilized elementary similarity
!                        transformations).
!
!                        if no eigenvectors are desired, the
!                        eigenvalues are found using the  qr  method.
!                        if all eigenvectors are desired, the
!                        stabilized elementary similarity trans-
!                        formations mentioned above are accumulated,
!                        then the eigenvalues and eigenvectors of
!                        the balanced matrix are found using the  qr
!                        method.  the eigenvectors of the original
!                        matrix are found from those of the
!                        balanced matrix by applying the back
!                        transformations used in balancing the
!                        matrix.  if some eigenvectors are desired,
!                        the eigenvalues are found using the  qr
!                        method, and information on the transformations
!                        applied is saved for use by eigrg2 in
!                        finding the eigenvectors.
!
! accuracy               depends on the type of matrix
!
! timing                 no timing tests have been done.
! *******************************************************************
!
! subroutine eigvec (nm,n,wr,wi,z,select,k,zr,zi,ierr)
!
! dimension of           wr(n),wi(n),z(nm,m),select(n),zr(n),zi(n)
! arguments              where  m = n (n as described below) if
!                        eigenvectors were found in eigrg1, and
!                        m = mm (mm as described in the eigrg2
!                        documentation) if eigenvectors were found
!                        in eigrg2.
!
! purpose                the eigenvectors computed by eigrg1 and eigrg2
!                        are stored in packed form. eigvec extracts the
!                        eigenvector corresponding to any selected
!                        eigenvalue after the eigenvectors are computed
!                        by eigrg1 or eigrg2.
!
! usage                  call eigvec (nm,n,wr,wi,z,select,k,zr,zi,ierr)
!
! arguments
!
! on input               nm
!                          the row (first) dimension in the calling
!                          program of the two-dimensional array z.
!
!                        n
!                          the order of the matrix whose eigenvalues
!                          and eigenvectors were found by eigrg1 and/or
!                          eigrg2.
!
!                        wr,wi
!                          arrays of dimension n containing the real
!                          and imaginary parts, respectively, of the
!                          eigenvalues computed by eigrg1.
!
!                        z
!                          a doubly dimensioned array whose columns
!                          contain the real and imaginary parts of the
!                          eigenvectors.  the row (first) dimension in
!                          the calling program must be nm.  the column
!                          dimension must be at least m, where m=n if
!                          eigenvectors were found in eigrg1, and m=mm
!                          (a parameter for eigrg2) if eigenvectors
!                          were found in eigrg2.  a description of the
!                          packed form of the eigenvectors in z can be
!                          found in the write-ups for eigrg1 and eigrg2.
!
!                        select
!                          a logical array of dimension n indicating
!                          which eigenvectors were computed.  if
!                          eigenvectors were computed in eigrg1, select
!                          must contain n values of .true..  if
!                          eigenvectors were computed in eigrg2, select
!                          must not be altered after output from eigrg2.
!
!                        k
!                          an index indicating the position in arrays
!                          wr and wi of the eigenvalue whose eigenvector
!                          is to be found.  the eigenvector
!                          corresponding to wr(k), wi(k) will be
!                          returned.
!
! on output              zr,zi
!                          arrays of dimension n containing the real
!                          and imaginary parts, respectively, of the
!                          eigenvector corresponding to eigenvalue
!                          wr(k),wi(k).  if an error exit is made, zr
!                          and zi are set to zero.
!
!                        ierr
!                          is set to
!
!                          zero    for normal return
!                          33      if select(k) is .false.. in this case
!                                  the eigenvector corresponding to
!                                  wr(k),wi(k) has not been computed,
!                                  so eigvec cannot unpack it.
!
! algorithm              an index, kvec, for the eigenvector array
!                        is calculated by examining the
!                        eigenvector array and the select
!                        array from the first element
!                        to the k-th element.  the vector determined
!                        by the index kvec is then returned.
!
! accuracy               no approximations are involved, so the results
!                        are completely accurate.
!
! timing                 no timing tests have been performed.
! *******************************************************************
      subroutine eigrg1 (nm,n,a,iflag,wr,wi,z,work1,work2,ierr)
      real a(nm,n),wr(n),wi(n),work1(nm,n),work2(1),z(nm,1)
! the following call is for gathering statistics on library use at ncar
!     logical q8q4
!     save q8q4
!     data q8q4 /.true./
!     if (q8q4) then
!         call q8qst4('loclib','eigrg1','eigrg1','version 03')
!         q8q4 = .false.
!     endif
      do 50 i = 1,n
      do 50 j = 1,n
      work1(i,j) = a(i,j)
   50 continue
      call balanc(nm,n,work1,work2(n+1),work2(n+2),work2)
      call elmhes(nm,n,work2(n+1),work2(n+2),work1,work2(n+3))
!
!     check for number of eigenvectors desired.
!
      if (iflag .ne. 0) go to 100
!
!     no eigenvectors desired
!
      call hqr(nm,n,work2(n+1),work2(n+2),work1,wr,wi,ierr)
      if (ierr .ne. 0) go to 400
      return
!
!     check again for number of eigenvectors desired.
!
  100 if (iflag .ne. n) go to 200
!
!     all eigenvectors desired
!
      call eltran(nm,n,work2(n+1),work2(n+2),work1,work2(n+3),z)
      call hqr2(nm,n,work2(n+1),work2(n+2),work1,wr,wi,z,ierr)
      if (ierr .ne. 0) go to 400
      call balbak(nm,n,work2(n+1),work2(n+2),work2,n,z)
      return
!
!     some eigenvectors desired
!
  200 do 300 i = 1,n
         do 300 j = 1,n
      z(i,j) = work1(i,j)
  300 continue
      call hqr(nm,n,work2(n+1),work2(n+2),z,wr,wi,ierr)
      if (ierr .eq. 0) return
  400 ierr = iabs(ierr) + 32
!     call uliber(ierr,42h eigrg1 failed to compute all eigenvalues.,42)
      return
!
!
!
      end
      subroutine balanc(nm,n,a,low,igh,scale)
!
      integer i,j,k,l,m,n,jj,nm,igh,low,iexc
      real a(nm,n),scale(n)
      real c,f,g,r,s,b2,radix
!     real abs
      logical noconv
!
!     this subroutine is a translation of the algol procedure balance,
!     num. math. 13, 293-304(1969) by parlett and reinsch.
!     handbook for auto. comp., vol.ii-linear algebra, 315-326(1971).
!
!     this subroutine balances a real matrix and isolates
!     eigenvalues whenever possible.
!
!     on input-
!
!        nm must be set to the row dimension of two-dimensional
!          array parameters as declared in the calling program
!          dimension statement,
!
!        n is the order of the matrix,
!
!        a contains the input matrix to be balanced.
!
!     on output-
!
!        a contains the balanced matrix,
!
!        low and igh are two integers such that a(i,j)
!          is equal to zero if
!           (1) i is greater than j and
!           (2) j=1,...,low-1 or i=igh+1,...,n,
!
!        scale contains information determining the
!           permutations and scaling factors used.
!
!     suppose that the principal submatrix in rows low through igh
!     has been balanced, that p(j) denotes the index interchanged
!     with j during the permutation step, and that the elements
!     of the diagonal matrix used are denoted by d(i,j).  then
!        scale(j) = p(j),    for j = 1,...,low-1
!                 = d(j,j),      j = low,...,igh
!                 = p(j)         j = igh+1,...,n.
!     the order in which the interchanges are made is n to igh+1,
!     then 1 to low-1.
!
!     note that 1 is returned for igh if igh is zero formally.
!
!     the algol procedure exc contained in balance appears in
!     balanc  in line.  (note that the algol roles of identifiers
!     k,l have been reversed.)
!
!     questions and comments should be directed to b. s. garbow,
!     applied mathematics division, argonne national laboratory
!
!     ------------------------------------------------------------------
!
!     ********** radix is a machine dependent parameter specifying
!                the base of the machine floating point representation.
!
!                **********
      radix = 2.
!
      b2 = radix * radix
      k = 1
      l = n
      go to 100
!     ********** in-line procedure for row and
!                column exchange **********
   20 scale(m) = j
      if (j .eq. m) go to 50
!
      do 30 i = 1, l
         f = a(i,j)
         a(i,j) = a(i,m)
         a(i,m) = f
   30 continue
!
      do 40 i = k, n
         f = a(j,i)
         a(j,i) = a(m,i)
         a(m,i) = f
   40 continue
!
   50 go to (80,130), iexc
!     ********** search for rows isolating an eigenvalue
!                and push them down **********
   80 if (l .eq. 1) go to 280
      l = l - 1
!     ********** for j=l step -1 until 1 do -- **********
  100 do 120 jj = 1, l
         j = l + 1 - jj
!
         do 110 i = 1, l
            if (i .eq. j) go to 110
            if (a(j,i) .ne. 0.0) go to 120
  110    continue
!
         m = l
         iexc = 1
         go to 20
  120 continue
!
      go to 140
!     ********** search for columns isolating an eigenvalue
!                and push them left **********
  130 k = k + 1
!
  140 do 170 j = k, l
!
         do 150 i = k, l
            if (i .eq. j) go to 150
            if (a(i,j) .ne. 0.0) go to 170
  150    continue
!
         m = k
         iexc = 2
         go to 20
  170 continue
!     ********** now balance the submatrix in rows k to l **********
      do 180 i = k, l
  180 scale(i) = 1.0
!     ********** iterative loop for norm reduction **********
  190 noconv = .false.
!
      do 270 i = k, l
         c = 0.0
         r = 0.0
!
         do 200 j = k, l
            if (j .eq. i) go to 200
            c = c + abs(a(j,i))
            r = r + abs(a(i,j))
  200    continue
!
         g = r / radix
         f = 1.0
         s = c + r
  210    if (c .ge. g) go to 220
         f = f * radix
         c = c * b2
         go to 210
  220    g = r * radix
  230    if (c .lt. g) go to 240
         f = f / radix
         c = c / b2
         go to 230
!     ********** now balance **********
  240    if ((c + r) / f .ge. 0.95 * s) go to 270
         g = 1.0 / f
         scale(i) = scale(i) * f
         noconv = .true.
!
         do 250 j = k, n
  250    a(i,j) = a(i,j) * g
!
         do 260 j = 1, l
  260    a(j,i) = a(j,i) * f
!
  270 continue
!
      if (noconv) go to 190
!
  280 low = k
      igh = l
      return
!     ********** last card of balanc **********
!
!
!
      end
      subroutine elmhes(nm,n,low,igh,a,int)
!
      integer i,j,m,n,la,nm,igh,kp1,low,mm1,mp1
      real a(nm,n)
      real x,y
!     real abs
      integer int(igh)
!
!     this subroutine is a translation of the algol procedure elmhes,
!     num. math. 12, 349-368(1968) by martin and wilkinson.
!     handbook for auto. comp., vol.ii-linear algebra, 339-358(1971).
!
!     given a real general matrix, this subroutine
!     reduces a submatrix situated in rows and columns
!     low through igh to upper hessenberg form by
!     stabilized elementary similarity transformations.
!
!     on input-
!
!        nm must be set to the row dimension of two-dimensional
!          array parameters as declared in the calling program
!          dimension statement,
!
!        n is the order of the matrix,
!
!        low and igh are integers determined by the balancing
!          subroutine  balanc.  if  balanc  has not been used,
!          set low=1, igh=n,
!
!        a contains the input matrix.
!
!     on output-
!
!        a contains the hessenberg matrix.  the multipliers
!          which were used in the reduction are stored in the
!          remaining triangle under the hessenberg matrix,
!
!        int contains information on the rows and columns
!          interchanged in the reduction.
!          only elements low through igh are used.
!
!     questions and comments should be directed to b. s. garbow,
!     applied mathematics division, argonne national laboratory
!
!     ------------------------------------------------------------------
!
      la = igh - 1
      kp1 = low + 1
      if (la .lt. kp1) go to 200
!
      do 180 m = kp1, la
         mm1 = m - 1
         x = 0.0
         i = m
!
         do 100 j = m, igh
            if (abs(a(j,mm1)) .le. abs(x)) go to 100
            x = a(j,mm1)
            i = j
  100    continue
!
         int(m) = i
         if (i .eq. m) go to 130
!    ********** interchange rows and columns of a **********
         do 110 j = mm1, n
            y = a(i,j)
            a(i,j) = a(m,j)
            a(m,j) = y
  110    continue
!
         do 120 j = 1, igh
            y = a(j,i)
            a(j,i) = a(j,m)
            a(j,m) = y
  120    continue
!    ********** end interchange **********
  130    if (x .eq. 0.0) go to 180
         mp1 = m + 1
!
         do 160 i = mp1, igh
            y = a(i,mm1)
            if (y .eq. 0.0) go to 160
            y = y / x
            a(i,mm1) = y
!
            do 140 j = m, n
  140       a(i,j) = a(i,j) - y * a(m,j)
!
            do 150 j = 1, igh
  150       a(j,m) = a(j,m) + y * a(j,i)
!
  160    continue
!
  180 continue
!
  200 return
!    ********** last card of elmhes **********
!
!
!
      end
      subroutine hqr(nm,n,low,igh,h,wr,wi,ierr)
!
      integer i,j,k,l,m,n,en,ll,mm,na,nm,igh,its,low,mp2,enm2,ierr
      real h(nm,n),wr(n),wi(n)
      real p,q,r,s,t,w,x,y,zz,norm,machep
!     real sqrt,abs,sign
!     integer min0
      logical notlas
!
!     this subroutine is a translation of the algol procedure hqr,
!     num. math. 14, 219-231(1970) by martin, peters, and wilkinson.
!     handbook for auto. comp., vol.ii-linear algebra, 359-371(1971).
!
!     this subroutine finds the eigenvalues of a real
!     upper hessenberg matrix by the qr method.
!
!     on input-
!
!        nm must be set to the row dimension of two-dimensional
!          array parameters as declared in the calling program
!          dimension statement,
!
!        n is the order of the matrix,
!
!        low and igh are integers determined by the balancing
!          subroutine  balanc.  if  balanc  has not been used,
!          set low=1, igh=n,
!
!        h contains the upper hessenberg matrix.  information about
!          the transformations used in the reduction to hessenberg
!          form by  elmhes  or  orthes, if performed, is stored
!          in the remaining triangle under the hessenberg matrix.
!
!     on output-
!
!        h has been destroyed.  therefore, it must be saved
!          before calling  hqr  if subsequent calculation and
!          back transformation of eigenvectors is to be performed,
!
!        wr and wi contain the real and imaginary parts,
!          respectively, of the eigenvalues.  the eigenvalues
!          are unordered except that complex conjugate pairs
!          of values appear consecutively with the eigenvalue
!          having the positive imaginary part first.  if an
!          error exit is made, the eigenvalues should be correct
!          for indices ierr+1,...,n,
!
!        ierr is set to
!          zero       for normal return,
!          j          if the j-th eigenvalue has not been
!                     determined after 30 iterations.
!
!     questions and comments should be directed to b. s. garbow,
!     applied mathematics division, argonne national laboratory
!
!     ------------------------------------------------------------------
!
!     ********** machep is a machine dependent parameter specifying
!                the relative precision of floating point arithmetic.
!
!                **********
      machep = 2.**(-47)
!
      ierr = 0
      norm = 0.0
      k = 1
!     ********** store roots isolated by balanc
!                and compute matrix norm **********
      do 50 i = 1, n
!
         do 40 j = k, n
   40    norm = norm + abs(h(i,j))
!
         k = i
         if (i .ge. low .and. i .le. igh) go to 50
         wr(i) = h(i,i)
         wi(i) = 0.0
   50 continue
!
      en = igh
      t = 0.0
!     ********** search for next eigenvalues **********
   60 if (en .lt. low) go to 1001
      its = 0
      na = en - 1
      enm2 = na - 1
!     ********** look for single small sub-diagonal element
!                for l=en step -1 until low do -- **********
   70 do 80 ll = low, en
         l = en + low - ll
         if (l .eq. low) go to 100
         s = abs(h(l-1,l-1)) + abs(h(l,l))
         if (s .eq. 0.0) s = norm
         if (abs(h(l,l-1)) .le. machep * s) go to 100
   80 continue
!     ********** form shift **********
  100 x = h(en,en)
      if (l .eq. en) go to 270
      y = h(na,na)
      w = h(en,na) * h(na,en)
      if (l .eq. na) go to 280
      if (its .eq. 30) go to 1000
      if (its .ne. 10 .and. its .ne. 20) go to 130
!     ********** form exceptional shift **********
      t = t + x
!
      do 120 i = low, en
  120 h(i,i) = h(i,i) - x
!
      s = abs(h(en,na)) + abs(h(na,enm2))
      x = 0.75 * s
      y = x
      w = -0.4375 * s * s
  130 its = its + 1
!     ********** look for two consecutive small
!                sub-diagonal elements.
!                for m=en-2 step -1 until l do -- **********
      do 140 mm = l, enm2
         m = enm2 + l - mm
         zz = h(m,m)
         r = x - zz
         s = y - zz
         p = (r * s - w) / h(m+1,m) + h(m,m+1)
         q = h(m+1,m+1) - zz - r - s
         r = h(m+2,m+1)
         s = abs(p) + abs(q) + abs(r)
         p = p / s
         q = q / s
         r = r / s
         if (m .eq. l) go to 150
         if (abs(h(m,m-1)) * (abs(q) + abs(r)) .le. machep * abs(p) * (abs(h(m-1,m-1)) + abs(zz) + abs(h(m+1,m+1)))) go to 150
  140 continue
!
  150 mp2 = m + 2
!
      do 160 i = mp2, en
         h(i,i-2) = 0.0
         if (i .eq. mp2) go to 160
         h(i,i-3) = 0.0
  160 continue
!     ********** double qr step involving rows l to en and
!                columns m to en **********
      do 260 k = m, na
         notlas = k .ne. na
         if (k .eq. m) go to 170
         p = h(k,k-1)
         q = h(k+1,k-1)
         r = 0.0
         if (notlas) r = h(k+2,k-1)
         x = abs(p) + abs(q) + abs(r)
         if (x .eq. 0.0) go to 260
         p = p / x
         q = q / x
         r = r / x
  170    s = sign(sqrt(p*p+q*q+r*r),p)
         if (k .eq. m) go to 180
         h(k,k-1) = -s * x
         go to 190
  180    if (l .ne. m) h(k,k-1) = -h(k,k-1)
  190    p = p + s
         x = p / s
         y = q / s
         zz = r / s
         q = q / p
         r = r / p
!     ********** row modification **********
         do 210 j = k, en
            p = h(k,j) + q * h(k+1,j)
            if (.not. notlas) go to 200
            p = p + r * h(k+2,j)
            h(k+2,j) = h(k+2,j) - p * zz
  200       h(k+1,j) = h(k+1,j) - p * y
            h(k,j) = h(k,j) - p * x
  210    continue
!
         j = min0(en,k+3)
!     ********** column modification **********
         do 230 i = l, j
            p = x * h(i,k) + y * h(i,k+1)
            if (.not. notlas) go to 220
            p = p + zz * h(i,k+2)
            h(i,k+2) = h(i,k+2) - p * r
  220       h(i,k+1) = h(i,k+1) - p * q
            h(i,k) = h(i,k) - p
  230    continue
!
  260 continue
!
      go to 70
!     ********** one root found **********
  270 wr(en) = x + t
      wi(en) = 0.0
      en = na
      go to 60
!     ********** two roots found **********
  280 p = (y - x) / 2.0
      q = p * p + w
      zz = sqrt(abs(q))
      x = x + t
      if (q .lt. 0.0) go to 320
!     ********** real pair **********
      zz = p + sign(zz,p)
      wr(na) = x + zz
      wr(en) = wr(na)
      if (zz .ne. 0.0) wr(en) = x - w / zz
      wi(na) = 0.0
      wi(en) = 0.0
      go to 330
!     ********** complex pair **********
  320 wr(na) = x + p
      wr(en) = x + p
      wi(na) = zz
      wi(en) = -zz
  330 en = enm2
      go to 60
!     ********** set error -- no convergence to an
!                eigenvalue after 30 iterations **********
 1000 ierr = en
 1001 return
!     ********** last card of hqr **********
!
!
!
      end
      subroutine eltran(nm,n,low,igh,a,int,z)
!
      integer i,j,n,kl,mm,mp,nm,igh,low,mp1
      real a(nm,igh),z(nm,n)
      integer int(igh)
!
!     this subroutine is a translation of the algol procedure elmtrans,
!     num. math. 16, 181-204(1970) by peters and wilkinson.
!     handbook for auto. comp., vol.ii-linear algebra, 372-395(1971).
!
!     this subroutine accumulates the stabilized elementary
!     similarity transformations used in the reduction of a
!     real general matrix to upper hessenberg form by  elmhes.
!
!     on input-
!
!        nm must be set to the row dimension of two-dimensional
!          array parameters as declared in the calling program
!          dimension statement,
!
!        n is the order of the matrix,
!
!        low and igh are integers determined by the balancing
!          subroutine  balanc.  if  balanc  has not been used,
!          set low=1, igh=n,
!
!        a contains the multipliers which were used in the
!          reduction by  elmhes  in its lower triangle
!          below the subdiagonal,
!
!        int contains information on the rows and columns
!          interchanged in the reduction by  elmhes.
!          only elements low through igh are used.
!
!     on output-
!
!        z contains the transformation matrix produced in the
!          reduction by  elmhes.
!
!     questions and comments should be directed to b. s. garbow,
!     applied mathematics division, argonne national laboratory
!
!     ------------------------------------------------------------------
!
!     ********** initialize z to identity matrix **********
      do 80 i = 1, n
!
         do 60 j = 1, n
   60    z(i,j) = 0.0
!
         z(i,i) = 1.0
   80 continue
!
      kl = igh - low - 1
      if (kl .lt. 1) go to 200
!     ********** for mp=igh-1 step -1 until low+1 do -- **********
      do 140 mm = 1, kl
         mp = igh - mm
         mp1 = mp + 1
!
         do 100 i = mp1, igh
  100    z(i,mp) = a(i,mp-1)
!
         i = int(mp)
         if (i .eq. mp) go to 140
!
         do 130 j = mp, igh
            z(mp,j) = z(i,j)
            z(i,j) = 0.0
  130    continue
!
         z(i,mp) = 1.0
  140 continue
!
  200 return
!     ********** last card of eltran **********
!
!
!
      end
      subroutine hqr2(nm,n,low,igh,h,wr,wi,z,ierr)
!
      integer i,j,k,l,m,n,en,ii,jj,ll,mm,na,nm,nn,igh,its,low,mp2,enm2,ierr
      real h(nm,n),wr(n),wi(n),z(nm,n)
      real p,q,r,s,t,w,x,y,ra,sa,vi,vr,zz,norm,machep
!     real sqrt,abs,sign
!     integer min0
      logical notlas
      complex z3
!     complex cmplx
!     real real,aimag
!
!
!
!
!
!     this subroutine is a translation of the algol procedure hqr2,
!     num. math. 16, 181-204(1970) by peters and wilkinson.
!     handbook for auto. comp., vol.ii-linear algebra, 372-395(1971).
!
!     this subroutine finds the eigenvalues and eigenvectors
!     of a real upper hessenberg matrix by the qr method.  the
!     eigenvectors of a real general matrix can also be found
!     if  elmhes  and  eltran  or  orthes  and  ortran  have
!     been used to reduce this general matrix to hessenberg form
!     and to accumulate the similarity transformations.
!
!     on input-
!
!        nm must be set to the row dimension of two-dimensional
!          array parameters as declared in the calling program
!          dimension statement,
!
!        n is the order of the matrix,
!
!        low and igh are integers determined by the balancing
!          subroutine  balanc.  if  balanc  has not been used,
!          set low=1, igh=n,
!
!        h contains the upper hessenberg matrix,
!
!        z contains the transformation matrix produced by  eltran
!          after the reduction by  elmhes, or by  ortran  after the
!          reduction by  orthes, if performed.  if the eigenvectors
!          of the hessenberg matrix are desired, z must contain the
!          identity matrix.
!
!     on output-
!
!        h has been destroyed,
!
!        wr and wi contain the real and imaginary parts,
!          respectively, of the eigenvalues.  the eigenvalues
!          are unordered except that complex conjugate pairs
!          of values appear consecutively with the eigenvalue
!          having the positive imaginary part first.  if an
!          error exit is made, the eigenvalues should be correct
!          for indices ierr+1,...,n,
!
!        z contains the real and imaginary parts of the eigenvectors.
!          if the i-th eigenvalue is real, the i-th column of z
!          contains its eigenvector.  if the i-th eigenvalue is complex
!          with positive imaginary part, the i-th and (i+1)-th
!          columns of z contain the real and imaginary parts of its
!          eigenvector.  the eigenvectors are unnormalized.  if an
!          error exit is made, none of the eigenvectors has been found,
!
!        ierr is set to
!          zero       for normal return,
!          j          if the j-th eigenvalue has not been
!                     determined after 30 iterations.
!
!     arithmetic is real except for the replacement of the algol
!     procedure cdiv by complex division.
!
!     questions and comments should be directed to b. s. garbow,
!     applied mathematics division, argonne national laboratory
!
!     ------------------------------------------------------------------
!
!     ********** machep is a machine dependent parameter specifying
!                the relative precision of floating point arithmetic.
!
!                **********
      machep = 2.**(-47)
!
      ierr = 0
      norm = 0.0
      k = 1
!     ********** store roots isolated by balanc
!                and compute matrix norm **********
      do 50 i = 1, n
!
         do 40 j = k, n
   40    norm = norm + abs(h(i,j))
!
         k = i
         if (i .ge. low .and. i .le. igh) go to 50
         wr(i) = h(i,i)
         wi(i) = 0.0
   50 continue
!
      en = igh
      t = 0.0
!     ********** search for next eigenvalues **********
   60 if (en .lt. low) go to 340
      its = 0
      na = en - 1
      enm2 = na - 1
!     ********** look for single small sub-diagonal element
!                for l=en step -1 until low do -- **********
   70 do 80 ll = low, en
         l = en + low - ll
         if (l .eq. low) go to 100
         s = abs(h(l-1,l-1)) + abs(h(l,l))
         if (s .eq. 0.0) s = norm
         if (abs(h(l,l-1)) .le. machep * s) go to 100
   80 continue
!     ********** form shift **********
  100 x = h(en,en)
      if (l .eq. en) go to 270
      y = h(na,na)
      w = h(en,na) * h(na,en)
      if (l .eq. na) go to 280
      if (its .eq. 30) go to 1000
      if (its .ne. 10 .and. its .ne. 20) go to 130
!     ********** form exceptional shift **********
      t = t + x
!
      do 120 i = low, en
  120 h(i,i) = h(i,i) - x
!
      s = abs(h(en,na)) + abs(h(na,enm2))
      x = 0.75 * s
      y = x
      w = -0.4375 * s * s
  130 its = its + 1
!     ********** look for two consecutive small
!                sub-diagonal elements.
!                for m=en-2 step -1 until l do -- **********
      do 140 mm = l, enm2
         m = enm2 + l - mm
         zz = h(m,m)
         r = x - zz
         s = y - zz
         p = (r * s - w) / h(m+1,m) + h(m,m+1)
         q = h(m+1,m+1) - zz - r - s
         r = h(m+2,m+1)
         s = abs(p) + abs(q) + abs(r)
         p = p / s
         q = q / s
         r = r / s
         if (m .eq. l) go to 150
         if (abs(h(m,m-1)) * (abs(q) + abs(r)) .le. machep * abs(p) * (abs(h(m-1,m-1)) + abs(zz) + abs(h(m+1,m+1)))) go to 150
  140 continue
!
  150 mp2 = m + 2
!
      do 160 i = mp2, en
         h(i,i-2) = 0.0
         if (i .eq. mp2) go to 160
         h(i,i-3) = 0.0
  160 continue
!     ********** double qr step involving rows l to en and
!                columns m to en **********
      do 260 k = m, na
         notlas = k .ne. na
         if (k .eq. m) go to 170
         p = h(k,k-1)
         q = h(k+1,k-1)
         r = 0.0
         if (notlas) r = h(k+2,k-1)
         x = abs(p) + abs(q) + abs(r)
         if (x .eq. 0.0) go to 260
         p = p / x
         q = q / x
         r = r / x
  170    s = sign(sqrt(p*p+q*q+r*r),p)
         if (k .eq. m) go to 180
         h(k,k-1) = -s * x
         go to 190
  180    if (l .ne. m) h(k,k-1) = -h(k,k-1)
  190    p = p + s
         x = p / s
         y = q / s
         zz = r / s
         q = q / p
         r = r / p
!     ********** row modification **********
         do 210 j = k, n
            p = h(k,j) + q * h(k+1,j)
            if (.not. notlas) go to 200
            p = p + r * h(k+2,j)
            h(k+2,j) = h(k+2,j) - p * zz
  200       h(k+1,j) = h(k+1,j) - p * y
            h(k,j) = h(k,j) - p * x
  210    continue
!
         j = min0(en,k+3)
!     ********** column modification **********
         do 230 i = 1, j
            p = x * h(i,k) + y * h(i,k+1)
            if (.not. notlas) go to 220
            p = p + zz * h(i,k+2)
            h(i,k+2) = h(i,k+2) - p * r
  220       h(i,k+1) = h(i,k+1) - p * q
            h(i,k) = h(i,k) - p
  230    continue
!     ********** accumulate transformations **********
         do 250 i = low, igh
            p = x * z(i,k) + y * z(i,k+1)
            if (.not. notlas) go to 240
            p = p + zz * z(i,k+2)
            z(i,k+2) = z(i,k+2) - p * r
  240       z(i,k+1) = z(i,k+1) - p * q
            z(i,k) = z(i,k) - p
  250    continue
!
  260 continue
!
      go to 70
!     ********** one root found **********
  270 h(en,en) = x + t
      wr(en) = h(en,en)
      wi(en) = 0.0
      en = na
      go to 60
!     ********** two roots found **********
  280 p = (y - x) / 2.0
      q = p * p + w
      zz = sqrt(abs(q))
      h(en,en) = x + t
      x = h(en,en)
      h(na,na) = y + t
      if (q .lt. 0.0) go to 320
!     ********** real pair **********
      zz = p + sign(zz,p)
      wr(na) = x + zz
      wr(en) = wr(na)
      if (zz .ne. 0.0) wr(en) = x - w / zz
      wi(na) = 0.0
      wi(en) = 0.0
      x = h(en,na)
      r = sqrt(x*x+zz*zz)
      p = x / r
      q = zz / r
!     ********** row modification **********
      do 290 j = na, n
         zz = h(na,j)
         h(na,j) = q * zz + p * h(en,j)
         h(en,j) = q * h(en,j) - p * zz
  290 continue
!     ********** column modification **********
      do 300 i = 1, en
         zz = h(i,na)
         h(i,na) = q * zz + p * h(i,en)
         h(i,en) = q * h(i,en) - p * zz
  300 continue
!     ********** accumulate transformations **********
      do 310 i = low, igh
         zz = z(i,na)
         z(i,na) = q * zz + p * z(i,en)
         z(i,en) = q * z(i,en) - p * zz
  310 continue
!
      go to 330
!     ********** complex pair **********
  320 wr(na) = x + p
      wr(en) = x + p
      wi(na) = zz
      wi(en) = -zz
  330 en = enm2
      go to 60
!     ********** all roots found.  backsubstitute to find
!                vectors of upper triangular form **********
  340 if (norm .eq. 0.0) go to 1001
!     ********** for en=n step -1 until 1 do -- **********
      do 800 nn = 1, n
         en = n + 1 - nn
         p = wr(en)
         q = wi(en)
         na = en - 1
         if (q) 710, 600, 800
!     ********** real vector **********
  600    m = en
         h(en,en) = 1.0
         if (na .eq. 0) go to 800
!     ********** for i=en-1 step -1 until 1 do -- **********
         do 700 ii = 1, na
            i = en - ii
            w = h(i,i) - p
            r = h(i,en)
            if (m .gt. na) go to 620
!
            do 610 j = m, na
  610       r = r + h(i,j) * h(j,en)
!
  620       if (wi(i) .ge. 0.0) go to 630
            zz = w
            s = r
            go to 700
  630       m = i
            if (wi(i) .ne. 0.0) go to 640
            t = w
            if (w .eq. 0.0) t = machep * norm
            h(i,en) = -r / t
            go to 700
!     ********** solve real equations **********
  640       x = h(i,i+1)
            y = h(i+1,i)
            q = (wr(i) - p) * (wr(i) - p) + wi(i) * wi(i)
            t = (x * s - zz * r) / q
            h(i,en) = t
            if (abs(x) .le. abs(zz)) go to 650
            h(i+1,en) = (-r - w * t) / x
            go to 700
  650       h(i+1,en) = (-s - y * t) / zz
  700    continue
!     ********** end real vector **********
         go to 800
!     ********** complex vector **********
  710    m = na
!     ********** last vector component chosen imaginary so that
!                eigenvector matrix is triangular **********
         if (abs(h(en,na)) .le. abs(h(na,en))) go to 720
         h(na,na) = q / h(en,na)
         h(na,en) = -(h(en,en) - p) / h(en,na)
         go to 730
  720    z3 = cmplx(0.0,-h(na,en)) / cmplx(h(na,na)-p,q)
         h(na,na) = real(z3)
         h(na,en) = aimag(z3)
  730    h(en,na) = 0.0
         h(en,en) = 1.0
         enm2 = na - 1
         if (enm2 .eq. 0) go to 800
!     ********** for i=en-2 step -1 until 1 do -- **********
         do 790 ii = 1, enm2
            i = na - ii
            w = h(i,i) - p
            ra = 0.0
            sa = h(i,en)
!
            do 760 j = m, na
               ra = ra + h(i,j) * h(j,na)
               sa = sa + h(i,j) * h(j,en)
  760       continue
!
            if (wi(i) .ge. 0.0) go to 770
            zz = w
            r = ra
            s = sa
            go to 790
  770       m = i
            if (wi(i) .ne. 0.0) go to 780
            z3 = cmplx(-ra,-sa) / cmplx(w,q)
            h(i,na) = real(z3)
            h(i,en) = aimag(z3)
            go to 790
!     ********** solve complex equations **********
  780       x = h(i,i+1)
            y = h(i+1,i)
            vr = (wr(i) - p) * (wr(i) - p) + wi(i) * wi(i) - q * q
            vi = (wr(i) - p) * 2.0 * q
            if (vr .eq. 0.0 .and. vi .eq. 0.0) vr = machep * norm * (abs(w) + abs(q) + abs(x) + abs(y) + abs(zz))
            z3 = cmplx(x*r-zz*ra+q*sa,x*s-zz*sa-q*ra) / cmplx(vr,vi)
            h(i,na) = real(z3)
            h(i,en) = aimag(z3)
            if (abs(x) .le. abs(zz) + abs(q)) go to 785
            h(i+1,na) = (-ra - w * h(i,na) + q * h(i,en)) / x
            h(i+1,en) = (-sa - w * h(i,en) - q * h(i,na)) / x
            go to 790
  785       z3 = cmplx(-r-y*h(i,na),-s-y*h(i,en)) / cmplx(zz,q)
            h(i+1,na) = real(z3)
            h(i+1,en) = aimag(z3)
  790    continue
!     ********** end complex vector **********
  800 continue
!     ********** end back substitution.
!                vectors of isolated roots **********
      do 840 i = 1, n
         if (i .ge. low .and. i .le. igh) go to 840
!
         do 820 j = i, n
  820    z(i,j) = h(i,j)
!
  840 continue
!     ********** multiply by transformation matrix to give
!                vectors of original full matrix.
!                for j=n step -1 until low do -- **********
      do 880 jj = low, n
         j = n + low - jj
         m = min0(j,igh)
!
         do 880 i = low, igh
            zz = 0.0
!
            do 860 k = low, m
  860       zz = zz + z(i,k) * h(k,j)
!
            z(i,j) = zz
  880 continue
!
      go to 1001
!     ********** set error -- no convergence to an
!                eigenvalue after 30 iterations **********
 1000 ierr = en
 1001 return
!     ********** last card of hqr2 **********
!
!
!
      end
      subroutine balbak(nm,n,low,igh,scale,m,z)
!
      integer i,j,k,m,n,ii,nm,igh,low
      real scale(n),z(nm,m)
      real s
!
!     this subroutine is a translation of the algol procedure balbak,
!     num. math. 13, 293-304(1969) by parlett and reinsch.
!     handbook for auto. comp., vol.ii-linear algebra, 315-326(1971).
!
!     this subroutine forms the eigenvectors of a real general
!     matrix by back transforming those of the corresponding
!     balanced matrix determined by  balanc.
!
!     on input-
!
!        nm must be set to the row dimension of two-dimensional
!          array parameters as declared in the calling program
!          dimension statement,
!
!        n is the order of the matrix,
!
!        low and igh are integers determined by  balanc,
!
!        scale contains information determining the permutations
!          and scaling factors used by  balanc,
!
!        m is the number of columns of z to be back transformed,
!
!        z contains the real and imaginary parts of the eigen-
!          vectors to be back transformed in its first m columns.
!
!     on output-
!
!        z contains the real and imaginary parts of the
!          transformed eigenvectors in its first m columns.
!
!     questions and comments should be directed to b. s. garbow,
!     applied mathematics division, argonne national laboratory
!
!     ------------------------------------------------------------------
!
      if (m .eq. 0) go to 200
      if (igh .eq. low) go to 120
!
      do 110 i = low, igh
         s = scale(i)
!     ********** left hand eigenvectors are back transformed
!                if the foregoing statement is replaced by
!                s=1.0/scale(i). **********
         do 100 j = 1, m
  100    z(i,j) = z(i,j) * s
!
  110 continue
!     ********- for i=low-1 step -1 until 1,
!               igh+1 step 1 until n do -- **********
  120 do 140 ii = 1, n
         i = ii
         if (i .ge. low .and. i .le. igh) go to 140
         if (i .lt. low) i = low - ii
         k = scale(i)
         if (k .eq. i) go to 140
!
         do 130 j = 1, m
            s = z(i,j)
            z(i,j) = z(k,j)
            z(k,j) = s
  130    continue
!
  140 continue
!
  200 return
!     ********** last card of balbak **********
      end
      subroutine eigvec (nm,n,wr,wi,z,select,k,zr,zi,ierr)
!
      integer  nm,n,k,ierr
      real wr(n),wi(n),z(nm,1),          zr(n),zi(n)
      logical select(n)
!
! this routine assumes all conjugate complex pairs of eigenvalues
! occur adjacent to one another in arrays wr and wi.  this is the order
! on output from rg1 and rg2.
!
! the following call is for gathering statistics on library use at ncar
      logical q8q4
      save q8q4
      data q8q4 /.true./
      if (q8q4) then
!         call q8qst4('loclib','eigrg1','eigvec','version 03')
          q8q4 = .false.
      endif
      ierr = 0
!
! if the k-th eigenvalue is not one of those selected, return with
! error flag set.
!
      if (select(k)) go to 10
      ierr = 33
!     call uliber(ierr,31h error in eigenvector selection, 31)
      do 5 i = 1,n
      zr(i) = 0.
      zi(i) = 0.
    5 continue
      return
   10 continue
!
! find the index of the appropriate eigenvector.
!
      sign = 1.
      kvec = 0
      do 30 i = 1,k
      if (.not. select(i)) go to 30
      if (wi(i) .eq. 0.) go to 20
!
! a complex eigenvector is stored in two columns.
!
      kvec = kvec + 2
      sign = 1.
      if (i .eq. 1) go to 30
!
! if this is the complex conjugate of a previously stored eigenvector,
! it is not stored.  reduce the count.
!
      if (wi(i) .ne. (-wi(i-1)) .or. .not. select(i-1)) go to 30
      kvec = kvec - 2
      sign = -1.
      go to 30
!
! a real eigenvector is stored in one column.
!
   20 kvec = kvec + 1
   30 continue
!
! store the eigenvector in zr and zi.
!
      if (wi(k) .eq. 0.) go to 50
!
! complex eigenvector.
!
      do 40 i = 1,n
      zr(i) = z(i,kvec-1)
      zi(i) = z(i,kvec)*sign
   40 continue
      return
   50 continue
!
! real eigenvector
!
      do 60 i = 1,n
      zr(i) = z(i,kvec)
      zi(i) = 0.
   60 continue
      return
!
! revision history---
!
! august 1977            modified the documentation to match that in
!                        volume 1 of the nssl manuals.
!
! january 1978     moved revision history to appear before
!                  the final end card and removed comment cards
!                  between subroutines
!-----------------------------------------------------------------------
      end
