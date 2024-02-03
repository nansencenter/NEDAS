c package eigrg1         documentation for individual routines
c                        follows the general package information
c
c latest revision        january 1985
c
c purpose                eigrg1 is a package of two subroutines, eigrg1
c                        and eigvec.  these subroutines, along with a
c                        separate subroutine, eigrg2, compute all the
c                        eigenvalues of a real general matrix.
c                        optionally, all eigenvectors can be computed
c                        or some selected eigenvectors can be computed.
c
c usage                  the user first calls eigrg1 which computes
c                        all eigenvalues.  there are then three
c                        possible courses of action--
c
c                        1.  no eigenvectors are desired.  the process
c                            stops here.
c
c                        2.  all eigenvectors are desired.  eigrg1
c                            computes and returns these in packed
c                            form.  the user can then write his own
c                            fortran to unpack the eigenvectors or
c                            he can call eigvec to extract the
c                            eigenvector corresponding to any given
c                            eigenvalue.
c
c                        3.  only some selected eigenvectors are
c                            desired.  eigrg2 computes and returns
c                            these in packed form.  the user can then
c                            write his own fortran to unpack the
c                            eigenvectors or he can call eigvec to
c                            extract the eigenvector corresponding
c                            to any given eigenvalue.
c
c i/o                    if an error occurs, an error message will
c                        be printed.
c
c precision              single
c
c required library       uliber and q8qst4, which are loaded by
c files                  default on ncar's cray machines.
c
c language               fortran
c
c history                written by members of ncar's scientific
c                        computing division in the early 1970's
c
c algorithm              these subroutines are principally composed
c                        of subroutines obtained from eispack.
c
c                        the matrix is balanced and eigenvalues are
c                        isolated whenever possible.  the matrix is
c                        then reduced to upper hessenberg form (using
c                        stabilized elementary similarity
c                        transformations).
c
c                        if no eigenvectors are desired, the
c                        eigenvalues are found using the qr method.
c                        if all eigenvectors are desired, the
c                        stabilized elementary similarity trans-
c                        formations mentioned above are accumulated,
c                        then the eigenvalues and eigenvectors of the
c                        balanced matrix are found using the qr method.
c                        the eigenvectors of the original matrix are
c                        found from those of the balanced matrix by
c                        applying the back transformations used in
c                        balancing the matrix.  if some eigenvectors
c                        are desired, the eigenvalues are found using
c                        the qr method, and information on the
c                        transformations applied is saved for use
c                        by eigrg2 in finding the eigenvectors.
c
c                        the eigenvectors can be unpacked using eigvec.
c
c portability            american national standards institute
c                        standard fortran.  there are two machine
c                        constants defined and used in subroutines
c                        balanc, hqr, and hqr2.  these are radix
c                        and machep.
c***********************************************************************
c
c subroutine eigrg1 (nm,n,a,iflag,wr,wi,z,work1,work2,ierr)
c
c dimension of           a(nm,n),wr(n),wi(n),z(nm,1),work1(nm,n),
c arguments              work2(n*n+5*n+2)
c
c purpose                eigrg1 computes all the eigenvalues of a
c                        general matrix by the  qr  method.  it also
c                        can compute all the eigenvalues or it
c                        can compute the information necessary for
c                        subroutine eigrg2 to determine selected
c                        eigenvectors.
c
c usage                  call eigrg1 (nm,n,a,iflag,wr,wi,z,work1,work2,
c                                  ierr)
c
c arguments
c
c on input               nm
c                          the row (first) dimension in the calling
c                          program of the two-dimensional arrays a, z,
c                          and work1.
c
c                        n
c                          order of the matrix represented in array a.
c
c                        a
c                          a doubly-dimensioned array with row (first)
c                          dimension nm and column dimension n.  a
c                          contains the matrix of order  n  whose
c                          eigenvalues are to be determined.
c
c                        iflag
c                          an integer variable indicating the number of
c                          eigenvectors to be computed.
c
c                          iflag = 0       no eigenvectors to be
c                                          determined by eigrg1
c                          0 < iflag < n   iflag selected eigenvectors
c                                          are to be found using
c                                          subroutine eigrg2.
c                          iflag = n       all eigenvectors to be
c                                          determined by eigrg1
c
c                        work1
c                          a doubly-dimensioned work array with
c                          row (first) dimension nm and column dimension
c                          at least n.  eigrg1 copies  a  into  work1
c                          so that the original matrix is not
c                          destroyed.  the array  a  is not used
c                          again.  if the user does not mind if his
c                          original matrix is destroyed, he can
c                          call eigrg1 with  a  substituted for
c                          work1.  if eigrg2 is called, work1 must
c                          be input to eigrg2 exactly as it was
c                          output from eigrg1.
c
c                        work2
c                          a work array.  if iflag = 0 (no eigenvectors
c                          desired) or iflag = n (all eigenvectors
c                          desired), then work2 must be dimensioned at
c                          least 2*n+2.  if 0 < iflag < n (some
c                          eigenvectors desired), then work2 must be
c                          dimensioned at least n*n+5*n+2 .  in this
c                          case, work2 must be input to eigrg2
c                          exactly as it was output from eigrg1.
c
c on output              wr, wi
c                          arrays of dimension n containing the real and
c                          imaginary parts, respectively, of the
c                          eigenvalues.
c
c                        z
c                          if  iflag = 0 (no eigenvectors desired),
c                          then  z  is not used.  it may be input
c                          to eigrg1 as a single variable.  if
c                          iflag.gt.0 (some or all eigenvectors
c                          desired), then  z  is a doubly-dimensioned
c                          array.  the row (first) dimension in
c                          the calling program must be  nm  and the
c                          column dimension at least  n.  if
c                          0.lt.iflag.lt.n (some eigenvectors
c                          desired),  z  is used as a work array.
c                          if  iflag = n (all eigenvectors desired),
c                          the columns of  z  contain the real and
c                          imaginary parts of the eigenvectors.
c                          if the i-th eigenvalue is real, the
c                          i-th column of  z  contains its
c                          eigenvector.  since eigenvalues with
c                          nonzero imaginary parts occur in complex
c                          conjugate pairs, if the i-th and (i+1)th
c                          eigenvalues are such a pair, the the
c                          i-th and (i+1)th columns of  z  contain
c                          the real and imaginary parts of the
c                          eigenvector associated with the eigenvalue
c                          having positive imaginary part.
c
c                          the eigenvectors are unnormalized.
c                          subroutine eigvec can be used to extract
c                          the eigenvector corresponding to any
c                          eigenvalue.  if an error exit is made,
c                          none of the eigenvectors has been found.
c
c                        ierr
c                          is set to
c
c                          zero    for normal return.
c                          32+j    if the j-th eigenvalue has not been
c                                  determined after 30 iterations.
c
c special conditions     all eigenvalues are found unless an error
c                        message indicates otherwise.  if this occurs,
c                        no eigenvectors will be found, and
c                        subroutine eigrg2 should not be called to
c                        determine eigenvectors.
c
c algorithm              the matrix is balanced and eigenvalues
c                        are isolated whenever possible.  the matrix
c                        is then reduced to upper hessenberg form
c                        (using stabilized elementary similarity
c                        transformations).
c
c                        if no eigenvectors are desired, the
c                        eigenvalues are found using the  qr  method.
c                        if all eigenvectors are desired, the
c                        stabilized elementary similarity trans-
c                        formations mentioned above are accumulated,
c                        then the eigenvalues and eigenvectors of
c                        the balanced matrix are found using the  qr
c                        method.  the eigenvectors of the original
c                        matrix are found from those of the
c                        balanced matrix by applying the back
c                        transformations used in balancing the
c                        matrix.  if some eigenvectors are desired,
c                        the eigenvalues are found using the  qr
c                        method, and information on the transformations
c                        applied is saved for use by eigrg2 in
c                        finding the eigenvectors.
c
c accuracy               depends on the type of matrix
c
c timing                 no timing tests have been done.
c *******************************************************************
c
c subroutine eigvec (nm,n,wr,wi,z,select,k,zr,zi,ierr)
c
c dimension of           wr(n),wi(n),z(nm,m),select(n),zr(n),zi(n)
c arguments              where  m = n (n as described below) if
c                        eigenvectors were found in eigrg1, and
c                        m = mm (mm as described in the eigrg2
c                        documentation) if eigenvectors were found
c                        in eigrg2.
c
c purpose                the eigenvectors computed by eigrg1 and eigrg2
c                        are stored in packed form. eigvec extracts the
c                        eigenvector corresponding to any selected
c                        eigenvalue after the eigenvectors are computed
c                        by eigrg1 or eigrg2.
c
c usage                  call eigvec (nm,n,wr,wi,z,select,k,zr,zi,ierr)
c
c arguments
c
c on input               nm
c                          the row (first) dimension in the calling
c                          program of the two-dimensional array z.
c
c                        n
c                          the order of the matrix whose eigenvalues
c                          and eigenvectors were found by eigrg1 and/or
c                          eigrg2.
c
c                        wr,wi
c                          arrays of dimension n containing the real
c                          and imaginary parts, respectively, of the
c                          eigenvalues computed by eigrg1.
c
c                        z
c                          a doubly dimensioned array whose columns
c                          contain the real and imaginary parts of the
c                          eigenvectors.  the row (first) dimension in
c                          the calling program must be nm.  the column
c                          dimension must be at least m, where m=n if
c                          eigenvectors were found in eigrg1, and m=mm
c                          (a parameter for eigrg2) if eigenvectors
c                          were found in eigrg2.  a description of the
c                          packed form of the eigenvectors in z can be
c                          found in the write-ups for eigrg1 and eigrg2.
c
c                        select
c                          a logical array of dimension n indicating
c                          which eigenvectors were computed.  if
c                          eigenvectors were computed in eigrg1, select
c                          must contain n values of .true..  if
c                          eigenvectors were computed in eigrg2, select
c                          must not be altered after output from eigrg2.
c
c                        k
c                          an index indicating the position in arrays
c                          wr and wi of the eigenvalue whose eigenvector
c                          is to be found.  the eigenvector
c                          corresponding to wr(k), wi(k) will be
c                          returned.
c
c on output              zr,zi
c                          arrays of dimension n containing the real
c                          and imaginary parts, respectively, of the
c                          eigenvector corresponding to eigenvalue
c                          wr(k),wi(k).  if an error exit is made, zr
c                          and zi are set to zero.
c
c                        ierr
c                          is set to
c
c                          zero    for normal return
c                          33      if select(k) is .false.. in this case
c                                  the eigenvector corresponding to
c                                  wr(k),wi(k) has not been computed,
c                                  so eigvec cannot unpack it.
c
c algorithm              an index, kvec, for the eigenvector array
c                        is calculated by examining the
c                        eigenvector array and the select
c                        array from the first element
c                        to the k-th element.  the vector determined
c                        by the index kvec is then returned.
c
c accuracy               no approximations are involved, so the results
c                        are completely accurate.
c
c timing                 no timing tests have been performed.
c *******************************************************************
      subroutine eigrg1 (nm,n,a,iflag,wr,wi,z,work1,work2,ierr)
      real a(nm,n),wr(n),wi(n),work1(nm,n),work2(1),z(nm,1)
c the following call is for gathering statistics on library use at ncar
c     logical q8q4
c     save q8q4
c     data q8q4 /.true./
c     if (q8q4) then
c         call q8qst4('loclib','eigrg1','eigrg1','version 03')
c         q8q4 = .false.
c     endif
      do 50 i = 1,n
      do 50 j = 1,n
      work1(i,j) = a(i,j)
   50 continue
      call balanc(nm,n,work1,work2(n+1),work2(n+2),work2)
      call elmhes(nm,n,work2(n+1),work2(n+2),work1,work2(n+3))
c
c     check for number of eigenvectors desired.
c
      if (iflag .ne. 0) go to 100
c
c     no eigenvectors desired
c
      call hqr(nm,n,work2(n+1),work2(n+2),work1,wr,wi,ierr)
      if (ierr .ne. 0) go to 400
      return
c
c     check again for number of eigenvectors desired.
c
  100 if (iflag .ne. n) go to 200
c
c     all eigenvectors desired
c
      call eltran(nm,n,work2(n+1),work2(n+2),work1,work2(n+3),z)
      call hqr2(nm,n,work2(n+1),work2(n+2),work1,wr,wi,z,ierr)
      if (ierr .ne. 0) go to 400
      call balbak(nm,n,work2(n+1),work2(n+2),work2,n,z)
      return
c
c     some eigenvectors desired
c
  200 do 300 i = 1,n
         do 300 j = 1,n
      z(i,j) = work1(i,j)
  300 continue
      call hqr(nm,n,work2(n+1),work2(n+2),z,wr,wi,ierr)
      if (ierr .eq. 0) return
  400 ierr = iabs(ierr) + 32
c     call uliber(ierr,42h eigrg1 failed to compute all eigenvalues.,42)
      return
c
c
c
      end
      subroutine balanc(nm,n,a,low,igh,scale)
c
      integer i,j,k,l,m,n,jj,nm,igh,low,iexc
      real a(nm,n),scale(n)
      real c,f,g,r,s,b2,radix
c     real abs
      logical noconv
c
c     this subroutine is a translation of the algol procedure balance,
c     num. math. 13, 293-304(1969) by parlett and reinsch.
c     handbook for auto. comp., vol.ii-linear algebra, 315-326(1971).
c
c     this subroutine balances a real matrix and isolates
c     eigenvalues whenever possible.
c
c     on input-
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement,
c
c        n is the order of the matrix,
c
c        a contains the input matrix to be balanced.
c
c     on output-
c
c        a contains the balanced matrix,
c
c        low and igh are two integers such that a(i,j)
c          is equal to zero if
c           (1) i is greater than j and
c           (2) j=1,...,low-1 or i=igh+1,...,n,
c
c        scale contains information determining the
c           permutations and scaling factors used.
c
c     suppose that the principal submatrix in rows low through igh
c     has been balanced, that p(j) denotes the index interchanged
c     with j during the permutation step, and that the elements
c     of the diagonal matrix used are denoted by d(i,j).  then
c        scale(j) = p(j),    for j = 1,...,low-1
c                 = d(j,j),      j = low,...,igh
c                 = p(j)         j = igh+1,...,n.
c     the order in which the interchanges are made is n to igh+1,
c     then 1 to low-1.
c
c     note that 1 is returned for igh if igh is zero formally.
c
c     the algol procedure exc contained in balance appears in
c     balanc  in line.  (note that the algol roles of identifiers
c     k,l have been reversed.)
c
c     questions and comments should be directed to b. s. garbow,
c     applied mathematics division, argonne national laboratory
c
c     ------------------------------------------------------------------
c
c     ********** radix is a machine dependent parameter specifying
c                the base of the machine floating point representation.
c
c                **********
      radix = 2.
c
      b2 = radix * radix
      k = 1
      l = n
      go to 100
c     ********** in-line procedure for row and
c                column exchange **********
   20 scale(m) = j
      if (j .eq. m) go to 50
c
      do 30 i = 1, l
         f = a(i,j)
         a(i,j) = a(i,m)
         a(i,m) = f
   30 continue
c
      do 40 i = k, n
         f = a(j,i)
         a(j,i) = a(m,i)
         a(m,i) = f
   40 continue
c
   50 go to (80,130), iexc
c     ********** search for rows isolating an eigenvalue
c                and push them down **********
   80 if (l .eq. 1) go to 280
      l = l - 1
c     ********** for j=l step -1 until 1 do -- **********
  100 do 120 jj = 1, l
         j = l + 1 - jj
c
         do 110 i = 1, l
            if (i .eq. j) go to 110
            if (a(j,i) .ne. 0.0) go to 120
  110    continue
c
         m = l
         iexc = 1
         go to 20
  120 continue
c
      go to 140
c     ********** search for columns isolating an eigenvalue
c                and push them left **********
  130 k = k + 1
c
  140 do 170 j = k, l
c
         do 150 i = k, l
            if (i .eq. j) go to 150
            if (a(i,j) .ne. 0.0) go to 170
  150    continue
c
         m = k
         iexc = 2
         go to 20
  170 continue
c     ********** now balance the submatrix in rows k to l **********
      do 180 i = k, l
  180 scale(i) = 1.0
c     ********** iterative loop for norm reduction **********
  190 noconv = .false.
c
      do 270 i = k, l
         c = 0.0
         r = 0.0
c
         do 200 j = k, l
            if (j .eq. i) go to 200
            c = c + abs(a(j,i))
            r = r + abs(a(i,j))
  200    continue
c
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
c     ********** now balance **********
  240    if ((c + r) / f .ge. 0.95 * s) go to 270
         g = 1.0 / f
         scale(i) = scale(i) * f
         noconv = .true.
c
         do 250 j = k, n
  250    a(i,j) = a(i,j) * g
c
         do 260 j = 1, l
  260    a(j,i) = a(j,i) * f
c
  270 continue
c
      if (noconv) go to 190
c
  280 low = k
      igh = l
      return
c     ********** last card of balanc **********
c
c
c
      end
      subroutine elmhes(nm,n,low,igh,a,int)
c
      integer i,j,m,n,la,nm,igh,kp1,low,mm1,mp1
      real a(nm,n)
      real x,y
c     real abs
      integer int(igh)
c
c     this subroutine is a translation of the algol procedure elmhes,
c     num. math. 12, 349-368(1968) by martin and wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 339-358(1971).
c
c     given a real general matrix, this subroutine
c     reduces a submatrix situated in rows and columns
c     low through igh to upper hessenberg form by
c     stabilized elementary similarity transformations.
c
c     on input-
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement,
c
c        n is the order of the matrix,
c
c        low and igh are integers determined by the balancing
c          subroutine  balanc.  if  balanc  has not been used,
c          set low=1, igh=n,
c
c        a contains the input matrix.
c
c     on output-
c
c        a contains the hessenberg matrix.  the multipliers
c          which were used in the reduction are stored in the
c          remaining triangle under the hessenberg matrix,
c
c        int contains information on the rows and columns
c          interchanged in the reduction.
c          only elements low through igh are used.
c
c     questions and comments should be directed to b. s. garbow,
c     applied mathematics division, argonne national laboratory
c
c     ------------------------------------------------------------------
c
      la = igh - 1
      kp1 = low + 1
      if (la .lt. kp1) go to 200
c
      do 180 m = kp1, la
         mm1 = m - 1
         x = 0.0
         i = m
c
         do 100 j = m, igh
            if (abs(a(j,mm1)) .le. abs(x)) go to 100
            x = a(j,mm1)
            i = j
  100    continue
c
         int(m) = i
         if (i .eq. m) go to 130
c    ********** interchange rows and columns of a **********
         do 110 j = mm1, n
            y = a(i,j)
            a(i,j) = a(m,j)
            a(m,j) = y
  110    continue
c
         do 120 j = 1, igh
            y = a(j,i)
            a(j,i) = a(j,m)
            a(j,m) = y
  120    continue
c    ********** end interchange **********
  130    if (x .eq. 0.0) go to 180
         mp1 = m + 1
c
         do 160 i = mp1, igh
            y = a(i,mm1)
            if (y .eq. 0.0) go to 160
            y = y / x
            a(i,mm1) = y
c
            do 140 j = m, n
  140       a(i,j) = a(i,j) - y * a(m,j)
c
            do 150 j = 1, igh
  150       a(j,m) = a(j,m) + y * a(j,i)
c
  160    continue
c
  180 continue
c
  200 return
c    ********** last card of elmhes **********
c
c
c
      end
      subroutine hqr(nm,n,low,igh,h,wr,wi,ierr)
c
      integer i,j,k,l,m,n,en,ll,mm,na,nm,igh,its,low,mp2,enm2,ierr
      real h(nm,n),wr(n),wi(n)
      real p,q,r,s,t,w,x,y,zz,norm,machep
c     real sqrt,abs,sign
c     integer min0
      logical notlas
c
c     this subroutine is a translation of the algol procedure hqr,
c     num. math. 14, 219-231(1970) by martin, peters, and wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 359-371(1971).
c
c     this subroutine finds the eigenvalues of a real
c     upper hessenberg matrix by the qr method.
c
c     on input-
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement,
c
c        n is the order of the matrix,
c
c        low and igh are integers determined by the balancing
c          subroutine  balanc.  if  balanc  has not been used,
c          set low=1, igh=n,
c
c        h contains the upper hessenberg matrix.  information about
c          the transformations used in the reduction to hessenberg
c          form by  elmhes  or  orthes, if performed, is stored
c          in the remaining triangle under the hessenberg matrix.
c
c     on output-
c
c        h has been destroyed.  therefore, it must be saved
c          before calling  hqr  if subsequent calculation and
c          back transformation of eigenvectors is to be performed,
c
c        wr and wi contain the real and imaginary parts,
c          respectively, of the eigenvalues.  the eigenvalues
c          are unordered except that complex conjugate pairs
c          of values appear consecutively with the eigenvalue
c          having the positive imaginary part first.  if an
c          error exit is made, the eigenvalues should be correct
c          for indices ierr+1,...,n,
c
c        ierr is set to
c          zero       for normal return,
c          j          if the j-th eigenvalue has not been
c                     determined after 30 iterations.
c
c     questions and comments should be directed to b. s. garbow,
c     applied mathematics division, argonne national laboratory
c
c     ------------------------------------------------------------------
c
c     ********** machep is a machine dependent parameter specifying
c                the relative precision of floating point arithmetic.
c
c                **********
      machep = 2.**(-47)
c
      ierr = 0
      norm = 0.0
      k = 1
c     ********** store roots isolated by balanc
c                and compute matrix norm **********
      do 50 i = 1, n
c
         do 40 j = k, n
   40    norm = norm + abs(h(i,j))
c
         k = i
         if (i .ge. low .and. i .le. igh) go to 50
         wr(i) = h(i,i)
         wi(i) = 0.0
   50 continue
c
      en = igh
      t = 0.0
c     ********** search for next eigenvalues **********
   60 if (en .lt. low) go to 1001
      its = 0
      na = en - 1
      enm2 = na - 1
c     ********** look for single small sub-diagonal element
c                for l=en step -1 until low do -- **********
   70 do 80 ll = low, en
         l = en + low - ll
         if (l .eq. low) go to 100
         s = abs(h(l-1,l-1)) + abs(h(l,l))
         if (s .eq. 0.0) s = norm
         if (abs(h(l,l-1)) .le. machep * s) go to 100
   80 continue
c     ********** form shift **********
  100 x = h(en,en)
      if (l .eq. en) go to 270
      y = h(na,na)
      w = h(en,na) * h(na,en)
      if (l .eq. na) go to 280
      if (its .eq. 30) go to 1000
      if (its .ne. 10 .and. its .ne. 20) go to 130
c     ********** form exceptional shift **********
      t = t + x
c
      do 120 i = low, en
  120 h(i,i) = h(i,i) - x
c
      s = abs(h(en,na)) + abs(h(na,enm2))
      x = 0.75 * s
      y = x
      w = -0.4375 * s * s
  130 its = its + 1
c     ********** look for two consecutive small
c                sub-diagonal elements.
c                for m=en-2 step -1 until l do -- **********
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
         if (abs(h(m,m-1)) * (abs(q) + abs(r)) .le. machep * abs(p)
     1    * (abs(h(m-1,m-1)) + abs(zz) + abs(h(m+1,m+1)))) go to 150
  140 continue
c
  150 mp2 = m + 2
c
      do 160 i = mp2, en
         h(i,i-2) = 0.0
         if (i .eq. mp2) go to 160
         h(i,i-3) = 0.0
  160 continue
c     ********** double qr step involving rows l to en and
c                columns m to en **********
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
c     ********** row modification **********
         do 210 j = k, en
            p = h(k,j) + q * h(k+1,j)
            if (.not. notlas) go to 200
            p = p + r * h(k+2,j)
            h(k+2,j) = h(k+2,j) - p * zz
  200       h(k+1,j) = h(k+1,j) - p * y
            h(k,j) = h(k,j) - p * x
  210    continue
c
         j = min0(en,k+3)
c     ********** column modification **********
         do 230 i = l, j
            p = x * h(i,k) + y * h(i,k+1)
            if (.not. notlas) go to 220
            p = p + zz * h(i,k+2)
            h(i,k+2) = h(i,k+2) - p * r
  220       h(i,k+1) = h(i,k+1) - p * q
            h(i,k) = h(i,k) - p
  230    continue
c
  260 continue
c
      go to 70
c     ********** one root found **********
  270 wr(en) = x + t
      wi(en) = 0.0
      en = na
      go to 60
c     ********** two roots found **********
  280 p = (y - x) / 2.0
      q = p * p + w
      zz = sqrt(abs(q))
      x = x + t
      if (q .lt. 0.0) go to 320
c     ********** real pair **********
      zz = p + sign(zz,p)
      wr(na) = x + zz
      wr(en) = wr(na)
      if (zz .ne. 0.0) wr(en) = x - w / zz
      wi(na) = 0.0
      wi(en) = 0.0
      go to 330
c     ********** complex pair **********
  320 wr(na) = x + p
      wr(en) = x + p
      wi(na) = zz
      wi(en) = -zz
  330 en = enm2
      go to 60
c     ********** set error -- no convergence to an
c                eigenvalue after 30 iterations **********
 1000 ierr = en
 1001 return
c     ********** last card of hqr **********
c
c
c
      end
      subroutine eltran(nm,n,low,igh,a,int,z)
c
      integer i,j,n,kl,mm,mp,nm,igh,low,mp1
      real a(nm,igh),z(nm,n)
      integer int(igh)
c
c     this subroutine is a translation of the algol procedure elmtrans,
c     num. math. 16, 181-204(1970) by peters and wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 372-395(1971).
c
c     this subroutine accumulates the stabilized elementary
c     similarity transformations used in the reduction of a
c     real general matrix to upper hessenberg form by  elmhes.
c
c     on input-
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement,
c
c        n is the order of the matrix,
c
c        low and igh are integers determined by the balancing
c          subroutine  balanc.  if  balanc  has not been used,
c          set low=1, igh=n,
c
c        a contains the multipliers which were used in the
c          reduction by  elmhes  in its lower triangle
c          below the subdiagonal,
c
c        int contains information on the rows and columns
c          interchanged in the reduction by  elmhes.
c          only elements low through igh are used.
c
c     on output-
c
c        z contains the transformation matrix produced in the
c          reduction by  elmhes.
c
c     questions and comments should be directed to b. s. garbow,
c     applied mathematics division, argonne national laboratory
c
c     ------------------------------------------------------------------
c
c     ********** initialize z to identity matrix **********
      do 80 i = 1, n
c
         do 60 j = 1, n
   60    z(i,j) = 0.0
c
         z(i,i) = 1.0
   80 continue
c
      kl = igh - low - 1
      if (kl .lt. 1) go to 200
c     ********** for mp=igh-1 step -1 until low+1 do -- **********
      do 140 mm = 1, kl
         mp = igh - mm
         mp1 = mp + 1
c
         do 100 i = mp1, igh
  100    z(i,mp) = a(i,mp-1)
c
         i = int(mp)
         if (i .eq. mp) go to 140
c
         do 130 j = mp, igh
            z(mp,j) = z(i,j)
            z(i,j) = 0.0
  130    continue
c
         z(i,mp) = 1.0
  140 continue
c
  200 return
c     ********** last card of eltran **********
c
c
c
      end
      subroutine hqr2(nm,n,low,igh,h,wr,wi,z,ierr)
c
      integer i,j,k,l,m,n,en,ii,jj,ll,mm,na,nm,nn,
     1        igh,its,low,mp2,enm2,ierr
      real h(nm,n),wr(n),wi(n),z(nm,n)
      real p,q,r,s,t,w,x,y,ra,sa,vi,vr,zz,norm,machep
c     real sqrt,abs,sign
c     integer min0
      logical notlas
      complex z3
c     complex cmplx
c     real real,aimag
c
c
c
c
c
c     this subroutine is a translation of the algol procedure hqr2,
c     num. math. 16, 181-204(1970) by peters and wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 372-395(1971).
c
c     this subroutine finds the eigenvalues and eigenvectors
c     of a real upper hessenberg matrix by the qr method.  the
c     eigenvectors of a real general matrix can also be found
c     if  elmhes  and  eltran  or  orthes  and  ortran  have
c     been used to reduce this general matrix to hessenberg form
c     and to accumulate the similarity transformations.
c
c     on input-
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement,
c
c        n is the order of the matrix,
c
c        low and igh are integers determined by the balancing
c          subroutine  balanc.  if  balanc  has not been used,
c          set low=1, igh=n,
c
c        h contains the upper hessenberg matrix,
c
c        z contains the transformation matrix produced by  eltran
c          after the reduction by  elmhes, or by  ortran  after the
c          reduction by  orthes, if performed.  if the eigenvectors
c          of the hessenberg matrix are desired, z must contain the
c          identity matrix.
c
c     on output-
c
c        h has been destroyed,
c
c        wr and wi contain the real and imaginary parts,
c          respectively, of the eigenvalues.  the eigenvalues
c          are unordered except that complex conjugate pairs
c          of values appear consecutively with the eigenvalue
c          having the positive imaginary part first.  if an
c          error exit is made, the eigenvalues should be correct
c          for indices ierr+1,...,n,
c
c        z contains the real and imaginary parts of the eigenvectors.
c          if the i-th eigenvalue is real, the i-th column of z
c          contains its eigenvector.  if the i-th eigenvalue is complex
c          with positive imaginary part, the i-th and (i+1)-th
c          columns of z contain the real and imaginary parts of its
c          eigenvector.  the eigenvectors are unnormalized.  if an
c          error exit is made, none of the eigenvectors has been found,
c
c        ierr is set to
c          zero       for normal return,
c          j          if the j-th eigenvalue has not been
c                     determined after 30 iterations.
c
c     arithmetic is real except for the replacement of the algol
c     procedure cdiv by complex division.
c
c     questions and comments should be directed to b. s. garbow,
c     applied mathematics division, argonne national laboratory
c
c     ------------------------------------------------------------------
c
c     ********** machep is a machine dependent parameter specifying
c                the relative precision of floating point arithmetic.
c
c                **********
      machep = 2.**(-47)
c
      ierr = 0
      norm = 0.0
      k = 1
c     ********** store roots isolated by balanc
c                and compute matrix norm **********
      do 50 i = 1, n
c
         do 40 j = k, n
   40    norm = norm + abs(h(i,j))
c
         k = i
         if (i .ge. low .and. i .le. igh) go to 50
         wr(i) = h(i,i)
         wi(i) = 0.0
   50 continue
c
      en = igh
      t = 0.0
c     ********** search for next eigenvalues **********
   60 if (en .lt. low) go to 340
      its = 0
      na = en - 1
      enm2 = na - 1
c     ********** look for single small sub-diagonal element
c                for l=en step -1 until low do -- **********
   70 do 80 ll = low, en
         l = en + low - ll
         if (l .eq. low) go to 100
         s = abs(h(l-1,l-1)) + abs(h(l,l))
         if (s .eq. 0.0) s = norm
         if (abs(h(l,l-1)) .le. machep * s) go to 100
   80 continue
c     ********** form shift **********
  100 x = h(en,en)
      if (l .eq. en) go to 270
      y = h(na,na)
      w = h(en,na) * h(na,en)
      if (l .eq. na) go to 280
      if (its .eq. 30) go to 1000
      if (its .ne. 10 .and. its .ne. 20) go to 130
c     ********** form exceptional shift **********
      t = t + x
c
      do 120 i = low, en
  120 h(i,i) = h(i,i) - x
c
      s = abs(h(en,na)) + abs(h(na,enm2))
      x = 0.75 * s
      y = x
      w = -0.4375 * s * s
  130 its = its + 1
c     ********** look for two consecutive small
c                sub-diagonal elements.
c                for m=en-2 step -1 until l do -- **********
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
         if (abs(h(m,m-1)) * (abs(q) + abs(r)) .le. machep * abs(p)
     1    * (abs(h(m-1,m-1)) + abs(zz) + abs(h(m+1,m+1)))) go to 150
  140 continue
c
  150 mp2 = m + 2
c
      do 160 i = mp2, en
         h(i,i-2) = 0.0
         if (i .eq. mp2) go to 160
         h(i,i-3) = 0.0
  160 continue
c     ********** double qr step involving rows l to en and
c                columns m to en **********
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
c     ********** row modification **********
         do 210 j = k, n
            p = h(k,j) + q * h(k+1,j)
            if (.not. notlas) go to 200
            p = p + r * h(k+2,j)
            h(k+2,j) = h(k+2,j) - p * zz
  200       h(k+1,j) = h(k+1,j) - p * y
            h(k,j) = h(k,j) - p * x
  210    continue
c
         j = min0(en,k+3)
c     ********** column modification **********
         do 230 i = 1, j
            p = x * h(i,k) + y * h(i,k+1)
            if (.not. notlas) go to 220
            p = p + zz * h(i,k+2)
            h(i,k+2) = h(i,k+2) - p * r
  220       h(i,k+1) = h(i,k+1) - p * q
            h(i,k) = h(i,k) - p
  230    continue
c     ********** accumulate transformations **********
         do 250 i = low, igh
            p = x * z(i,k) + y * z(i,k+1)
            if (.not. notlas) go to 240
            p = p + zz * z(i,k+2)
            z(i,k+2) = z(i,k+2) - p * r
  240       z(i,k+1) = z(i,k+1) - p * q
            z(i,k) = z(i,k) - p
  250    continue
c
  260 continue
c
      go to 70
c     ********** one root found **********
  270 h(en,en) = x + t
      wr(en) = h(en,en)
      wi(en) = 0.0
      en = na
      go to 60
c     ********** two roots found **********
  280 p = (y - x) / 2.0
      q = p * p + w
      zz = sqrt(abs(q))
      h(en,en) = x + t
      x = h(en,en)
      h(na,na) = y + t
      if (q .lt. 0.0) go to 320
c     ********** real pair **********
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
c     ********** row modification **********
      do 290 j = na, n
         zz = h(na,j)
         h(na,j) = q * zz + p * h(en,j)
         h(en,j) = q * h(en,j) - p * zz
  290 continue
c     ********** column modification **********
      do 300 i = 1, en
         zz = h(i,na)
         h(i,na) = q * zz + p * h(i,en)
         h(i,en) = q * h(i,en) - p * zz
  300 continue
c     ********** accumulate transformations **********
      do 310 i = low, igh
         zz = z(i,na)
         z(i,na) = q * zz + p * z(i,en)
         z(i,en) = q * z(i,en) - p * zz
  310 continue
c
      go to 330
c     ********** complex pair **********
  320 wr(na) = x + p
      wr(en) = x + p
      wi(na) = zz
      wi(en) = -zz
  330 en = enm2
      go to 60
c     ********** all roots found.  backsubstitute to find
c                vectors of upper triangular form **********
  340 if (norm .eq. 0.0) go to 1001
c     ********** for en=n step -1 until 1 do -- **********
      do 800 nn = 1, n
         en = n + 1 - nn
         p = wr(en)
         q = wi(en)
         na = en - 1
         if (q) 710, 600, 800
c     ********** real vector **********
  600    m = en
         h(en,en) = 1.0
         if (na .eq. 0) go to 800
c     ********** for i=en-1 step -1 until 1 do -- **********
         do 700 ii = 1, na
            i = en - ii
            w = h(i,i) - p
            r = h(i,en)
            if (m .gt. na) go to 620
c
            do 610 j = m, na
  610       r = r + h(i,j) * h(j,en)
c
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
c     ********** solve real equations **********
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
c     ********** end real vector **********
         go to 800
c     ********** complex vector **********
  710    m = na
c     ********** last vector component chosen imaginary so that
c                eigenvector matrix is triangular **********
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
c     ********** for i=en-2 step -1 until 1 do -- **********
         do 790 ii = 1, enm2
            i = na - ii
            w = h(i,i) - p
            ra = 0.0
            sa = h(i,en)
c
            do 760 j = m, na
               ra = ra + h(i,j) * h(j,na)
               sa = sa + h(i,j) * h(j,en)
  760       continue
c
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
c     ********** solve complex equations **********
  780       x = h(i,i+1)
            y = h(i+1,i)
            vr = (wr(i) - p) * (wr(i) - p) + wi(i) * wi(i) - q * q
            vi = (wr(i) - p) * 2.0 * q
            if (vr .eq. 0.0 .and. vi .eq. 0.0) vr = machep * norm
     1       * (abs(w) + abs(q) + abs(x) + abs(y) + abs(zz))
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
c     ********** end complex vector **********
  800 continue
c     ********** end back substitution.
c                vectors of isolated roots **********
      do 840 i = 1, n
         if (i .ge. low .and. i .le. igh) go to 840
c
         do 820 j = i, n
  820    z(i,j) = h(i,j)
c
  840 continue
c     ********** multiply by transformation matrix to give
c                vectors of original full matrix.
c                for j=n step -1 until low do -- **********
      do 880 jj = low, n
         j = n + low - jj
         m = min0(j,igh)
c
         do 880 i = low, igh
            zz = 0.0
c
            do 860 k = low, m
  860       zz = zz + z(i,k) * h(k,j)
c
            z(i,j) = zz
  880 continue
c
      go to 1001
c     ********** set error -- no convergence to an
c                eigenvalue after 30 iterations **********
 1000 ierr = en
 1001 return
c     ********** last card of hqr2 **********
c
c
c
      end
      subroutine balbak(nm,n,low,igh,scale,m,z)
c
      integer i,j,k,m,n,ii,nm,igh,low
      real scale(n),z(nm,m)
      real s
c
c     this subroutine is a translation of the algol procedure balbak,
c     num. math. 13, 293-304(1969) by parlett and reinsch.
c     handbook for auto. comp., vol.ii-linear algebra, 315-326(1971).
c
c     this subroutine forms the eigenvectors of a real general
c     matrix by back transforming those of the corresponding
c     balanced matrix determined by  balanc.
c
c     on input-
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement,
c
c        n is the order of the matrix,
c
c        low and igh are integers determined by  balanc,
c
c        scale contains information determining the permutations
c          and scaling factors used by  balanc,
c
c        m is the number of columns of z to be back transformed,
c
c        z contains the real and imaginary parts of the eigen-
c          vectors to be back transformed in its first m columns.
c
c     on output-
c
c        z contains the real and imaginary parts of the
c          transformed eigenvectors in its first m columns.
c
c     questions and comments should be directed to b. s. garbow,
c     applied mathematics division, argonne national laboratory
c
c     ------------------------------------------------------------------
c
      if (m .eq. 0) go to 200
      if (igh .eq. low) go to 120
c
      do 110 i = low, igh
         s = scale(i)
c     ********** left hand eigenvectors are back transformed
c                if the foregoing statement is replaced by
c                s=1.0/scale(i). **********
         do 100 j = 1, m
  100    z(i,j) = z(i,j) * s
c
  110 continue
c     ********- for i=low-1 step -1 until 1,
c               igh+1 step 1 until n do -- **********
  120 do 140 ii = 1, n
         i = ii
         if (i .ge. low .and. i .le. igh) go to 140
         if (i .lt. low) i = low - ii
         k = scale(i)
         if (k .eq. i) go to 140
c
         do 130 j = 1, m
            s = z(i,j)
            z(i,j) = z(k,j)
            z(k,j) = s
  130    continue
c
  140 continue
c
  200 return
c     ********** last card of balbak **********
      end
      subroutine eigvec (nm,n,wr,wi,z,select,k,zr,zi,ierr)
c
      integer  nm,n,k,ierr
      real wr(n),wi(n),z(nm,1),          zr(n),zi(n)
      logical select(n)
c
c this routine assumes all conjugate complex pairs of eigenvalues
c occur adjacent to one another in arrays wr and wi.  this is the order
c on output from rg1 and rg2.
c
c the following call is for gathering statistics on library use at ncar
      logical q8q4
      save q8q4
      data q8q4 /.true./
      if (q8q4) then
c         call q8qst4('loclib','eigrg1','eigvec','version 03')
          q8q4 = .false.
      endif
      ierr = 0
c
c if the k-th eigenvalue is not one of those selected, return with
c error flag set.
c
      if (select(k)) go to 10
      ierr = 33
c     call uliber(ierr,31h error in eigenvector selection, 31)
      do 5 i = 1,n
      zr(i) = 0.
      zi(i) = 0.
    5 continue
      return
   10 continue
c
c find the index of the appropriate eigenvector.
c
      sign = 1.
      kvec = 0
      do 30 i = 1,k
      if (.not. select(i)) go to 30
      if (wi(i) .eq. 0.) go to 20
c
c a complex eigenvector is stored in two columns.
c
      kvec = kvec + 2
      sign = 1.
      if (i .eq. 1) go to 30
c
c if this is the complex conjugate of a previously stored eigenvector,
c it is not stored.  reduce the count.
c
      if (wi(i) .ne. (-wi(i-1)) .or. .not. select(i-1)) go to 30
      kvec = kvec - 2
      sign = -1.
      go to 30
c
c a real eigenvector is stored in one column.
c
   20 kvec = kvec + 1
   30 continue
c
c store the eigenvector in zr and zi.
c
      if (wi(k) .eq. 0.) go to 50
c
c complex eigenvector.
c
      do 40 i = 1,n
      zr(i) = z(i,kvec-1)
      zi(i) = z(i,kvec)*sign
   40 continue
      return
   50 continue
c
c real eigenvector
c
      do 60 i = 1,n
      zr(i) = z(i,kvec)
      zi(i) = 0.
   60 continue
      return
c
c revision history---
c
c august 1977            modified the documentation to match that in
c                        volume 1 of the nssl manuals.
c
c january 1978     moved revision history to appear before
c                  the final end card and removed comment cards
c                  between subroutines
c-----------------------------------------------------------------------
      end
