program rd_wr_binary

!
! read and average (if needed) morphological parameters, 
! and write in the binary form needed by WPS
!
   implicit none

   integer :: istatus
   character (len=256) :: name_land,name_ufrac,name_urb_par

   real, allocatable, dimension(:,:) :: xlat_o,xlong_o,ufrac_o
   real, allocatable, dimension(:,:) :: xlat_i,xlong_i,ufrac_i
   real, allocatable, dimension(:,:) :: landuse_i
   real, allocatable, dimension(:,:) :: landuse_o,rarray
   
   integer :: nx           ! x-dimension of the array
   integer :: ny           ! y-dimension of the array
   integer :: nz           ! z-dimension of the array
   integer :: isigned      ! 0=unsigned data, 1=signed data
   integer :: endian       ! 0=big endian, 1=little endian
   real :: scalefactor     ! value to divide array elements by before truncation to integers
   integer :: wordsize     ! number of bytes to use for each array element
   real :: latmin,longmin,latmax,longmax,dx,dy,latk,lonk,laty,lonx,pi,umax,a1,a2,a3
   real*8 xx,yy,xlat,xlong
   real :: dlat, dlon
   real :: conv
   double precision a,b,c,d,rt,dist,dmin
   real xlongmax,utot_i,utot_o
   integer :: np,ip,ipmin,ix,iy,iz,i,j,iip,iizone,ispher,iutm,imin,jmin
   integer nurbm,icmax,imax,nzu,nzmean
   integer igsize

! The following must be set before compiling
 
   character*70 aa
! size of the array with the data of the city!CHANGE nx and ny 
   parameter (nx=212,ny=216,nz=1,nurbm=15,nzu=15)
   integer icount(nurbm),itot,itot_urb
   real hmean_i(nx,ny),lambda_b_i(nx,ny),lambda_p_i(nx,ny),lambda_u_i(nx,ny),hdist_i(nx,ny,nzu)
   real hmean_o(nx,ny),lambda_b_o(nx,ny),lambda_p_o(nx,ny),lambda_u_o(nx,ny),hdist_o(nx,ny,nzu)
   real wmean(nx,ny),frc_u(nx,ny)
   real urb_param(nx,ny,132)
   real wrk(nx)
   real dz_u,alpha,h5
   integer ih5,ic
   real w,xmin,ymin,dlim,res
    real xlatst,xlonst,xlat_bar,xlong_bar
    integer ixmin,iymin,iystmin,ixstmin
   real htot
   character*35 names_hist(15)
   
! define the parameters for classes
! check and adapt if needed



   name_land='landuse_urban' 
   name_ufrac='ufrac'
   name_urb_par='urb_param'

! Did you extract the data in UTM or in WGS84?
! flag input data. If iutm=1 input data in UTM after extraction, if iutm=0 input data in WGS84 (lat long)
   iutm=1

! resolution of input in UTM (meters)

   res=300.


! Implement coordinates of the domain regarding the Region Of Interest
!!!!ADAPT!!!!!!!!!!!
! If iutm=1 change the iizone regarding the UTM zone of the domain
!!!!!ADAPT HERE !! adapt to your city!!!!!  
  if(iutm.eq.1)then
   iizone=31
   ispher=15

! Here we compute the lat, long coordinates of the lower left corner of the input domain   
  
   xx=421166.1701 ! x UTM coordinates of the lower left
   yy=5372682.242 ! y UTM coordinates of the lower left

! store the values of these coordinates in xmin and ymin

   xmin=xx
   ymin=yy

! note that utm2ll needs double precision inputs, and this is why xx, and yy are dfined as real*8

  call utm2ll(xx,yy,xlat,xlong,conv,iizone,ispher)

   latmin=xlat/3600.
   longmin=-xlong/3600.


! Here we compute the lat, long coordinates of the upper right corner of the input domain 

   xx=484466.1701 ! x UTM coordinates of the upper right
   yy=5437182.242 ! y UTM coordinates of the upper right

  call utm2ll(xx,yy,xlat,xlong,conv,iizone,ispher)
   
   latmax=xlat/3600.   
   longmax=-xlong/3600.

   write(*,*)'min',latmin,longmin
   write(*,*)'max',latmax,longmax

  endif

!! isigned = 0
!! endian = 0
!! wordsize = 1
!! scalefactor = 1.0

!   isigned = 1
!   endian = 1
!   wordsize = 4
!   scalefactor = 0.01

! Do not change anything here

    isigned = 1
    endian = 1
    wordsize = 1
    scalefactor = 1

   allocate(ufrac_o(nx,ny))
   allocate(xlat_o(nx,ny))
   allocate(xlong_o(nx,ny))
   allocate(landuse_o(nx,ny))
   allocate(rarray(nx,ny))

   ufrac_o=0.
   landuse_o=0

   allocate(xlat_i(nx,ny))
   allocate(xlong_i(nx,ny))
   allocate(ufrac_i(nx,ny))
   allocate(landuse_i(nx,ny))
   
    dx=(longmax-longmin)/nx
    dy=(latmax-latmin)/ny 
  
   write(*,*)'the values below must be introduced in the file index'
   write(*,*)'dx=',dx,'dy=',dy
   rt=6371000.
   pi=3.1415926535
   xlongmax=-100000000000000000.

!!!!!read the morphological data
! change this
   
   open(unit=20,file='LambdaP_average_1kmx1km.txt',status='old')
   do i=1,nx
    do j=ny,1,-1
    read(20,*)lambda_p_i(i,j)
    enddo
   enddo
   close(20)
!
   open(unit=20,file='Lambdab_average_1kmx1km.txt',status='old')
   do i=1,nx
    do j=ny,1,-1
    read(20,*)lambda_b_i(i,j)
    enddo
   enddo
   close(20)
!
   open(unit=20,file='MeanBuildingHeight_average_1kmx1km.txt',status='old')
   do i=1,nx
    do j=ny,1,-1
    read(20,*)hmean_i(i,j)
    enddo
   enddo
   close(20)
!
   open(unit=20,file='LambdaU_average_1kmx1km.txt',status='old')
   do i=1,nx
    do j=ny,1,-1
    read(20,*)lambda_u_i (i,j)
    enddo
   enddo
   close(20)
 
   open(unit=20,file='LCZ_average_1kmx1km.txt',status='old')
   do i=1,nx
    do j=ny,1,-1
    read(20,*)landuse_i (i,j)
    enddo
   enddo
   close(20)

   open(unit=21,file='histograms/histo_files')
   do iz=1,nzu
    read(21,'(a35)')names_hist(iz)
   enddo
   do iz=1,nzu
    open(unit=20,file='histograms/'//names_hist(iz),status='old')
    do i=1,nx
    do j=ny,1,-1
    read(20,*)hdist_i(i,j,iz)
    enddo
   enddo
   close(20)
   enddo

   do ix=1,nx
   do iy=1,ny
    htot=0.
    do iz=1,nzu
     htot=hdist_i(ix,iy,iz)+htot
    enddo
    if(htot.gt.0)then
    if(htot.ne.1.and.htot.ne.0)then
      write(89,'(2(1x,i3),11(1x,f8.4))')ix,iy,htot,(hdist_i(ix,iy,iz),iz=1,nzu)
    endif
      do iz=1,nzu
       hdist_i(ix,iy,iz)=hdist_i(ix,iy,iz)/htot
      enddo
    endif
   enddo
   enddo

  do ix=1,nx
   do iy=1,ny
    htot=0.
    do iz=1,nzu
     htot=hdist_i(ix,iy,iz)+htot
    enddo
    
   enddo
   enddo


   ip=0
   iutm=1
   do j=1,ny
   do i=1,nx
    xx=xmin+res*(i-1)
    yy=ymin+res*(j-1)
    if(iutm.eq.1)then
     call utm2ll(xx,yy,xlat,xlong,conv,iizone,ispher)
     xlong_i(i,j)=-xlong/3600.
     xlat_i(i,j)=xlat/3600.
    elseif(iutm.eq.0)then
     xlong_i(i,j)=xx
     xlat_i(i,j)=yy
    endif
!    write(*,*)ip
   enddo
   enddo
   
   write(*,*)'end'

   close(20)
  
     
     do ix=1,nx
     do iy=1,ny
      xlat_o(ix,iy)=latmin+dy*(iy-1)
      xlong_o(ix,iy)=longmin+dx*(ix-1)
     enddo
     enddo
   
 !    write(*,*)xlat_o(1,1),xlong_o(1,1),xlat_i(1,1),xlong_i(1,1)
    
   
     lambda_p_o(:,:)=0.
     lambda_b_o(:,:)=0.
     hmean_o(:,:)=0.
     lambda_u_o(:,:)=0.
     landuse_o(:,:)=0.

      write(*,*)'start looking for the closest',dx,dy
    
     do ix=1,nx
     do iy=1,ny
       imin=-1
       jmin=-1
       dmin=1000000.
       do i=max(ix-2,1),min(ix+2,nx)
       do j=max(iy-1,1),min(iy+2,ny) 
        latk=xlat_i(i,j)
        lonk=xlong_i(i,j)
        laty=xlat_o(ix,iy)
        lonx=xlong_o(ix,iy)
        a=cos((latk+laty)*pi/360.)
        dlat=rt*(latk-laty)*pi/180.
        dlon=rt*a*(lonk-lonx)*pi/180.
        dist=(dlat**2.+dlon**2.)**.5   
     !   write(*,*)dist,dlat,dlon,latk,lonk,laty,lonx           
        if(dist.lt.dmin)then
         dmin=dist
         imin=i
         jmin=j
        endif
       enddo
       enddo

       if(imin.gt.0.and.jmin.gt.0)then
         lambda_p_o(ix,iy)=lambda_p_i(imin,jmin)
         lambda_b_o(ix,iy)=lambda_b_i(imin,jmin)
         lambda_u_o(ix,iy)=lambda_u_i(imin,jmin)
	 landuse_o(ix,iy)=landuse_i(imin,jmin)
         hmean_o(ix,iy)=hmean_i(imin,jmin)
         do iz=1,nzu
          hdist_o(ix,iy,iz)=hdist_i(imin,jmin,iz)
         enddo
       endif
      enddo
      enddo
      open(unit=40,file='test_o.dat')
      do iy=1,ny
       write(40,'(300(1x,f5.2))')(hmean_o(ix,iy),ix=1,nx)
      enddo
      close(40)
      open(unit=40,file='test_i.dat')
      do iy=1,ny
       write(40,'(300(1x,f5.2))')(hmean_i(ix,iy),ix=1,nx)
      enddo
     

!!

   
         
!       endif


!

! CAREFUL : Don't change the routine and subroutine down below
!!!!!!!!!!!
    

   isigned = 1
   endian = 0!1
   wordsize = 4
   scalefactor = 0.01

   

   call write_geogrid((name_ufrac), len_trim(name_ufrac), lambda_u_o, nx, ny, nz, isigned, endian, scalefactor, wordsize)

   isigned = 1
   endian = 1
   wordsize = 1
   scalefactor = 1

   call write_geogrid((name_land), len_trim(name_land), landuse_o, nx, ny, nz, isigned, endian, scalefactor, wordsize)

   urb_param(:,:,:)=0.
   urb_param(:,:,91)=lambda_p_o(:,:)
   urb_param(:,:,95)=lambda_b_o(:,:)
   urb_param(:,:,94)=hmean_o(:,:)
   do iz=1,nzu
    urb_param(:,:,117+iz)=hdist_o(:,:,iz)
   enddo
!add more for the SLUCM if needed  
 
   isigned = 1
   endian = 0!1
   wordsize = 4
   scalefactor = 0.01

   call write_geogrid((name_urb_par), len_trim(name_urb_par), urb_param, nx, ny, 132, isigned, endian, scalefactor, wordsize)



   deallocate(ufrac_o)
   deallocate(landuse_o)
   

!   do ix=1,nx
!   do iy=1,ny
!    if(rarray(ix,iy).eq.31)write(*,*)'31',ix,iy
!   enddo
!   enddo


end program rd_wr_binary

     subroutine utm2ll(x,y,slat,slon,conv,iizone,ispher)
!
!     universal transverse mercator conversion
!
!     rewritten 6-16-83 by j.f. waananen using formulation
!     by j.p. snyder in bulletin 1532, pages 68-69
!
!     convert utm.f to convert from utm to lat lon (idir > 0)
!      2/5/2001  j. klinck
! 
      implicit none
!
      real*8 axis(19),bxis(19)
      real*8 radsec
      real*8 ak0,a,b,es
      real*8 x,y,slat,slon,conv,cm,phi,dlam,epri,en,t
      real*8 c,em,xx,yy,um,e1
      real*8 phi1,r,d,alam
!
      integer iizone,ispher,izone
      integer iutz
!
      data axis/6378206.4d0,6378249.145d0,6377397.155d0, &
       6378157.5d0,6378388.d0,6378135.d0,6377276.3452d0, & 
       6378145.d0,6378137.d0,6377563.396d0,6377304.063d0, & 
       6377341.89d0,6376896.0d0,6378155.0d0,6378160.d0, & 
       6378245.d0,6378270.d0,6378166.d0,6378150.d0/
!
      data bxis/6356583.8d0,6356514.86955d0,6356078.96284d0, & 
       6356772.2d0,6356911.94613d0,6356750.519915d0,6356075.4133d0, & 
       6356759.769356d0,6356752.31414d0,6356256.91d0,6356103.039d0, & 
       6356036.143d0,6355834.8467d0,6356773.3205d0,6356774.719d0, & 
       6356863.0188d0,6356794.343479d0,6356784.283666d0, & 
       6356768.337303d0/
!
      data ak0/0.9996d0/
!
      data radsec/206264.8062470964d0/
!
      a = axis(ispher)
      b = bxis(ispher)
      es= (a**2-b**2)/a**2
 
    

       
      izone = iizone
 
!     compute utm zone(izone) and central meridian in seconds for
!     geodetic to utm conversion where zone is not input.
!
      if (izone.eq.0 ) then
         write(*,*) ' *************   error exit from utm. '
         write(*,*) '  zone is not given for conversion from '
         write(*,*) '  x,y to lat,lon. '
         return
      endif
!
      if(iabs(izone).gt.30) then
         iutz = iabs(izone)-30
         cm=((iutz*6.0d0)-3.0d0)*(-3600.0d0)
      else
         iutz = 30-iabs(izone)
         cm=((iutz*6.0d0)+3.0d0)*3600.0d0
      endif
!
!---------------------------------------------------------------
!     inverse computation
!---------------------------------------------------------------
         yy = y
         if (izone.lt.0) yy = yy - 1.0d7
         xx = x - 5.0d5 
         em = yy/ak0
         um = em/(a*(1.d0-(es/4.d0)-(3.d0*es*es/6.4d1)- &
              (5.d0*es*es*es/2.56d2)))
         e1 = (1.d0-dsqrt(1.d0-es))/(1.d0+dsqrt(1.d0-es))
!
         phi1 = um+((3.d0*e1/2.d0)-(2.7d1*e1**3/3.2d1))*dsin(2.d0*um) + &
              ((2.1d1*e1*e1/1.6d1)-(5.5d1*e1**4/3.2d1))*dsin(4.d0*um) + &
              (1.51d2*e1**3/9.6d1)*dsin(6.d0*um)
!
         en = a/dsqrt(1.0d0-es*dsin(phi1)**2)
         t  = dtan(phi1)**2
         epri = es/(1.d0-es)
         c  = epri*dcos(phi1)**2
         r  = (a*(1.d0-es))/((1.d0-es*dsin(phi1)**2)**1.5d0)
         d  = xx/(en*ak0)   
!
         phi = phi1 - (en*dtan(phi1)/r) * ((d*d/2.d0) -  &
              (5.d0+3.d0*t+10.d0*c-4.d0*c*c-9.d0*epri)*d**4/2.4d1   &
            + (6.1d1+9.d1*t+2.98d2*c+4.5d1*t*t  &
              -2.52d2*epri-3.d0*c*c)*d**6/7.2d2)
!
         alam = (cm/radsec)-(d-(1.d0+2.d0*t+c)*d**3/6.d0 +  &
              (5.d0-2.d0*c+2.8d1*t -3.d0*c*c+8.d0*epri+2.4d1*t*t)  &
              *d**5/1.2d2)/dcos(phi1)
!

     
         slat = phi*radsec
         slon = alam*radsec
         dlam = -(slon-cm)/radsec
!---------------------------------------------------------------
!
!    test to see if within definition of utm projection
!
!  ---- latitude within 84 degrees
      if (dabs(slat).gt.302400) then
         write(99,*) ' *************   error exit from utm. '
         write(99,*) '   latitude value is poleward of 84 degrees'
         write(99,*) '   utm is not valid.'
         write(99,*) '   calculation has continued but values '
         write(99,*) '   may not be valid.'
         write(99,*) slat/3600.d0, slon/3600.d0
      endif
!  ---- delta longitude within 0.16 radians of central meridian
      if (dabs(dlam).gt.1.6d-1) then
         write(99,*) ' *************   error exit from utm. '
         write(99,*) '  d lon not within 0.16 radians of central merid.'
         write(99,*) '   calculation has continued but values '
         write(99,*) '   may not be valid.'
         write(99,*) slat/3600.d0, slon/3600.d0
      endif
!
!
!     compute convergence
!
          conv = dlam*(dsin(phi)+1.9587d-12*(dlam**2)  &
               *dsin(phi)*dcos(phi*phi))
          conv = conv*radsec
!     
      return
      end 

