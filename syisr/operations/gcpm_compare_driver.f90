! Batch interface to external legacy GCPMv2.4, using its own GEO-to-SM transformation.
program gcpm_compare
implicit none
character(256) basepath
common /refpath/basepath
integer y,d,ios,itime(2)
real ut,lat,lon,z,kp,r,mlt,mlat,geo(3),sm(3),dens(4),pi
call get_command_argument(1,basepath)
pi=acos(-1.)
do
 read(*,*,iostat=ios) y,d,ut,lat,lon,z,kp
 if(ios/=0) exit
 itime=[y*1000+d,int(ut*3600000.)]
 r=1.+z/6371.
 geo=r*[cos(lat*pi/180.)*cos(lon*pi/180.),cos(lat*pi/180.)*sin(lon*pi/180.),sin(lat*pi/180.)]
 call geo_to_sm(itime,geo,sm)
 r=sqrt(sum(sm**2));mlt=modulo(atan2(-sm(2),-sm(1))*12./pi,24.)
 mlat=atan2(sm(3),sqrt(sm(1)**2+sm(2)**2))
 call gcpm_v24(itime,r,mlt,mlat,kp,dens)
 write(*,'(5ES23.14)') z,dens
enddo
end program
subroutine getlibpath(path,n)
implicit none
character(*) path
integer n
character(256) basepath
common /refpath/basepath
path=trim(basepath);n=len_trim(basepath)
end subroutine
