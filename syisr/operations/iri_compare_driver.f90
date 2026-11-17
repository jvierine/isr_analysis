! Batch interface to external official IRI2020. Input: Y M D UT latitude longitude altitude.
program compare_iri
implicit none
logical jf(50)
integer y,m,d,ios
real ut,lat,lon,z,oarr(100),outf(20,1000)
call read_ig_rz
call readapf107
jf=.true.
jf(4:6)=.false.
jf(12)=.false.
jf(22)=.true. ! Ion fractions in percent of total ions, not renormalized H/He/O.
jf(23)=.false.
jf(28:30)=.false.
jf(33:35)=.false.
do
 read(*,*,iostat=ios) y,m,d,ut,lat,lon,z
 if(ios/=0) exit
 oarr=0.
 call IRI_SUB(jf,0,lat,lon,y,m*100+d,ut+25.,z,z,1.,outf,oarr)
 write(*,'(7ES23.14)') z,outf(1,1),outf(6,1)/100.,outf(7,1)/100.,outf(5,1)/100.,outf(11,1)/100.,outf(3,1)
enddo
end program
