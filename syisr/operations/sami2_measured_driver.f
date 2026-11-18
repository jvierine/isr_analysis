! Optional day-specific equatorial drift table. Values are m/s, like vd.
! Absent/outside/unqualified data leave the native empirical driver intact.
      subroutine measured_drift(hour,vd)
      real hour,vd,times(1000),speeds(1000),alpha
      integer flags(1000),n,k
      logical initialized,available
      save initialized,available,times,speeds,flags,n
      data initialized /.false./
      if (.not.initialized) then
        initialized=.true.
        inquire(file='measured-drift.inp',exist=available)
        if (available) then
          open(94,file='measured-drift.inp',status='old')
          read(94,*) n
          if (n.lt.2.or.n.gt.1000) stop 94
          do k=1,n
            read(94,*) times(k),speeds(k),flags(k)
            if(k.gt.1) then
              if(times(k).le.times(k-1)) stop 95
            endif
          enddo
          close(94)
        endif
      endif
      if (.not.available) return
      if(hour.lt.times(1).or.hour.gt.times(n)) return
      do k=1,n-1
        if(hour.ge.times(k).and.hour.le.times(k+1)) then
          if(flags(k).eq.1.and.flags(k+1).eq.1) then
            alpha=(hour-times(k))/(times(k+1)-times(k))
            vd=(1.-alpha)*speeds(k)+alpha*speeds(k+1)
          endif
          return
        endif
      enddo
      end
