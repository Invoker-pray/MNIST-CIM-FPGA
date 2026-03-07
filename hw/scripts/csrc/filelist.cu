PIC_LD=ld

ARCHIVE_OBJS=
ARCHIVE_OBJS += _24209_archive_1.so
_24209_archive_1.so : archive.8/_24209_archive_1.a
	@$(AR) -s $<
	@$(PIC_LD) -shared  -Bsymbolic  -o .//../../sim/psum_accum_simv.daidir//_24209_archive_1.so --whole-archive $< --no-whole-archive
	@rm -f $@
	@ln -sf .//../../sim/psum_accum_simv.daidir//_24209_archive_1.so $@


ARCHIVE_OBJS += _prev_archive_1.so
_prev_archive_1.so : archive.8/_prev_archive_1.a
	@$(AR) -s $<
	@$(PIC_LD) -shared  -Bsymbolic  -o .//../../sim/psum_accum_simv.daidir//_prev_archive_1.so --whole-archive $< --no-whole-archive
	@rm -f $@
	@ln -sf .//../../sim/psum_accum_simv.daidir//_prev_archive_1.so $@





O0_OBJS =

$(O0_OBJS) : %.o: %.c
	$(CC_CG) $(CFLAGS_O0) -c -o $@ $<
 

%.o: %.c
	$(CC_CG) $(CFLAGS_CG) -c -o $@ $<
CU_UDP_OBJS = \


CU_LVL_OBJS = \
SIM_l.o 

MAIN_OBJS = \
objs/amcQw_d.o 

CU_OBJS = $(MAIN_OBJS) $(ARCHIVE_OBJS) $(CU_UDP_OBJS) $(CU_LVL_OBJS)

