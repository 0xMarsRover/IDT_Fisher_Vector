-include .build_release/DenseTrackStab.d

.build_release/DenseTrackStab.l: \
	.build_release/DenseTrackStab.d \

release/DenseTrackStab: .build_release/DenseTrackStab.l \
	.build_release/DenseTrackStab.o \

