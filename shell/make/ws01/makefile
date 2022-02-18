buildfiles := $(shell echo build/{00..99}.txt)

build/%.txt:
	@[ -d build ] || mkdir build
	echo $* > $@

build: $(buildfiles)
	@# seems not work at all
	echo $<
all: build
clean:
	@[ -d build ] && rm -rf build
.PHONY: build all clean
