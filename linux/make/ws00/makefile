demo_echo_comment:
	# echo a comment
	@# simple comment without echo
	echo "[line3] line without @ sign"
	@echo "[line4] line with @ sign"
demo_env:
	export foo=bar
	echo "foo=[${foo}]" #empty
	foo=bar; echo "foo=[$$foo]" #pass
	export foo=bar; echo "foo=[$$foo]" #pass
demo_special_var:
	echo $@
build:
	mkdir build
build/demo_depend_src.txt: build
	echo $@
	echo $(@D)
	echo $(@F)
	echo "hello world" > build/demo_depend_src.txt
build/demo_depend_dst.txt: build/demo_depend_src.txt
	cp build/demo_depend_src.txt build/demo_depend_dst.txt
all: build/demo_depend_dst.txt
clean:
	@[ -d build ] && rm -rf build
.PHONY: demo_echo_comment demo_env demo_special_var all clean
