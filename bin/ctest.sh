#!/bin/sh

py.test
sleep 1

while inotifywait -qq -r -e modify -e create -e move -e delete \
       --exclude '\.sw.?$' simoa
do
	clear
	py.test
	sleep 1
done
