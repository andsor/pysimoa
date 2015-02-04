#!/bin/sh

while inotifywait -qq -r -e modify -e create -e move -e delete \
       --exclude '\.sw.?$' tests simoa
do
	clear
	py.test --cov=simoa tests
	sleep 1
done
