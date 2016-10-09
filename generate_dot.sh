#!/bin/bash
rm data.txt
hadoop fs -getmerge output data.txt
echo "digraph Giraph {" > giraph.dot
cat data.txt >> giraph.dot
echo "}" >> giraph.dot
circo giraph.dot -Tpng -ogiraph.png
