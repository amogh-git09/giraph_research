./gradlew build -x test
rm -f .aggreg*
rm -f aggre*
giraph build/libs/giraph.jar GiraphHelloWorld -vip src/main/resources/1 -vif SimpleTextVertexInputFormat -w 1 -ca giraph.SplitMasterWorker=false,giraph.logLevel=error
rm -rf output
