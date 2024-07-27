@echo off
start /B D:\kafka_2.13-3.5.1\bin\windows\zookeeper-server-start.bat D:\kafka_2.13-3.5.1\config\zookeeper.properties
timeout /T 10
start /B D:\kafka_2.13-3.5.1\bin\windows\kafka-server-start.bat D:\kafka_2.13-3.5.1\config\server.properties
