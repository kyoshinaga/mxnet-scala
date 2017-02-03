import AssemblyKeys._

assemblySettings

name := "mxnet-scala"

scalaVersion := "2.11.7"

version := "0.0"

mainClass in assembly := Some("mxnetScala.main.Run")

resolvers ++= Seq(
  "Sonatype Snapshots" at "http://oss.sonatype.org/content/repositories/snapshots",
  "Sonatype Releases" at "http://oss.sonatype.org/content/repositories/releases"
)

libraryDependencies ++= Seq(
  "junit" % "junit" % "4.11" % "test",
  "com.novocode" % "junit-interface" % "0.10-M4" % "test->default",
  "org.scalatest" %% "scalatest" % "2.2.6" % "test",
  "org.scala-lang.modules" %% "scala-xml" % "1.0.5",
  "org.json4s" %% "json4s-jackson" % "3.3.0",
//  "ml.dmlc.mxnet" % "libmxnet-scala-linux-x86_64-gpu" % "0.1.1"
  "ml.dmlc.mxnet" % "mxnet-core_2.10" % "0.1.1",
  "args4j" % "args4j" % "2.0.29"
)
