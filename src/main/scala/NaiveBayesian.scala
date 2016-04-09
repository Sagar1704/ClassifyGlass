package main.scala

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object NaiveBayesian {
    def main(args: Array[String]): Unit = {
        val conf = new SparkConf().setAppName("glassClassify")
        val sc = new SparkContext(conf)

        if (args.length != 1) {
            println("Usage: [glass.data]")
        } else {
            val input = sc.textFile(args(0))
            //val input = sc.textFile("S:\\Spring2016\\BigData\\Homeworks\\Homework3\\dataset\\glass.data")

            val data = input.map { line =>
                val parts = line.split(',')
                LabeledPoint(parts(10).toDouble, Vectors.dense(parts.drop(1).reverse.drop(1).map(_.toDouble)))
            }

            val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
            val (trainingData, testingData) = (splits(0), splits(1))

            val lambda = 1.0
            val modelType = "multinomial"

            val model = NaiveBayes.train(trainingData, lambda, modelType)

            val labelPredictions = testingData.map { point =>
                val prediction = model.predict(point.features)
                (point.label, prediction)
            }

            val accuracy = labelPredictions.filter(data => (data._1 == data._2)).count().toDouble / testingData.count()

            println("Naive Bayesian accuracy:: " + accuracy)
        }
    }
}