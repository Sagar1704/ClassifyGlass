package main.scala

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.NaiveBayes

object NaiveBayesian {
  def main(args: Array[String]): Unit = {
        val conf = new SparkConf().setAppName("glassClassify")
        val sc = new SparkContext(conf)

        val input = sc.textFile(args(0))
        //val input = sc.textFile("S:\\Spring2016\\BigData\\Homeworks\\Homework3\\dataset\\glass.data")
        input.map(line => line.split(",").tail.reverse.mkString(",")).saveAsTextFile(".\\intermediate")

        val data = MLUtils.loadLabeledPoints(sc, ".\\intermediate")
        //        val data = MLUtils.loadLabeledPoints(sc, "S:\\Spring2016\\BigData\\Homeworks\\Homework3\\dataset\\glass.data")
            
        val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
        val (trainingData, testingData) = (splits(0), splits(1))
        
        val lambda = 1.0
        
        val model = NaiveBayes.train(data, lambda)
        
        val labelPredictions = testingData.map { point =>
            val prediction = model.predict(point.features)
            (point.label, prediction)
        }

        val accuracy = labelPredictions.filter(data => (data._1 == data._2)).count().toDouble / testingData.count()

        println("Naive Bayesian accuracy:: " + accuracy)
    }
}