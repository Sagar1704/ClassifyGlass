package main.scala

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object DecisionTrees {
    def main(args: Array[String]): Unit = {
        val conf = new SparkConf().setAppName("glassClassify")
        val sc = new SparkContext(conf)

        val input = sc.textFile(args(0))
        //val input = sc.textFile("S:\\Spring2016\\BigData\\Homeworks\\Homework3\\dataset\\glass.data")

        val data = input.map { line =>
            val parts = line.split(',')
            LabeledPoint(parts(10).toDouble, Vectors.dense(parts.drop(1).reverse.drop(1).map(_.toDouble)))
        }

        val splits = data.randomSplit(Array(0.6, 0.4))
        val (trainingData, testingData) = (splits(0), splits(1))

        val numClasses = 8
        val categoricalFeaturesInfo = Map[Int, Int]()
        val impurity = "gini"
        val maxDepth = 5
        val maxBins = 32

        val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
            impurity, maxDepth, maxBins)

        val labelPredictions = testingData.map { point =>
            val prediction = model.predict(point.features)
            (point.label, prediction)
        }

        val accuracy = labelPredictions.filter(data => (data._1 == data._2)).count().toDouble / testingData.count()

        println("Decision Tree accuracy:: " + accuracy)
    }
}