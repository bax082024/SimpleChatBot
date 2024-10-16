using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;

namespace SimpleChatbot
{
  class Program
  {
    static void Main(string[] args)
    {
      MLContext mlContext = new MLContext();

      var trainingData = mlContext.Data.LoadFromTextFile<ModelInput>("TrainingData.tsv", separatorChar: '\t', hasHeader: false);

      var pipeline = mlContext.Transforms.Text.FeaturizeText("Feature", nameof(ModelInput.Text))
        .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(ModelOutput.Prediction)))
        .Append(mlContext.Transforms.Concatenate("Feature", "Features"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("Prediction", "Label"))
        .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "Label"));

      var model = pipeline.Fit(trainingData);

      var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

      Console.WriteLine("Chatbot is ready! type something (type 'exit' to end program.)");

      

    }

  }

}
