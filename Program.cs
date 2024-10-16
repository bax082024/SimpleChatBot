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

      //Load the TrainingData.tsv file
      var trainingData = mlContext.Data.LoadFromTextFile<ModelInput>("Data/TrainingData.tsv", separatorChar: '\t', hasHeader: false);

      //Note to self, defines how we transform text into futures(FeaturizeText). SdcaMaximumEntropy is a multiclass classification algorithm
      var pipeline = mlContext.Transforms.Text.FeaturizeText("Feature", nameof(ModelInput.Text))
        .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(ModelOutput.Prediction)))
        .Append(mlContext.Transforms.Concatenate("Feature", "Features"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("Prediction", "Label"))
        .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "Label"));

      var model = pipeline.Fit(trainingData);

      var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

      Console.WriteLine("Chatbot is ready! type something (type 'exit' to end program.)");

      while (true)
      {
        var userInput = Console.ReadLine();

        if (string.IsNullOrEmpty(userInput))
        {
          continue;
        }

        if (userInput.ToLower() == "exit")
          break;

        var prediction = predictionEngine.Predict(new ModelInput { Text = userInput});

        switch (prediction.Prediction)
        {
          case "Greeting":
          Console.WriteLine("Hello! How can i help you today?");
          break;
          case "Weather":
          Console.WriteLine("I dont know, go out and check.");
          break;
          case  "Goodbye":
          Console.WriteLine("Goodbye! have a good day.");
          return;
          default:
            Console.WriteLine("Sorry, i dont understand.");
            break;

        }
      }

    }

  }

}
