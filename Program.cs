using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SimpleChatbot
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // In-memory training data
            var data = new[]
            {
                new ModelInput { Text = "Hello", Label = "Greeting" },
                new ModelInput { Text = "Hi", Label = "Greeting" },
                new ModelInput { Text = "How's the weather?", Label = "Weather" },
                new ModelInput { Text = "What's the weather like?", Label = "Weather" },
                new ModelInput { Text = "Goodbye", Label = "Goodbye" },
                new ModelInput { Text = "Bye", Label = "Goodbye" }
            };

            // Load the in-memory data as the training dataset
            var trainingData = mlContext.Data.LoadFromEnumerable(data);

            // Define the pipeline for transforming text into features and train the model
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(ModelInput.Text))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(ModelInput.Label)))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "Label"));

            // Train the model
            var model = pipeline.Fit(trainingData);

            // Create a prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            Console.WriteLine("Chatbot is ready! Type something (type 'exit' to end program).");

            // Chat loop
            while (true)
            {
                var userInput = Console.ReadLine();

                if (string.IsNullOrEmpty(userInput))
                {
                    continue;
                }

                if (userInput.ToLower() == "exit")
                    break;

                // Predict the user's input
                var prediction = predictionEngine.Predict(new ModelInput { Text = userInput });

                // Use PredictedLabel to respond to the user
                switch (prediction.PredictedLabel)
                {
                    case "Greeting":
                        Console.WriteLine("Hello! How can I help you today?");
                        break;
                    case "Weather":
                        Console.WriteLine("I don't know, go out and check.");
                        break;
                    case "Goodbye":
                        Console.WriteLine("Goodbye! Have a good day.");
                        return;
                    default:
                        Console.WriteLine("Sorry, I don't understand.");
                        break;
                }
            }
        }
    }

    // ModelInput class for the training data
    public class ModelInput
    {
        [LoadColumn(0)] // First column of the file 
        public string? Text { get; set; }

        [LoadColumn(1)] // Second column (Label)
        public string? Label { get; set; }
    }

    // ModelOutput class for predictions
    public class ModelOutput
    {
        public string? PredictedLabel { get; set; }
    }
}
