using Microsoft.ML.Data;

public class ModelInput
{
  [LoadColumn(0)] //first column of file
  public string? Text { get; set; }

  [LoadColumn(1)] // second column
  public string? Label { get; set; }
}

public class ModelOutput
{
  public string? PredictedLabel { get; set; }
}