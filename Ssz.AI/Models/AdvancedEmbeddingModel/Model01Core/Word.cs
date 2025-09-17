namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;

public class Word
{
    public Word()
    {
        OldVector = new float[Model01.Constants.OldVectorLength];
        OldVectorNormalized = new float[Model01.Constants.OldVectorLength];
        DiscreteVector_ToDisplay = new float[Model01.Constants.DiscreteVectorLength];
    }

    /// <summary>
    ///     Index in Words Array.
    ///     Index == 0: Empty word
    /// </summary>
    public int Index;

    public string Name = null!;

    public double Freq;

    /// <summary>
    ///     Initialized when Cortex is initialized.
    /// </summary>
    public Point Point = null!;

    /// <summary>
    ///     Original normalized vector (module 1).
    /// </summary>
    public readonly float[] OldVector;

    /// <summary>
    ///     Original normalized vector (module 1).
    /// </summary>
    public readonly float[] OldVectorNormalized;

    public float[]? DiscreteVector_ToDisplay;

    public bool Temp_Flag;
}