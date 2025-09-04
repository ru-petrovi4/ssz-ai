using System.Collections.Generic;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public class LanguageInfo
{
    /// <summary>        
    ///     <para>Ordered Descending by Freq</para>      
    /// </summary>
    public List<Word> Words = null!;

    /// <summary>
    ///     [WordIndex1, WordIndex2] Words correlation matrix.
    /// </summary>
    /// <remarks>
    ///     Normalized vectors scalar product.       
    /// </remarks>
    public MatrixFloat ProxWordsOldMatrix = null!;

    public Clusterization_AlgorithmData Clusterization_AlgorithmData = null!;

    public ProjectionOptimization_AlgorithmData ProjectionOptimization_AlgorithmData = null!;
}
