using System.Collections.Generic;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;

public class LanguageInfo
{    
    /// <summary>        
    ///     <para>Ordered Descending by Freq</para>      
    /// </summary>
    public List<Word> Words = null!;

    /// <summary>
    ///     [WordIndex1, WordIndex2] Words energy matrix.
    /// </summary>    
    public MatrixFloat WordsDistancesOldMatrix = null!;

    public Clusterization_AlgorithmData Clusterization_AlgorithmData = null!;

    public ProjectionOptimization_AlgorithmData ProjectionOptimization_AlgorithmData = null!;

    public DiscreteVectorsAndMatrices DiscreteVectorsAndMatrices = null!;
}