using Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;
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
    public MatrixFloat Temp_WordsDistancesOldMatrix = null!;

    public Clusterization_AlgorithmData Temp_Clusterization_AlgorithmData = null!;

    public ProjectionOptimization_AlgorithmData Temp_ProjectionOptimization_AlgorithmData = null!;

    public DiscreteVectorsAndMatrices Temp_DiscreteVectorsAndMatrices = null!;
}