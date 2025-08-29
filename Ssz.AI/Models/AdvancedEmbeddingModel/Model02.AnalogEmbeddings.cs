using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics.Tensors;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.Providers.LinearAlgebra;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Ssz.AI.Helpers;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public partial class Model02
{
    #region construction and destruction

    public Model02()
    {
        _loggersSet = new LoggersSet(NullLogger.Instance, new UserFriendlyLogger((l, id, s) => DebugWindow.Instance.AddLine(s)));
    }            

    #endregion

    #region public functions

    public const int OldVectorLength = 300;

    /// <summary>
    ///     RusVectores        
    /// </summary>
    public readonly LanguageInfo LanguageInfo_RU = new();

    /// <summary>
    ///     GloVe (Stanford)        
    /// </summary>
    public readonly LanguageInfo LanguageInfo_EN = new();        

    public void Initialize()
    {
        Task.Run(async () =>
        {
            WordsHelper.InitializeWords_RU(LanguageInfo_RU, _loggersSet);
            var dim = WordsHelper.OldVectorLength_RU;
            var ruEmb = new MatrixFloat(dim, LanguageInfo_RU.Words.Count);
            for (int j = 0; j < LanguageInfo_RU.Words.Count; j++)
            {
                var col = LanguageInfo_RU.Words[j];
                for (int i = 0; i < dim; i++)
                {
                    ruEmb[i, j] = col.OldVector[i];
                }
            }

            WordsHelper.InitializeWords_EN(LanguageInfo_EN, _loggersSet);
            dim = WordsHelper.OldVectorLength_EN;
            var enEmb = new MatrixFloat(dim, LanguageInfo_EN.Words.Count);
            for (int j = 0; j < LanguageInfo_EN.Words.Count; j++)
            {
                var col = LanguageInfo_EN.Words[j];
                for (int i = 0; i < dim; i++)
                {
                    enEmb[i, j] = col.OldVector[i];
                }
            }


            var mapper = new BilingualMapper(300, _loggersSet);
            mapper.NormalizeAndCenter(ruEmb);
            mapper.NormalizeAndCenter(enEmb);

            var opts = new BilingualMapper.TrainOptions
            {
                Epochs = 200,
                BatchSize = 2048,
                Lr = 0.01f,
                CycleWeight = 1.0f,
                CoralWeight = 1.0f,
                MeanWeight = 0.1f,
                OrthoWeight = 0.1f,
                CoupleWeight = 0.1f,
                OrthoRetraction = 0.01f,
                RetractionEvery = 10
            };

            mapper.Fit(ruEmb, enEmb, opts);


            string fileName = "AdvancedEmbedding_LanguageInfo_A.bin";
            Helpers.SerializationHelper.SaveToFile(fileName, mapper.W12, null);
            fileName = "AdvancedEmbedding_LanguageInfo_B.bin";
            Helpers.SerializationHelper.SaveToFile(fileName, mapper.W21, null);
            _loggersSet.UserFriendlyLogger.LogInformation($"Saved");
        });            
    }

    public void CheckResult()
    {
        Task.Run(async () =>
        {
            WordsHelper.InitializeWords_RU(LanguageInfo_RU, _loggersSet);
            var dim = WordsHelper.OldVectorLength_RU;
            var R = new MatrixFloat(dim, LanguageInfo_RU.Words.Count);
            for (int j = 0; j < LanguageInfo_RU.Words.Count; j++)
            {
                var col = LanguageInfo_RU.Words[j];
                for (int i = 0; i < dim; i++)
                {
                    R[i, j] = col.OldVector[i];
                }
            }

            WordsHelper.InitializeWords_EN(LanguageInfo_EN, _loggersSet);
            dim = WordsHelper.OldVectorLength_EN;
            var E = new MatrixFloat(dim, LanguageInfo_EN.Words.Count);
            for (int j = 0; j < LanguageInfo_EN.Words.Count; j++)
            {
                var col = LanguageInfo_EN.Words[j];
                for (int i = 0; i < dim; i++)
                {
                    E[i, j] = col.OldVector[i];
                }
            }

            MatrixFloat A = new();
            MatrixFloat B = new();
            string fileName = "AdvancedEmbedding_LanguageInfo_A.bin";
            Helpers.SerializationHelper.LoadFromFileIfExists(fileName, A, null);
            fileName = "AdvancedEmbedding_LanguageInfo_B.bin";
            Helpers.SerializationHelper.LoadFromFileIfExists(fileName, B, null);
            var r = new float[300];
            LinAlg.MatVec(A, LanguageInfo_RU.Words[50].OldVectorNormalized, r);
            LinAlg.MatVec(B, r, r);
            var distance = TensorPrimitives.Distance(r, LanguageInfo_RU.Words[50].OldVectorNormalized);
            _loggersSet.UserFriendlyLogger.LogInformation($"{distance}");
        });
    }

    public void Close()
    {            
    }            

    #endregion

    #region private functions

    

    #endregion

    #region private fields

    private readonly ILoggersSet _loggersSet; 

    #endregion

    //public class Constants
    //{
    //    public float lr = 1e-2;             // шаг обучения
    //    public int iters = 5000;             // число итераций
    //    public float wCycleRU = 1.0;        // вес цикла на RU: ||B A X - X||_F^2
    //    public float wCycleEN = 1.0;        // вес цикла на EN: ||A B Y - Y||_F^2
    //    public float wOrtho = 1e-3;         // вес ортогональной регуляризации: ||A^T A - I||_F^2 + ||B^T B - I||_F^2
    //    public float wGlobalInv = 1e-4;     // слабая глобальная инвертируемость: ||A B - I||_F^2 + ||B A - I||_F^2        
    //}
}