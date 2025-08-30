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
            LinAlg.NormalizeAndCenter(ruEmb);

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
            LinAlg.NormalizeAndCenter(enEmb);


            var mapper = new BilingualMapper(300, _loggersSet);            
            var opts = new BilingualMapper.TrainOptions
            {
                Epochs = 100,
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
            var ruEmb = new MatrixFloat(dim, LanguageInfo_RU.Words.Count);
            for (int j = 0; j < LanguageInfo_RU.Words.Count; j++)
            {
                var col = LanguageInfo_RU.Words[j];
                for (int i = 0; i < dim; i++)
                {
                    ruEmb[i, j] = col.OldVector[i];
                }
            }
            LinAlg.NormalizeAndCenter(ruEmb);

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
            LinAlg.NormalizeAndCenter(enEmb);

            var mapper = new BilingualMapper(300, _loggersSet);
            string fileName = "AdvancedEmbedding_LanguageInfo_A.bin";
            Helpers.SerializationHelper.LoadFromFileIfExists(fileName, mapper.W12, null);
            fileName = "AdvancedEmbedding_LanguageInfo_B.bin";
            Helpers.SerializationHelper.LoadFromFileIfExists(fileName, mapper.W21, null);

            var r1 = new float[300];
            var r2 = new float[300];
            var e1 = new float[300];
            var e2 = new float[300];
            for (int i = 50; i < 100; i++)
            {
                var ruW = ruEmb.GetColumn(i);
                mapper.ApplyF12(ruW, r1);
                mapper.ApplyF21(r1, r2);
                var dot = TensorPrimitives.Dot(ruW, r2);
                var enIndex = LinAlg.NearestColumnIndex(enEmb, r1);
                _loggersSet.UserFriendlyLogger.LogInformation($"W21(W12(v)) dot v: {dot}; RU: {LanguageInfo_RU.Words[i].Name}; EN: {LanguageInfo_EN.Words[enIndex].Name}");

                var enW = enEmb.GetColumn(i);
                mapper.ApplyF12(enW, e1);
                mapper.ApplyF21(e1, e2);
                dot = TensorPrimitives.Dot(enW, e2);
                enIndex = LinAlg.NearestColumnIndex(enEmb, e1);
                _loggersSet.UserFriendlyLogger.LogInformation($"W21(W12(v)) dot v: {dot}; EN: {LanguageInfo_EN.Words[i].Name}; RU: {LanguageInfo_RU.Words[enIndex].Name}");
            }
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