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
            var d = WordsHelper.OldVectorLength_RU;
            var ruEmb = new MatrixFloat(d, LanguageInfo_RU.Words.Count);
            for (int j = 0; j < LanguageInfo_RU.Words.Count; j++)
            {
                var col = LanguageInfo_RU.Words[j];
                for (int i = 0; i < d; i++)
                {
                    ruEmb[i, j] = col.OldVectorNormalized[i];
                }
            }

            WordsHelper.InitializeWords_EN(LanguageInfo_EN, _loggersSet);
            d = WordsHelper.OldVectorLength_EN;
            var enEmb = new MatrixFloat(d, LanguageInfo_EN.Words.Count);
            for (int j = 0; j < LanguageInfo_EN.Words.Count; j++)
            {
                var col = LanguageInfo_EN.Words[j];
                for (int i = 0; i < d; i++)
                {
                    enEmb[i, j] = col.OldVectorNormalized[i];
                }
            }


            var align = new BilingualAlignment(_loggersSet, d: d, kCsls: 10, betaOrtho: 0.01f);            

            // Инициализация W случайной ортогональной матрицей (упрощённо — единичная)
            var W = new MatrixFloat(new[] { d, d });
            for (int i = 0; i < d; i++) W[i, i] = 1f;

            // (Опционально) Несколько шагов «ортогонализирующего» апдейта для стабилизации
            for (int t = 0; t < 5; t++) align.Orthonormalize(W);

            // Уточнение по Procrustes с синтетическим словарём (одна итерация; можно 2–3)
            for (int it = 0; it < 2; it++)
            {
                W = align.Refine(ruEmb, enEmb, W);
                align.Orthonormalize(W);
            }

            string fileName = "AdvancedEmbedding_LanguageInfo_W.bin";
            Helpers.SerializationHelper.SaveToFile(fileName, W, null);            
            _loggersSet.UserFriendlyLogger.LogInformation($"Saved");
        });            
    }

    public void CheckResult()
    {
        Task.Run(async () =>
        {
            WordsHelper.InitializeWords_RU(LanguageInfo_RU, _loggersSet);
            var d = WordsHelper.OldVectorLength_RU;
            var ruEmb = new MatrixFloat(d, LanguageInfo_RU.Words.Count);
            for (int j = 0; j < LanguageInfo_RU.Words.Count; j++)
            {
                var col = LanguageInfo_RU.Words[j];
                for (int i = 0; i < d; i++)
                {
                    ruEmb[i, j] = col.OldVectorNormalized[i];
                }
            }

            WordsHelper.InitializeWords_EN(LanguageInfo_EN, _loggersSet);
            d = WordsHelper.OldVectorLength_EN;
            var enEmb = new MatrixFloat(d, LanguageInfo_EN.Words.Count);
            for (int j = 0; j < LanguageInfo_EN.Words.Count; j++)
            {
                var col = LanguageInfo_EN.Words[j];
                for (int i = 0; i < d; i++)
                {
                    enEmb[i, j] = col.OldVectorNormalized[i];
                }
            }

            var align = new BilingualAlignment(_loggersSet, d: d, kCsls: 10, betaOrtho: 0.01f);

            var W = new MatrixFloat(new[] { d, d });
            string fileName = "AdvancedEmbedding_LanguageInfo_W.bin";
            Helpers.SerializationHelper.LoadFromFileIfExists(fileName, W, null);            

            var r1 = new float[300];
            var r2 = new float[300];
            var e1 = new float[300];
            var e2 = new float[300];
            for (int i = 50; i < 55; i++)
            {
                

                var ruW = ruEmb.GetColumn(i);
                //mapper.ApplyF12(ruW, r1);
                //mapper.ApplyF21(r1, r2);
                var dot = TensorPrimitives.CosineSimilarity(ruW, r2);
                // Перевод слова с индексом i
                int enIndex = align.Translate(enEmb, ruEmb, W, iSrc: i);
                if (enIndex < LanguageInfo_EN.Words.Count)
                    _loggersSet.UserFriendlyLogger.LogInformation($"RU: F21(F12(v)) cosine: {dot}; RU: {LanguageInfo_RU.Words[i].Name}; EN: {LanguageInfo_EN.Words[enIndex].Name}");
                else
                    _loggersSet.UserFriendlyLogger.LogInformation($"RU: F21(F12(v)) cosine: {dot}; EN: ---");

                var enW = enEmb.GetColumn(i);
                //mapper.ApplyF21(enW, e1);
                //mapper.ApplyF12(e1, e2);
                dot = TensorPrimitives.CosineSimilarity(enW, e2);
                // Перевод слова с индексом i
                int ruIndex = align.Translate(ruEmb, enEmb, W, iSrc: i);
                if (ruIndex < LanguageInfo_RU.Words.Count)
                    _loggersSet.UserFriendlyLogger.LogInformation($"EN: F12(F21(v)) cosine: {dot}; EN: {LanguageInfo_EN.Words[i].Name}; RU: {LanguageInfo_RU.Words[ruIndex].Name}");
                else
                    _loggersSet.UserFriendlyLogger.LogInformation($"EN: F12(F21(v)) cosine: {dot}; EN: ---");
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